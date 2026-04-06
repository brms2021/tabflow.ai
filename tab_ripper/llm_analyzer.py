"""LLM-based guitar technique analysis and fret assignment refinement.

Uses Claude to identify guitar playing techniques (sweeps, pull-offs,
hammer-ons, position shifts) and refine fret assignments to match
natural guitar conventions.

Works in phrases (groups of notes separated by pauses) for focused analysis.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass

import pretty_midi

from .tabber import STANDARD_TUNING, TabEvent, TabNote

logger = logging.getLogger(__name__)

# Notes closer than this are in the same phrase
PHRASE_PAUSE_THRESHOLD = 0.4  # seconds
# Max notes per LLM call (keeps prompts manageable)
MAX_NOTES_PER_PHRASE = 32


@dataclass
class TechniqueAnnotation:
    """Technique annotation for a note or group."""

    event_index: int  # index into the TabEvent list
    note_index: int  # index within the TabEvent's notes
    technique: str  # "hammer-on", "pull-off", "sweep", "slide", "bend", "normal"
    suggested_string: int | None = None
    suggested_fret: int | None = None


def split_into_phrases(events: list[TabEvent], pause_threshold: float = PHRASE_PAUSE_THRESHOLD) -> list[list[int]]:
    """Split tab events into musical phrases separated by pauses.

    Returns list of phrases, each phrase is a list of event indices.
    """
    if not events:
        return []

    phrases: list[list[int]] = [[0]]

    for i in range(1, len(events)):
        gap = events[i].time - events[i - 1].time
        if gap >= pause_threshold or len(phrases[-1]) >= MAX_NOTES_PER_PHRASE:
            phrases.append([])
        phrases[-1].append(i)

    return [p for p in phrases if p]


def _events_to_prompt_lines(events: list[TabEvent], indices: list[int], tuning: list[int]) -> list[str]:
    """Format a phrase of tab events as readable lines for the LLM prompt."""
    lines = []
    for seq_num, evt_idx in enumerate(indices):
        event = events[evt_idx]
        notes_desc = []
        for note in event.notes:
            note_name = pretty_midi.note_number_to_name(note.midi_pitch)
            notes_desc.append(f"string={note.string + 1} fret={note.fret} ({note_name}) dur={note.duration:.3f}s")
        time_since_prev = ""
        if seq_num > 0:
            prev_evt = events[indices[seq_num - 1]]
            gap = event.time - prev_evt.time
            time_since_prev = f" [+{gap:.3f}s]"
        lines.append(f"  [{seq_num}]{time_since_prev} t={event.time:.3f}s: {' | '.join(notes_desc)}")
    return lines


def _build_analysis_prompt(
    phrase_lines: list[str],
    tuning: list[int],
    string_names: list[str],
) -> str:
    tuning_str = " ".join(f"{name}={pretty_midi.note_number_to_name(p)}" for name, p in zip(string_names, tuning))
    num_strings = len(tuning)

    return f"""You are an expert guitarist analyzing a transcribed guitar phrase to improve tablature accuracy.

Guitar tuning (string 1=lowest, string {num_strings}=highest): {tuning_str}
Max fret: 24

Here is a phrase of transcribed notes. Each line shows: sequence number, time offset, time, and note positions.

{chr(10).join(phrase_lines)}

Your task:
1. Identify the playing technique for each note position:
   - "sweep": part of a sweep picking arpeggio (consecutive notes on adjacent strings, same direction)
   - "hammer-on": ascending legato note (quick succession, same or higher fret, adjacent string)
   - "pull-off": descending legato note (quick succession, lower fret, adjacent or same string)
   - "slide": note approached by sliding from previous fret
   - "bend": note bent up from a lower pitch
   - "normal": standard picked note

2. For sweep arpeggios specifically: identify the full sweep group and ensure all notes use
   the MOST NATURAL hand position — the position that keeps the fret span minimal across
   the involved strings. Economy of motion is key: avoid jumps of more than 2-3 frets
   between adjacent string notes in a sweep.

3. If any note has a clearly better (string, fret) assignment that:
   - Reduces hand movement from the previous note
   - Makes a sweep or legato run more ergonomic
   - Avoids unrealistic position jumps
   ...then suggest the correction.

IMPORTANT: Respond with ONLY a raw JSON array. No markdown fences. No explanation text.
Each element must be exactly this shape:
{{"seq": 0, "technique": "normal", "suggested_string": null, "suggested_fret": null, "reason": null}}

Return one object per sequence number [0..{len(phrase_lines) - 1}]. For multi-note events (chords), target the
highest pitched note. If no position change is needed, set suggested_string and suggested_fret to null.

Begin your response with [ and end with ]. Nothing else."""


def _parse_json_response(raw: str) -> list[dict] | None:
    """Extract a JSON array from an LLM response, handling various formats.

    Handles: raw JSON, markdown code fences, preamble text, and empty responses.
    Returns None if no valid JSON array can be extracted.
    """
    if not raw:
        return None

    # Strategy 1: Try direct parse (ideal case — model followed instructions)
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", raw, re.DOTALL)
    if fence_match:
        try:
            result = json.loads(fence_match.group(1).strip())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    # Strategy 3: Find [ ... ] bounds in the text
    start = raw.find("[")
    end = raw.rfind("]")
    if start != -1 and end > start:
        try:
            result = json.loads(raw[start : end + 1])
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return None


def analyze_and_refine(
    events: list[TabEvent],
    tuning: list[int] = STANDARD_TUNING,
    string_names: list[str] | None = None,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
    max_phrases: int | None = 20,
) -> tuple[list[TabEvent], list[TechniqueAnnotation]]:
    """Use Claude to analyze guitar techniques and refine fret assignments.

    Processes the tab events in phrases, asking Claude to identify techniques
    and suggest more natural fret positions.

    Args:
        events: Tab events from Viterbi fret assignment.
        tuning: Open string MIDI pitches.
        string_names: Display labels per string.
        api_key: Anthropic API key. Uses ANTHROPIC_API_KEY env var if not provided.
        model: Claude model to use.

    Returns:
        Tuple of (refined events, technique annotations).
    """
    import anthropic

    if string_names is None:
        from .tabber import STRING_NAMES

        string_names = STRING_NAMES[: len(tuning)]

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        logger.warning("No ANTHROPIC_API_KEY found — skipping LLM analysis")
        return events, []

    client = anthropic.Anthropic(api_key=key)

    # Work on a mutable copy
    refined = [
        TabEvent(
            time=e.time,
            notes=[
                TabNote(
                    time=n.time,
                    duration=n.duration,
                    midi_pitch=n.midi_pitch,
                    string=n.string,
                    fret=n.fret,
                    velocity=n.velocity,
                )
                for n in e.notes
            ],
        )
        for e in events
    ]

    phrases = split_into_phrases(refined)
    if max_phrases is not None and len(phrases) > max_phrases:
        logger.info("Capping to first %d of %d phrases", max_phrases, len(phrases))
        phrases = phrases[:max_phrases]
    all_annotations: list[TechniqueAnnotation] = []
    skipped = 0

    logger.info("Analyzing %d phrases (%d events) with %s...", len(phrases), len(events), model)

    for phrase_num, phrase_indices in enumerate(phrases):
        logger.info("Phrase %d/%d (%d notes)...", phrase_num + 1, len(phrases), len(phrase_indices))
        phrase_events = refined
        phrase_lines = _events_to_prompt_lines(phrase_events, phrase_indices, tuning)
        prompt = _build_analysis_prompt(phrase_lines, tuning, string_names)

        suggestions = None
        last_err = None
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": prompt}],
                )
                raw = response.content[0].text.strip()
                suggestions = _parse_json_response(raw)
                if suggestions is not None:
                    break
                last_err = "no valid JSON in response"
            except anthropic.RateLimitError as e:
                last_err = e
            except anthropic.InternalServerError as e:
                last_err = e
            except anthropic.APIConnectionError as e:
                last_err = e
            except Exception as e:
                # Non-retryable (auth errors, bad request, etc.)
                logger.error("Phrase %d/%d: error — %s", phrase_num + 1, len(phrases), e)
                skipped += 1
                break
            else:
                # _parse_json_response returned None, retry with backoff
                pass

            if attempt < 2:
                wait = 2**attempt  # 1s, 2s
                time.sleep(wait)
        else:
            if suggestions is None:
                logger.warning("Phrase %d/%d: skipped after 3 attempts — %s", phrase_num + 1, len(phrases), last_err)
                skipped += 1
                continue

        if suggestions is None:
            continue

        # Apply suggestions
        for suggestion in suggestions:
            seq = suggestion.get("seq")
            technique = suggestion.get("technique", "normal")
            new_string_1based = suggestion.get("suggested_string")
            new_fret = suggestion.get("suggested_fret")

            if seq is None or seq >= len(phrase_indices):
                continue

            evt_idx = phrase_indices[seq]
            event = refined[evt_idx]

            if not event.notes:
                continue

            # Annotate the primary note (highest pitched)
            primary_note_idx = max(range(len(event.notes)), key=lambda i: event.notes[i].midi_pitch)
            annotation = TechniqueAnnotation(
                event_index=evt_idx,
                note_index=primary_note_idx,
                technique=technique,
            )

            # Apply position suggestion if valid
            if new_string_1based is not None and new_fret is not None:
                new_string = new_string_1based - 1  # convert to 0-based
                if (
                    0 <= new_string < len(tuning)
                    and 0 <= new_fret <= 24
                    and (tuning[new_string] + new_fret) == event.notes[primary_note_idx].midi_pitch
                ):
                    old_str = event.notes[primary_note_idx].string
                    old_fret = event.notes[primary_note_idx].fret
                    event.notes[primary_note_idx].string = new_string
                    event.notes[primary_note_idx].fret = new_fret
                    annotation.suggested_string = new_string
                    annotation.suggested_fret = new_fret
                    if old_str != new_string or old_fret != new_fret:
                        reason = suggestion.get("reason", "")
                        logger.info(
                            "evt[%d] %s: string %d→%d, fret %d→%d (%s)",
                            evt_idx,
                            technique,
                            old_str + 1,
                            new_string + 1,
                            old_fret,
                            new_fret,
                            reason,
                        )

            all_annotations.append(annotation)

    if skipped:
        logger.warning("%d/%d phrases skipped due to errors", skipped, len(phrases))
    logger.info("Analysis complete. %d annotations from %d phrases.", len(all_annotations), len(phrases) - skipped)
    return refined, all_annotations
