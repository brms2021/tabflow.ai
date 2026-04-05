"""LLM-based guitar technique analysis and fret assignment refinement.

Uses Claude to identify guitar playing techniques (sweeps, pull-offs,
hammer-ons, position shifts) and refine fret assignments to match
natural guitar conventions.

Works in phrases (groups of notes separated by pauses) for focused analysis.
"""

import json
import os
from dataclasses import dataclass

import pretty_midi

from .tabber import TabEvent, TabNote, NoteEvent, STANDARD_TUNING, pitch_to_fret_options

# Notes closer than this are in the same phrase
PHRASE_PAUSE_THRESHOLD = 0.4  # seconds
# Max notes per LLM call (keeps prompts manageable)
MAX_NOTES_PER_PHRASE = 32


@dataclass
class TechniqueAnnotation:
    """Technique annotation for a note or group."""
    event_index: int       # index into the TabEvent list
    note_index: int        # index within the TabEvent's notes
    technique: str         # "hammer-on", "pull-off", "sweep", "slide", "bend", "normal"
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
            notes_desc.append(
                f"string={note.string+1} fret={note.fret} ({note_name}) dur={note.duration:.3f}s"
            )
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
    tuning_str = " ".join(
        f"{name}={pretty_midi.note_number_to_name(p)}"
        for name, p in zip(string_names, tuning)
    )
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

Respond with ONLY a raw JSON array (no markdown, no explanation). Each element:
{{"seq": 0, "technique": "normal", "suggested_string": null, "suggested_fret": null, "reason": null}}

Return one object per sequence number [0..N-1]. For multi-note events (chords), target the
highest pitched note. If no position change is needed, set suggested_string and suggested_fret to null."""


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
        string_names = STRING_NAMES[:len(tuning)]

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        print("[llm] No ANTHROPIC_API_KEY found — skipping LLM analysis")
        return events, []

    client = anthropic.Anthropic(api_key=key)

    # Work on a mutable copy
    refined = [
        TabEvent(time=e.time, notes=[
            TabNote(
                time=n.time, duration=n.duration, midi_pitch=n.midi_pitch,
                string=n.string, fret=n.fret, velocity=n.velocity,
            )
            for n in e.notes
        ])
        for e in events
    ]

    phrases = split_into_phrases(refined)
    if max_phrases is not None and len(phrases) > max_phrases:
        print(f"[llm] Capping to first {max_phrases} of {len(phrases)} phrases")
        phrases = phrases[:max_phrases]
    all_annotations: list[TechniqueAnnotation] = []

    print(f"[llm] Analyzing {len(phrases)} phrases ({len(events)} events) with {model}...")

    for phrase_num, phrase_indices in enumerate(phrases):
        phrase_events = refined
        phrase_lines = _events_to_prompt_lines(phrase_events, phrase_indices, tuning)
        prompt = _build_analysis_prompt(phrase_lines, tuning, string_names)

        try:
            response = client.messages.create(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip markdown code fences if present
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            # Find the JSON array bounds
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1:
                raise ValueError(f"No JSON array found in response: {raw[:200]!r}")
            raw = raw[start:end + 1]

            suggestions = json.loads(raw)

        except Exception as e:
            print(f"[llm] Phrase {phrase_num+1}: error — {e}")
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
                    0 <= new_string < len(tuning) and
                    0 <= new_fret <= 24 and
                    (tuning[new_string] + new_fret) == event.notes[primary_note_idx].midi_pitch
                ):
                    old_str = event.notes[primary_note_idx].string
                    old_fret = event.notes[primary_note_idx].fret
                    event.notes[primary_note_idx].string = new_string
                    event.notes[primary_note_idx].fret = new_fret
                    annotation.suggested_string = new_string
                    annotation.suggested_fret = new_fret
                    if old_str != new_string or old_fret != new_fret:
                        reason = suggestion.get("reason", "")
                        print(f"[llm] evt[{evt_idx}] {technique}: string {old_str+1}→{new_string+1}, fret {old_fret}→{new_fret} ({reason})")

            all_annotations.append(annotation)

    print(f"[llm] Analysis complete. {len(all_annotations)} annotations.")
    return refined, all_annotations
