"""MIDI-to-tablature converter.

Converts MIDI note events into guitar tablature by mapping pitches
to string/fret positions using playability heuristics.

Supports arbitrary tunings and string counts (6-string, 7-string, etc.).
Each string has up to max_fret frets (default 24).

Uses a Viterbi-style dynamic programming algorithm to find globally
optimal fret assignments that minimize hand movement and stay in
natural playing positions.
"""

import logging
from dataclasses import dataclass
from itertools import product

import pretty_midi

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning presets
# ---------------------------------------------------------------------------

TUNING_PRESETS = {
    "standard": ([40, 45, 50, 55, 59, 64], ["E", "A", "D", "G", "B", "e"]),
    "drop-d": ([38, 45, 50, 55, 59, 64], ["D", "A", "D", "G", "B", "e"]),
    "d": ([38, 43, 48, 53, 57, 62], ["D", "G", "C", "F", "A", "d"]),
    "7-string": ([35, 40, 45, 50, 55, 59, 64], ["B", "E", "A", "D", "G", "B", "e"]),
    "drop-a7": ([33, 40, 45, 50, 55, 59, 64], ["A", "E", "A", "D", "G", "B", "e"]),
}

STANDARD_TUNING = TUNING_PRESETS["standard"][0]
STRING_NAMES = TUNING_PRESETS["standard"][1]
MAX_FRET = 24


def parse_tuning(tuning_str: str) -> tuple[list[int], list[str]]:
    """Parse a tuning string into MIDI pitches and string names.

    Accepts either a preset name (e.g. '7-string') or a comma-separated
    list of note names (e.g. 'B1,E2,A2,D3,G3,B3,E4').
    """
    tuning_str = tuning_str.strip().lower()
    if tuning_str in TUNING_PRESETS:
        return TUNING_PRESETS[tuning_str]

    note_names = [n.strip() for n in tuning_str.split(",")]
    pitches = []
    labels = []
    for name in note_names:
        try:
            pitch = pretty_midi.note_name_to_number(name)
        except Exception:
            raise ValueError(
                f"Unknown tuning '{tuning_str}'. Use a preset "
                f"({', '.join(TUNING_PRESETS.keys())}) or comma-separated "
                f"note names like 'B1,E2,A2,D3,G3,B3,E4'."
            )
        pitches.append(pitch)
        labels.append(name.rstrip("0123456789"))
    if labels:
        labels[-1] = labels[-1].lower()
        for i in range(len(labels) - 1):
            labels[i] = labels[i].upper()
    return pitches, labels


def tuning_freq_range(tuning: list[int]) -> tuple[float, float]:
    """Compute the playable frequency range for a tuning.

    Returns (min_hz, max_hz) covering open lowest string to
    highest fret of highest string.
    """
    min_midi = min(tuning)
    max_midi = max(tuning) + MAX_FRET
    min_hz = 440.0 * (2.0 ** ((min_midi - 69) / 12.0))
    max_hz = 440.0 * (2.0 ** ((max_midi - 69) / 12.0))
    # Add a little margin
    return min_hz * 0.9, max_hz * 1.1


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NoteEvent:
    """A note with amplitude/confidence from transcription."""

    start: float
    end: float
    pitch: int
    amplitude: float
    velocity: int

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class TabNote:
    """A single note positioned on the fretboard."""

    time: float
    duration: float
    midi_pitch: int
    string: int  # 0 = lowest string
    fret: int  # 0-24
    velocity: int


@dataclass
class TabEvent:
    """A group of simultaneous notes (a 'column' in tab)."""

    time: float
    notes: list[TabNote]


# ---------------------------------------------------------------------------
# Note filtering
# ---------------------------------------------------------------------------


def filter_notes(
    note_events: list[tuple],
    tuning: list[int] = STANDARD_TUNING,
    amplitude_threshold: float = 0.4,
    min_duration_ms: float = 50.0,
    dedup_semitone_range: int = 1,
    dedup_time_ms: float = 30.0,
) -> list[NoteEvent]:
    """Filter raw transcription output to remove noise and artifacts.

    Args:
        note_events: Raw output from Basic Pitch predict():
            list of (start_s, end_s, pitch_midi, amplitude, pitch_bends).
        tuning: Open string MIDI pitches for guitar range filtering.
        amplitude_threshold: Drop notes below this confidence.
        min_duration_ms: Drop notes shorter than this (ms), unless in fast passage.
        dedup_semitone_range: Dedup notes within this pitch range.
        dedup_time_ms: Dedup notes within this time window (ms).

    Returns:
        Filtered list of NoteEvent objects.
    """
    num_strings = len(tuning)
    min_pitch = min(tuning)
    max_pitch = max(tuning) + MAX_FRET
    min_dur_s = min_duration_ms / 1000.0
    dedup_time_s = dedup_time_ms / 1000.0

    # Convert to NoteEvent objects
    notes = []
    for ev in note_events:
        start, end, pitch, amplitude = ev[0], ev[1], ev[2], ev[3]
        notes.append(
            NoteEvent(
                start=start,
                end=end,
                pitch=int(pitch),
                amplitude=float(amplitude),
                velocity=min(127, max(1, int(amplitude * 127))),
            )
        )

    original_count = len(notes)

    # 1. Amplitude filter
    notes = [n for n in notes if n.amplitude >= amplitude_threshold]
    after_amp = len(notes)

    # 2. Guitar range filter
    notes = [n for n in notes if min_pitch <= n.pitch <= max_pitch]
    after_range = len(notes)

    # 3. Short note filter (preserve fast passages)
    notes.sort(key=lambda n: n.start)
    keep = []
    for i, note in enumerate(notes):
        if note.duration >= min_dur_s:
            keep.append(note)
            continue
        # Check if part of a fast passage (neighbors within 100ms)
        has_neighbor = False
        for j in range(max(0, i - 3), min(len(notes), i + 4)):
            if j == i:
                continue
            if abs(notes[j].start - note.start) < 0.1:
                has_neighbor = True
                break
        if has_neighbor:
            keep.append(note)
    notes = keep
    after_short = len(notes)

    # 4. Deduplication (overlapping notes close in pitch)
    notes.sort(key=lambda n: (n.start, -n.amplitude))
    deduped = []
    for note in notes:
        is_dup = False
        for existing in deduped:
            if (
                abs(note.start - existing.start) <= dedup_time_s
                and abs(note.pitch - existing.pitch) <= dedup_semitone_range
            ):
                is_dup = True
                break
        if not is_dup:
            deduped.append(note)
    notes = deduped
    after_dedup = len(notes)

    # 5. Density cap (max simultaneous notes = number of strings)
    notes.sort(key=lambda n: n.start)
    SIMULTANEOUS_THRESHOLD = 0.03
    capped = []
    i = 0
    while i < len(notes):
        group = [notes[i]]
        j = i + 1
        while j < len(notes) and notes[j].start - group[0].start <= SIMULTANEOUS_THRESHOLD:
            group.append(notes[j])
            j += 1
        # Keep top N by amplitude
        group.sort(key=lambda n: -n.amplitude)
        capped.extend(group[:num_strings])
        i = j
    notes = capped

    notes.sort(key=lambda n: n.start)

    logger.info(
        "%d -> amp:%d range:%d short:%d dedup:%d cap:%d",
        original_count,
        after_amp,
        after_range,
        after_short,
        after_dedup,
        len(notes),
    )

    return notes


# ---------------------------------------------------------------------------
# Fret mapping helpers
# ---------------------------------------------------------------------------


def pitch_to_fret_options(pitch: int, tuning: list[int] = STANDARD_TUNING) -> list[tuple[int, int]]:
    """Return all valid (string, fret) pairs for a MIDI pitch."""
    options = []
    for string_idx, open_pitch in enumerate(tuning):
        fret = pitch - open_pitch
        if 0 <= fret <= MAX_FRET:
            options.append((string_idx, fret))
    return options


# ---------------------------------------------------------------------------
# Viterbi fret assignment
# ---------------------------------------------------------------------------


def _enumerate_configs(
    note_pitches: list[int],
    tuning: list[int],
    max_span: int = 5,
) -> list[tuple[tuple[int, int], ...]]:
    """Enumerate all valid (string, fret) configurations for a note group.

    Constraints:
    - No two notes on the same string
    - Fret span (excluding open strings) <= max_span
    """
    per_note_options = [pitch_to_fret_options(p, tuning) for p in note_pitches]

    # If any note has no options, skip it
    if any(len(opts) == 0 for opts in per_note_options):
        # Filter out unplayable notes and try again
        playable = [(i, opts) for i, opts in enumerate(per_note_options) if opts]
        if not playable:
            return []
        per_note_options = [opts for _, opts in playable]

    valid = []
    for combo in product(*per_note_options):
        # Check string uniqueness
        strings = [s for s, f in combo]
        if len(set(strings)) < len(strings):
            continue
        # Check fret span (open strings excluded)
        frets = [f for s, f in combo if f > 0]
        if frets and (max(frets) - min(frets)) > max_span:
            continue
        valid.append(combo)

    return valid


def _config_center(config: tuple[tuple[int, int], ...]) -> float:
    """Average fret position of a configuration (excluding open strings)."""
    frets = [f for _, f in config if f > 0]
    if not frets:
        return 0.0
    return sum(frets) / len(frets)


def _config_cost(config: tuple[tuple[int, int], ...]) -> float:
    """Internal cost of a single configuration (fret span penalty)."""
    frets = [f for _, f in config if f > 0]
    if not frets:
        return 0.0
    span = max(frets) - min(frets)
    if span <= 4:
        return 0.0
    elif span == 5:
        return 5.0
    else:
        return span * 10.0


def _transition_cost(
    prev_config: tuple[tuple[int, int], ...],
    curr_config: tuple[tuple[int, int], ...],
    time_gap: float,
) -> float:
    """Cost of transitioning between two configurations."""
    prev_center = _config_center(prev_config)
    curr_center = _config_center(curr_config)
    delta = abs(curr_center - prev_center)

    # Piecewise position shift cost
    if delta <= 2:
        shift_cost = delta * 1.0
    elif delta <= 4:
        shift_cost = 2.0 + (delta - 2) * 3.0
    elif delta <= 7:
        shift_cost = 8.0 + (delta - 4) * 6.0
    else:
        shift_cost = 26.0 + (delta - 7) * 15.0

    # Time discount: long gaps make shifts cheaper
    if time_gap > 0.5:
        shift_cost *= 0.3
    elif time_gap > 0.2:
        shift_cost *= 0.6

    # Fast passage: reward adjacent strings, penalize string skipping
    if time_gap < 0.15:
        prev_strings = {s for s, _ in prev_config}
        curr_strings = {s for s, _ in curr_config}
        min_string_gap = min(abs(cs - ps) for cs in curr_strings for ps in prev_strings)
        if min_string_gap <= 1:
            # Strong adjacent-string bonus for sweep-like passages
            shift_cost = max(0, shift_cost - 8.0)
            # Extra bonus if fret positions are close (sweep economy)
            prev_frets = [f for _, f in prev_config if f > 0]
            curr_frets = [f for _, f in curr_config if f > 0]
            if prev_frets and curr_frets:
                fret_delta = abs(min(curr_frets) - min(prev_frets))
                if fret_delta <= 2:
                    shift_cost = max(0, shift_cost - 5.0)
        elif min_string_gap >= 3:
            shift_cost += min_string_gap * 5.0

    return max(0, shift_cost)


def _group_notes(notes: list[NoteEvent], threshold: float = 0.03) -> list[list[NoteEvent]]:
    """Group simultaneous notes (within threshold seconds)."""
    if not notes:
        return []
    notes_sorted = sorted(notes, key=lambda n: n.start)
    groups: list[list[NoteEvent]] = []
    current_group = [notes_sorted[0]]

    for note in notes_sorted[1:]:
        if note.start - current_group[0].start <= threshold:
            current_group.append(note)
        else:
            groups.append(current_group)
            current_group = [note]
    groups.append(current_group)
    return groups


def assign_frets(
    notes: list[NoteEvent],
    tuning: list[int] = STANDARD_TUNING,
    max_hand_span: int = 5,
    beam_width: int = 200,
) -> list[TabEvent]:
    """Convert notes to fretboard positions using Viterbi DP.

    Finds the globally optimal sequence of fret assignments that
    minimizes hand movement and stays in natural playing positions.

    Args:
        notes: Filtered NoteEvent list from filter_notes().
        tuning: Open string MIDI pitches.
        max_hand_span: Max fret span allowed in a single hand position.
        beam_width: Max configurations to keep per group (for performance).

    Returns:
        List of TabEvents in chronological order.
    """
    if not notes:
        return []

    groups = _group_notes(notes)

    # For each group, enumerate valid configs
    group_configs: list[list[tuple[tuple[int, int], ...]]] = []
    group_times: list[float] = []
    valid_group_indices: list[int] = []

    for i, group in enumerate(groups):
        pitches = [n.pitch for n in group]
        configs = _enumerate_configs(pitches, tuning, max_hand_span)

        if not configs:
            continue

        # Beam pruning: keep top configs by internal cost
        if len(configs) > beam_width:
            configs.sort(key=_config_cost)
            configs = configs[:beam_width]

        group_configs.append(configs)
        group_times.append(group[0].start)
        valid_group_indices.append(i)

    if not group_configs:
        return []

    n_groups = len(group_configs)

    # DP: best_cost[c] = min cost to reach config c at current group
    # Use indices into group_configs[i] for efficiency

    # Initialize first group
    prev_costs = [_config_cost(c) for c in group_configs[0]]
    prev_backtrack = [None] * len(group_configs[0])

    # Store full backtrack table
    backtrack = [prev_backtrack]

    for i in range(1, n_groups):
        time_gap = group_times[i] - group_times[i - 1]
        curr_configs = group_configs[i]
        curr_costs = []
        curr_backtrack = []

        for c_idx, c_config in enumerate(curr_configs):
            c_internal = _config_cost(c_config)
            best_total = float("inf")
            best_prev = 0

            for p_idx, p_config in enumerate(group_configs[i - 1]):
                total = prev_costs[p_idx] + _transition_cost(p_config, c_config, time_gap) + c_internal
                if total < best_total:
                    best_total = total
                    best_prev = p_idx

            curr_costs.append(best_total)
            curr_backtrack.append(best_prev)

        prev_costs = curr_costs
        backtrack.append(curr_backtrack)

    # Backtrack to find optimal path
    best_final_idx = min(range(len(prev_costs)), key=lambda i: prev_costs[i])
    path_indices = [best_final_idx]
    for i in range(n_groups - 1, 0, -1):
        path_indices.append(backtrack[i][path_indices[-1]])
    path_indices.reverse()

    # Build TabEvents from the optimal path
    events = []
    for gi, config_idx in enumerate(path_indices):
        orig_group_idx = valid_group_indices[gi]
        group = groups[orig_group_idx]
        config = group_configs[gi][config_idx]

        tab_notes = []
        # Match config entries to notes (same order as pitches were given)
        playable_notes = [n for n in group if pitch_to_fret_options(n.pitch, tuning)]
        for note, (string, fret) in zip(playable_notes, config):
            tab_notes.append(
                TabNote(
                    time=note.start,
                    duration=note.duration,
                    midi_pitch=note.pitch,
                    string=string,
                    fret=fret,
                    velocity=note.velocity,
                )
            )

        if tab_notes:
            events.append(TabEvent(time=tab_notes[0].time, notes=tab_notes))

    return events


def assign_frets_greedy(
    notes: list[NoteEvent],
    tuning: list[int] = STANDARD_TUNING,
) -> list[TabEvent]:
    """Greedy fret assignment (legacy fallback).

    Kept for comparison. Processes each note independently,
    picking the closest fret to the previous note.
    """
    if not notes:
        return []

    groups = _group_notes(notes)
    last_fret = 5
    events = []

    for group in groups:
        used_strings: set[int] = set()
        tab_notes = []
        group.sort(key=lambda n: n.pitch)

        for note in group:
            options = pitch_to_fret_options(note.pitch, tuning)
            if not options:
                continue
            available = [(s, f) for s, f in options if s not in used_strings]
            if not available:
                available = options

            lf = last_fret  # capture for closure

            def score(option: tuple[int, int], _lf: int = lf) -> float:
                s, f = option
                return abs(f - _lf) + f * 0.3 + (10 if s in used_strings else 0)

            best_string, best_fret = min(available, key=score)
            tab_notes.append(
                TabNote(
                    time=note.start,
                    duration=note.duration,
                    midi_pitch=note.pitch,
                    string=best_string,
                    fret=best_fret,
                    velocity=note.velocity,
                )
            )
            used_strings.add(best_string)
            last_fret = best_fret

        if tab_notes:
            events.append(TabEvent(time=tab_notes[0].time, notes=tab_notes))

    return events


# ---------------------------------------------------------------------------
# ASCII rendering
# ---------------------------------------------------------------------------


def render_ascii_tab(
    events: list[TabEvent],
    tuning: list[int] = STANDARD_TUNING,
    string_names: list[str] | None = None,
    columns_per_line: int = 80,
    time_resolution: float = 0.1,
) -> str:
    """Render tab events as ASCII tablature."""
    if not events:
        return "(no notes detected)"

    if string_names is None:
        string_names = STRING_NAMES[: len(tuning)]

    max_time = max(e.time for e in events) + 1.0
    total_columns = int(max_time / time_resolution) + 1

    grid = [["-"] * total_columns for _ in range(len(tuning))]

    for event in events:
        col = int(event.time / time_resolution)
        if col >= total_columns:
            col = total_columns - 1

        for note in event.notes:
            fret_str = str(note.fret)
            for i, ch in enumerate(fret_str):
                c = col + i
                if c < total_columns:
                    row = len(tuning) - 1 - note.string
                    grid[row][c] = ch

    lines = []
    for chunk_start in range(0, total_columns, columns_per_line - 4):
        chunk_end = min(chunk_start + columns_per_line - 4, total_columns)
        for row, name in enumerate(reversed(string_names)):
            line = f"{name}|" + "".join(grid[len(tuning) - 1 - row][chunk_start:chunk_end]) + "|"
            lines.append(line)
        lines.append("")

    return "\n".join(lines)


def format_tab_header(
    audio_name: str,
    note_count: int,
    duration: float,
    tuning: list[int] = STANDARD_TUNING,
    string_names: list[str] | None = None,
) -> str:
    """Generate a header for the tablature output."""
    tuning_str = " ".join(pretty_midi.note_number_to_name(p) for p in tuning)
    if string_names is not None:
        tuning_str += f"  ({' '.join(string_names)})"
    return (
        f"{'=' * 60}\n"
        f"  TAB-RIPPER — {audio_name}\n"
        f"  Notes: {note_count} | Duration: {duration:.1f}s\n"
        f"  Tuning: {tuning_str}\n"
        f"{'=' * 60}\n"
    )
