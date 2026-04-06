"""Export tablature to Guitar Pro (.gp5) format.

Uses pyguitarpro to create Guitar Pro files that can be opened
in Guitar Pro, TuxGuitar, and other tab editors.
"""

import logging
from pathlib import Path

import guitarpro

from .tabber import STANDARD_TUNING, STRING_NAMES, TabEvent

logger = logging.getLogger(__name__)

# Map MIDI pitches to Guitar Pro string tuning values
# Guitar Pro uses the same MIDI numbering


def export_gp5(
    events: list[TabEvent],
    output_path: str | Path,
    title: str = "",
    tuning: list[int] = STANDARD_TUNING,
    string_names: list[str] | None = None,
    bpm: float = 120.0,
) -> Path:
    """Export tab events to a Guitar Pro 5 file.

    Args:
        events: List of TabEvents from assign_frets().
        output_path: Where to save the .gp5 file.
        title: Song title.
        tuning: Open string MIDI pitches (low to high).
        string_names: Display labels per string.
        bpm: Tempo in beats per minute.

    Returns:
        Path to the generated .gp5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if string_names is None:
        string_names = STRING_NAMES[: len(tuning)]

    song = guitarpro.models.Song()
    song.title = title
    song.tempo = guitarpro.models.Tempo(int(bpm))

    # Create guitar track
    track = song.tracks[0]
    track.name = "Guitar"
    track.channel.instrument = 25  # Steel guitar
    track.isPercussionTrack = False

    # Set tuning — Guitar Pro strings are numbered 1 (highest) to N (lowest)
    num_strings = len(tuning)
    track.strings = []
    for i in range(num_strings):
        # GP strings are high-to-low, our tuning is low-to-high
        string_idx = num_strings - 1 - i
        gp_string = guitarpro.models.GuitarString(i + 1, tuning[string_idx])
        track.strings.append(gp_string)

    # Pad to 7 strings if needed (GP expects at least 6)
    while len(track.strings) < 6:
        track.strings.append(guitarpro.models.GuitarString(len(track.strings) + 1, 0))

    # Group events into measures (4/4, 4 beats per measure at given BPM)
    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * 4

    # Create measures and populate with notes
    track.measures = []

    if not events:
        guitarpro.write(song, str(output_path))
        logger.info("GP5 saved to %s (empty)", output_path)
        return output_path

    total_time = max(e.time for e in events) + measure_duration
    num_measures = int(total_time / measure_duration) + 1

    # Build measure headers
    song.measureHeaders = []
    for m in range(num_measures):
        header = guitarpro.models.MeasureHeader()
        header.number = m + 1
        header.tempo = guitarpro.models.Tempo(int(bpm))
        header.timeSignature = guitarpro.models.TimeSignature(
            guitarpro.models.Duration(value=4),  # quarter note
            numerator=4,
        )
        song.measureHeaders.append(header)

    # Build measures for the track
    track.measures = []
    for m_idx, header in enumerate(song.measureHeaders):
        measure = guitarpro.models.Measure(header=header)
        voice = measure.voices[0]
        voice.beats = []

        m_start = m_idx * measure_duration
        m_end = m_start + measure_duration

        # Find events in this measure
        measure_events = [e for e in events if m_start <= e.time < m_end]

        if not measure_events:
            # Empty measure — add a whole rest
            beat = guitarpro.models.Beat(voice)
            beat.duration = guitarpro.models.Duration(value=1)
            beat.status = guitarpro.models.BeatStatus.rest
            voice.beats.append(beat)
        else:
            for event in measure_events:
                beat = guitarpro.models.Beat(voice)
                # Use eighth note duration as default
                beat.duration = guitarpro.models.Duration(value=8)
                beat.notes = []

                for note_data in event.notes:
                    note = guitarpro.models.Note(beat)
                    # GP strings: 1=highest, so convert from our 0=lowest
                    note.string = num_strings - note_data.string
                    note.value = note_data.fret
                    note.velocity = note_data.velocity
                    beat.notes.append(note)

                voice.beats.append(beat)

        track.measures.append(measure)

    guitarpro.write(song, str(output_path))
    logger.info("GP5 saved to %s (%d measures)", output_path, num_measures)
    return output_path
