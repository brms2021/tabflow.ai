"""Export tablature to Guitar Pro (.gp5) format.

Uses pyguitarpro to create Guitar Pro files that can be opened
in Guitar Pro, TuxGuitar, and other tab editors.
"""

import logging
from pathlib import Path

import guitarpro

from .tabber import STANDARD_TUNING, STRING_NAMES, TabEvent

logger = logging.getLogger(__name__)


def export_gp5(
    events: list[TabEvent],
    output_path: str | Path,
    title: str = "",
    tuning: list[int] = STANDARD_TUNING,
    string_names: list[str] | None = None,
    bpm: float = 120.0,
) -> Path:
    """Export tab events to a Guitar Pro 5 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if string_names is None:
        string_names = STRING_NAMES[: len(tuning)]

    num_strings = len(tuning)
    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * 4

    if not events:
        num_measures = 1
    else:
        total_time = max(e.time for e in events) + measure_duration
        num_measures = max(1, int(total_time / measure_duration) + 1)

    # Build a song from scratch using guitarpro.parse on a minimal template
    # This avoids the complex parent-reference requirements
    song = guitarpro.models.Song()
    song.title = title
    song.tempo = int(bpm)

    # Set tuning on the default track
    track = song.tracks[0]
    track.name = "Guitar"
    track.channel.instrument = 25

    # Set strings: GP is 1=highest, our tuning is 0=lowest
    track.strings = []
    for i in range(num_strings):
        string_idx = num_strings - 1 - i
        track.strings.append(guitarpro.models.GuitarString(i + 1, tuning[string_idx]))
    while len(track.strings) < 6:
        track.strings.append(guitarpro.models.GuitarString(len(track.strings) + 1, 0))

    # The default song has 1 measure header + 1 measure per track.
    # Add more measure headers as needed.
    while len(song.measureHeaders) < num_measures:
        song.measureHeaders.append(guitarpro.models.MeasureHeader())

    # Number headers and set tempo
    for i, header in enumerate(song.measureHeaders):
        header.number = i + 1
        header.tempo = int(bpm)

    # Ensure track has matching number of measures
    while len(track.measures) < num_measures:
        header = song.measureHeaders[len(track.measures)]
        measure = guitarpro.models.Measure(track, header)
        track.measures.append(measure)

    # Now populate measures with notes
    for m_idx in range(num_measures):
        measure = track.measures[m_idx]
        voice = measure.voices[0]

        m_start = m_idx * measure_duration
        m_end = m_start + measure_duration
        measure_events = [e for e in events if m_start <= e.time < m_end]

        if not measure_events:
            # Keep default rest beat
            continue

        # Clear default beats and add our notes
        voice.beats = []
        for event in measure_events:
            beat = guitarpro.models.Beat(voice)
            beat.duration = guitarpro.models.Duration(value=8)

            for note_data in event.notes:
                note = guitarpro.models.Note(beat)
                note.string = num_strings - note_data.string
                note.value = note_data.fret
                note.velocity = min(127, max(1, note_data.velocity))
                note.type = guitarpro.NoteType.normal
                beat.notes.append(note)

            voice.beats.append(beat)

    guitarpro.write(song, str(output_path))
    logger.info("GP5 saved to %s (%d measures)", output_path, num_measures)
    return output_path
