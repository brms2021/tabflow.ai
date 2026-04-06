"""Tempo/BPM detection and beat-aligned bar line generation.

Uses librosa for beat tracking and provides bar line timestamps
for rendering in ASCII and PDF tablature.
"""

import logging

import librosa
import numpy as np

logger = logging.getLogger(__name__)


def detect_tempo(
    audio_path: str,
    sr: int | None = None,
) -> tuple[float, np.ndarray]:
    """Detect BPM and beat positions from an audio file.

    Args:
        audio_path: Path to audio file.
        sr: Sample rate (None = librosa default 22050).

    Returns:
        Tuple of (bpm, beat_times_seconds).
    """
    logger.info("Loading audio for beat tracking...")
    y, sr_actual = librosa.load(str(audio_path), sr=sr, mono=True)

    logger.info("Detecting tempo...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_actual)

    # Convert beat frames to time in seconds
    beat_times = librosa.frames_to_time(beat_frames, sr=sr_actual)

    # tempo may be an array; extract scalar
    bpm = float(np.atleast_1d(tempo)[0])
    logger.info("Detected BPM: %.1f (%d beats)", bpm, len(beat_times))

    return bpm, beat_times


def generate_bar_lines(
    beat_times: np.ndarray,
    beats_per_bar: int = 4,
) -> list[float]:
    """Generate bar line timestamps from beat positions.

    Args:
        beat_times: Array of beat timestamps in seconds.
        beats_per_bar: Beats per measure (4 for 4/4, 3 for 3/4).

    Returns:
        List of bar line timestamps (every Nth beat).
    """
    bar_lines = []
    for i in range(0, len(beat_times), beats_per_bar):
        bar_lines.append(float(beat_times[i]))
    return bar_lines


def quantize_events(
    events: list,
    beat_times: np.ndarray,
    strength: float = 0.5,
) -> list:
    """Optionally snap event times toward the nearest beat.

    Args:
        events: List of TabEvent objects.
        beat_times: Array of beat timestamps.
        strength: 0.0 = no quantization, 1.0 = snap to beat.

    Returns:
        Events with adjusted times (non-destructive copy).
    """
    if strength <= 0 or len(beat_times) == 0:
        return events

    from .tabber import TabEvent, TabNote

    quantized = []
    for event in events:
        nearest_idx = np.argmin(np.abs(beat_times - event.time))
        nearest_beat = float(beat_times[nearest_idx])
        new_time = event.time + (nearest_beat - event.time) * strength

        new_notes = [
            TabNote(
                time=new_time,
                duration=n.duration,
                midi_pitch=n.midi_pitch,
                string=n.string,
                fret=n.fret,
                velocity=n.velocity,
            )
            for n in event.notes
        ]
        quantized.append(TabEvent(time=new_time, notes=new_notes))

    return quantized
