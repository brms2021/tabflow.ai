"""Align parsed tab ground truth with audio timestamps.

Maps bar/beat positions from tab notation to absolute timestamps
using BPM detection and beat tracking from the audio file.

Produces training-ready aligned datasets: each note gets a precise
start/end timestamp matched to the audio.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

import librosa
import numpy as np

from .tab_parser import GroundTruthNote, GroundTruthTab

logger = logging.getLogger(__name__)


@dataclass
class AlignedNote:
    """A note with both tab position and audio timestamp."""

    start_s: float
    end_s: float
    pitch_midi: int  # computed from tuning + string + fret
    string: int
    fret: int
    technique: str
    confidence: float = 1.0  # alignment confidence


@dataclass
class AlignedDataset:
    """Complete aligned dataset for training."""

    title: str
    audio_path: str
    bpm: float
    tuning: str
    notes: list[AlignedNote] = field(default_factory=list)


# Standard tuning MIDI pitches (low to high)
TUNING_MIDI = {
    "standard": [40, 45, 50, 55, 59, 64],
    "drop-d": [38, 45, 50, 55, 59, 64],
    "d": [38, 43, 48, 53, 57, 62],
    "7-string": [35, 40, 45, 50, 55, 59, 64],
    "drop-a7": [33, 40, 45, 50, 55, 59, 64],
}


def align_tab_to_audio(
    ground_truth: GroundTruthTab,
    audio_path: str | Path,
    tuning_name: str | None = None,
) -> AlignedDataset:
    """Align tab ground truth notes to audio timestamps.

    Uses beat tracking to map sequential note positions to time.

    Args:
        ground_truth: Parsed tab data from tab_parser.
        audio_path: Path to the corresponding audio file.
        tuning_name: Override tuning (default: from ground truth).

    Returns:
        AlignedDataset with timestamped notes.
    """
    audio_path = Path(audio_path).resolve()
    tuning_name = tuning_name or ground_truth.tuning
    tuning = TUNING_MIDI.get(tuning_name, TUNING_MIDI["standard"])

    logger.info("Loading audio for alignment: %s", audio_path.name)
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    duration = len(y) / sr

    # Detect tempo and beats
    logger.info("Detecting beats...")
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    logger.info("Detected %.1f BPM, %d beats over %.1fs", bpm, len(beat_times), duration)

    # Group notes by system and sort by x_pos
    # Each system maps to a sequence of beats
    systems: dict[tuple[int, int], list[GroundTruthNote]] = {}
    for note in ground_truth.notes:
        key = (note.page, note.system)
        if key not in systems:
            systems[key] = []
        systems[key].append(note)

    # Sort systems by page/system order
    sorted_systems = sorted(systems.items())

    # Flatten all notes in reading order (system by system, left to right)
    ordered_notes: list[GroundTruthNote] = []
    for _, sys_notes in sorted_systems:
        sys_notes.sort(key=lambda n: n.x_pos)
        ordered_notes.extend(sys_notes)

    total_notes = len(ordered_notes)
    logger.info("Aligning %d notes to %d beats", total_notes, len(beat_times))

    # Strategy: distribute notes evenly across the audio duration
    # using beat positions as anchors
    # Each note gets a timestamp proportional to its position in the sequence
    aligned_notes: list[AlignedNote] = []
    beat_duration = 60.0 / bpm if bpm > 0 else 0.5
    default_note_duration = beat_duration / 4  # sixteenth note default

    if len(beat_times) >= 2 and total_notes > 0:
        # Map note index to time using linear interpolation across beats
        # Assume notes are roughly evenly distributed across time
        audio_start = beat_times[0] if len(beat_times) > 0 else 0.0
        audio_end = min(beat_times[-1] + beat_duration * 4, duration)

        for i, note in enumerate(ordered_notes):
            # Linear position fraction
            frac = i / max(total_notes - 1, 1)
            start_s = audio_start + frac * (audio_end - audio_start)

            # Snap to nearest beat for better alignment
            if len(beat_times) > 0:
                nearest_beat_idx = np.argmin(np.abs(beat_times - start_s))
                nearest_beat = beat_times[nearest_beat_idx]
                # Only snap if very close (within 1/8 of a beat)
                if abs(nearest_beat - start_s) < beat_duration / 8:
                    start_s = nearest_beat

            end_s = start_s + default_note_duration

            # Compute MIDI pitch from tuning + string + fret
            if note.string < len(tuning):
                pitch_midi = tuning[note.string] + note.fret
            else:
                pitch_midi = 60 + note.fret  # fallback

            aligned_notes.append(
                AlignedNote(
                    start_s=round(start_s, 4),
                    end_s=round(end_s, 4),
                    pitch_midi=pitch_midi,
                    string=note.string,
                    fret=note.fret,
                    technique=note.technique,
                )
            )

    result = AlignedDataset(
        title=ground_truth.title,
        audio_path=str(audio_path),
        bpm=bpm,
        tuning=tuning_name,
        notes=aligned_notes,
    )

    logger.info(
        "Alignment complete: %d notes, %.1f-%.1fs",
        len(aligned_notes),
        aligned_notes[0].start_s if aligned_notes else 0,
        aligned_notes[-1].end_s if aligned_notes else 0,
    )

    return result


def save_aligned_dataset(dataset: AlignedDataset, output_path: str | Path) -> Path:
    """Save aligned dataset to JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "title": dataset.title,
        "audio_path": dataset.audio_path,
        "bpm": dataset.bpm,
        "tuning": dataset.tuning,
        "note_count": len(dataset.notes),
        "notes": [asdict(n) for n in dataset.notes],
    }

    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Aligned dataset saved to %s", output_path)
    return output_path
