"""Ingest Guitar-TECHS dataset for technique classification training.

Guitar-TECHS provides labelled technique examples with per-string MIDI
annotations from a Fishman Triple Play hexaphonic pickup.

Techniques: Bendings, Harmonics, PalmMute, PinchHarmonics, Vibrato
+ SingleNotes (normal picking baseline)

Each technique file has:
- DI audio (48kHz WAV, 32-bit float)
- 6-track MIDI (one track per string)

Usage:
    python -m training.ingest_guitar_techs [--data-home data/external/guitar-techs]
"""

import json
import logging
import zipfile
from pathlib import Path

import click
import librosa
import numpy as np
import pretty_midi

logger = logging.getLogger(__name__)

# Technique name mapping from filenames to our labels
TECHNIQUE_MAP = {
    "Bendings": "bend",
    "Harmonics": "harmonic",
    "PalmMute": "palm-mute",
    "PinchHarmonics": "pinch-harmonic",
    "Vibrato": "vibrato",
    "allsinglenotes": "normal",
}

# Standard tuning MIDI pitches
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]


def ingest_guitar_techs(
    data_home: str | Path = "data/external/guitar-techs",
    output_dir: str | Path = "data/processed/guitar-techs",
) -> dict:
    """Extract technique-labelled training data from Guitar-TECHS.

    For each technique file:
    1. Parse the MIDI to get per-string note events with timing
    2. Pair each note with its technique label
    3. Save as training-ready JSON + record audio segment paths

    Returns summary stats.
    """
    data_home = Path(data_home)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_notes = 0
    total_segments = 0
    technique_counts: dict[str, int] = {}

    # Process each zip file
    for zip_name in sorted(data_home.glob("P*_techniques.zip")) + sorted(data_home.glob("P*_singlenotes.zip")):
        logger.info("Processing %s...", zip_name.name)
        player = zip_name.name.split("_")[0]  # P1 or P2

        # Extract to temp directory
        extract_dir = data_home / zip_name.stem
        if not extract_dir.exists():
            logger.info("Extracting %s...", zip_name.name)
            with zipfile.ZipFile(zip_name, "r") as zf:
                zf.extractall(data_home)

        # Find the actual extracted directory (may be nested)
        actual_dir = extract_dir
        if not (actual_dir / "audio").exists():
            # Try one level deeper
            subdirs = [d for d in actual_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]
            if subdirs:
                actual_dir = subdirs[0]

        audio_dir = actual_dir / "audio" / "directinput"
        midi_dir = actual_dir / "midi"

        if not midi_dir.exists():
            logger.warning("No midi/ directory in %s", actual_dir)
            continue

        # Process each MIDI file (each = one technique)
        for midi_file in sorted(midi_dir.glob("*.mid")):
            # Extract technique from filename: midi_Vibrato.mid -> Vibrato
            parts = midi_file.stem.split("_", 1)
            tech_name = parts[1] if len(parts) > 1 else parts[0]
            technique = TECHNIQUE_MAP.get(tech_name, tech_name.lower())

            # Find matching DI audio
            di_audio = audio_dir / f"directinput_{tech_name}.wav"
            if not di_audio.exists():
                logger.warning("No DI audio for %s", midi_file.name)
                continue

            logger.info("  %s/%s: %s", player, tech_name, technique)

            # Parse MIDI — 6 tracks, one per string
            try:
                notes = _parse_midi_with_strings(midi_file)
            except Exception as e:
                logger.warning("  Failed to parse MIDI %s: %s", midi_file.name, e)
                continue

            if not notes:
                logger.warning("  No notes in %s", midi_file.name)
                continue

            # Set technique label on all notes
            for note in notes:
                note["technique"] = technique

            technique_counts[technique] = technique_counts.get(technique, 0) + len(notes)
            total_notes += len(notes)

            # Extract audio segments for each note
            segments_dir = output_dir / player / technique
            segments_dir.mkdir(parents=True, exist_ok=True)

            try:
                y, sr = librosa.load(str(di_audio), sr=22050, mono=True)
                audio_duration = len(y) / sr

                for i, note in enumerate(notes):
                    # Extract 500ms window around note onset
                    onset = note["start_s"]
                    if onset >= audio_duration:
                        continue

                    win_start = max(0, onset - 0.1)
                    win_end = min(audio_duration, onset + 0.4)
                    start_sample = int(win_start * sr)
                    end_sample = int(win_end * sr)
                    segment = y[start_sample:end_sample]

                    if len(segment) < sr * 0.1:  # skip very short segments
                        continue

                    # Save segment
                    seg_path = segments_dir / f"{tech_name}_{i:04d}.npy"
                    np.save(seg_path, segment)
                    note["segment_path"] = str(seg_path)
                    total_segments += 1

            except Exception as e:
                logger.warning("  Failed to extract audio: %s", e)

            # Save ground truth JSON
            gt = {
                "player": player,
                "technique": technique,
                "source_midi": str(midi_file),
                "source_audio": str(di_audio),
                "note_count": len(notes),
                "notes": notes,
            }
            gt_path = output_dir / player / f"{tech_name}.ground_truth.json"
            gt_path.parent.mkdir(parents=True, exist_ok=True)
            gt_path.write_text(json.dumps(gt, indent=2))

    logger.info(
        "Guitar-TECHS ingestion complete: %d notes, %d segments",
        total_notes,
        total_segments,
    )
    logger.info("Technique distribution: %s", technique_counts)

    return {
        "total_notes": total_notes,
        "total_segments": total_segments,
        "techniques": technique_counts,
    }


def _parse_midi_with_strings(midi_path: Path) -> list[dict]:
    """Parse a Guitar-TECHS MIDI file with per-string track assignment.

    Guitar-TECHS uses 6-track MIDI from Fishman Triple Play:
    Track order: String 1 (high E) through String 6 (low E).
    """
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []

    for track_idx, instrument in enumerate(midi.instruments):
        # Map track to string (0-based, 0=lowest)
        # Fishman Triple Play: Track 0 = String 1 (high E) = our string 5
        string_idx = 5 - track_idx
        if string_idx < 0:
            string_idx = 0  # clamp for extra tracks

        for note in instrument.notes:
            # Compute fret from pitch and string
            if string_idx < len(STANDARD_TUNING):
                fret = note.pitch - STANDARD_TUNING[string_idx]
            else:
                fret = note.pitch - 40  # fallback

            notes.append(
                {
                    "start_s": round(note.start, 4),
                    "end_s": round(note.end, 4),
                    "duration_s": round(note.end - note.start, 4),
                    "pitch_midi": note.pitch,
                    "string": string_idx,
                    "fret": max(0, fret),
                    "velocity": note.velocity,
                    "technique": "normal",  # will be overridden
                }
            )

    # Sort by time
    notes.sort(key=lambda n: n["start_s"])
    return notes


@click.command()
@click.option("--data-home", default="data/external/guitar-techs", help="Guitar-TECHS data directory")
@click.option("--output", "-o", default="data/processed/guitar-techs", help="Output directory")
@click.option("--verbose", "-v", is_flag=True)
def main(data_home: str, output: str, verbose: bool):
    """Ingest Guitar-TECHS dataset for technique training."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    stats = ingest_guitar_techs(data_home, output)
    click.echo("\nGuitar-TECHS ingestion complete:")
    click.echo(f"  Notes:    {stats['total_notes']}")
    click.echo(f"  Segments: {stats['total_segments']}")
    click.echo(f"  Techniques: {stats['techniques']}")


if __name__ == "__main__":
    main()
