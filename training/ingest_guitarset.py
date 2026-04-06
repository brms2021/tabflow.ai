"""Ingest GuitarSet dataset into our training format.

GuitarSet provides 360 recordings with per-string annotations from
hexaphonic pickup. This converts JAMS annotations to our ground truth
JSON format for training.

Usage:
    python -m training.ingest_guitarset [--data-home data/external/guitarset]
"""

import json
import logging
from pathlib import Path

import click
import jams

logger = logging.getLogger(__name__)

# GuitarSet uses standard tuning
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
STRING_NAMES = ["E2", "A2", "D3", "G3", "B3", "E4"]


def ingest_guitarset(
    data_home: str | Path = "data/external/guitarset",
    output_dir: str | Path = "data/processed/guitarset",
) -> dict:
    """Convert GuitarSet JAMS annotations to our ground truth format.

    Returns summary stats.
    """
    data_home = Path(data_home)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    annotation_dir = data_home / "annotation"
    if not annotation_dir.exists():
        raise FileNotFoundError(
            f"GuitarSet annotations not found at {annotation_dir}. "
            "Run: python -c \"import mirdata; ds = mirdata.initialize('guitarset', "
            "data_home='data/external/guitarset'); ds.download()\""
        )

    jams_files = sorted(annotation_dir.glob("*.jams"))
    logger.info("Found %d JAMS files", len(jams_files))

    total_notes = 0
    total_tracks = 0
    technique_counts = {"normal": 0}

    for jams_path in jams_files:
        try:
            notes = _parse_jams(jams_path)
            if not notes:
                continue

            track_id = jams_path.stem
            gt = {
                "title": track_id,
                "artist": "GuitarSet",
                "bpm": 0,  # GuitarSet has variable tempo
                "tuning": STANDARD_TUNING,
                "tuning_name": "standard",
                "num_strings": 6,
                "source": "guitarset",
                "note_count": len(notes),
                "notes": notes,
            }

            out_path = output_dir / f"{track_id}.ground_truth.json"
            out_path.write_text(json.dumps(gt, indent=2))

            total_notes += len(notes)
            total_tracks += 1
            technique_counts["normal"] += len(notes)

            if total_tracks % 50 == 0:
                logger.info("Processed %d tracks (%d notes so far)", total_tracks, total_notes)

        except Exception as e:
            logger.warning("Failed to parse %s: %s", jams_path.name, e)

    logger.info(
        "GuitarSet ingestion complete: %d tracks, %d notes",
        total_tracks,
        total_notes,
    )
    return {
        "tracks": total_tracks,
        "notes": total_notes,
        "techniques": technique_counts,
    }


def _parse_jams(jams_path: Path) -> list[dict]:
    """Parse a JAMS file into our note format.

    GuitarSet JAMS files contain note_midi annotations per string.
    """
    jam = jams.load(str(jams_path))
    notes = []

    # GuitarSet stores per-string annotations in separate namespaces
    # Look for note_midi or pitch_midi annotations
    for ann in jam.annotations:
        namespace = ann.namespace

        if namespace == "note_midi":
            # This contains (time, duration, midi_pitch) per note
            # The annotation sandbox may contain string info
            string_idx = _get_string_from_annotation(ann)

            for obs in ann.data:
                midi_pitch = int(round(obs.value))
                start_s = float(obs.time)
                duration = float(obs.duration)

                # If we know the string, compute fret
                if string_idx is not None and string_idx < len(STANDARD_TUNING):
                    fret = midi_pitch - STANDARD_TUNING[string_idx]
                    if fret < 0 or fret > 24:
                        # Wrong string assignment, try to find correct one
                        string_idx_found, fret = _find_best_string_fret(midi_pitch)
                        if string_idx_found is not None:
                            string_idx = string_idx_found
                        else:
                            continue
                else:
                    string_idx_found, fret = _find_best_string_fret(midi_pitch)
                    if string_idx_found is not None:
                        string_idx = string_idx_found
                    else:
                        continue

                notes.append(
                    {
                        "bar": 0,
                        "beat": round(start_s, 4),
                        "string": string_idx,
                        "fret": fret,
                        "pitch_midi": midi_pitch,
                        "duration_beats": round(duration, 4),
                        "technique": "normal",
                        "start_s": round(start_s, 4),
                        "end_s": round(start_s + duration, 4),
                    }
                )

        elif namespace == "pitch_midi":
            # Continuous pitch contour — extract note onsets
            # Less useful for training but still has pitch info
            pass

    return notes


def _get_string_from_annotation(ann) -> int | None:
    """Try to extract string index from JAMS annotation metadata."""
    # GuitarSet annotations sometimes have string info in sandbox
    if hasattr(ann, "sandbox") and ann.sandbox:
        if hasattr(ann.sandbox, "string"):
            return int(ann.sandbox.string)

    # Check annotation_metadata
    if ann.annotation_metadata:
        if hasattr(ann.annotation_metadata, "data_source"):
            source = str(ann.annotation_metadata.data_source)
            # GuitarSet uses "hex_cqt_0" through "hex_cqt_5" for strings
            for i in range(6):
                if f"hex_cqt_{i}" in source or f"string_{i}" in source:
                    return i

    return None


def _find_best_string_fret(midi_pitch: int) -> tuple[int | None, int]:
    """Find the most natural (string, fret) for a given MIDI pitch."""
    best_string = None
    best_fret = 999

    for s, open_pitch in enumerate(STANDARD_TUNING):
        fret = midi_pitch - open_pitch
        if 0 <= fret <= 24:
            # Prefer lower fret positions
            if fret < best_fret:
                best_fret = fret
                best_string = s

    return best_string, best_fret


@click.command()
@click.option("--data-home", default="data/external/guitarset", help="GuitarSet data directory")
@click.option("--output", "-o", default="data/processed/guitarset", help="Output directory")
@click.option("--verbose", "-v", is_flag=True)
def main(data_home: str, output: str, verbose: bool):
    """Ingest GuitarSet into training format."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(name)s] %(message)s",
    )

    stats = ingest_guitarset(data_home, output)
    click.echo("\nGuitarSet ingestion complete:")
    click.echo(f"  Tracks: {stats['tracks']}")
    click.echo(f"  Notes:  {stats['notes']}")
    click.echo(f"  Output: {output}")


if __name__ == "__main__":
    main()
