"""GuitarSet benchmark evaluation pipeline.

Runs the tab-ripper pipeline on GuitarSet tracks and computes
note-level precision/recall/F1 against ground truth annotations.

Usage:
    python -m tab_ripper.benchmark [--backend basic-pitch] [--max-tracks 5]
"""

import json
import logging
from pathlib import Path

import click
import mir_eval
import numpy as np

logger = logging.getLogger(__name__)


def load_guitarset():
    """Load GuitarSet via mirdata."""
    import mirdata

    ds = mirdata.initialize("guitarset")
    ds.download()
    return ds


def evaluate_track(
    track,
    backend: str = "basic-pitch",
    tuning_pitches: list[int] | None = None,
) -> dict:
    """Evaluate a single GuitarSet track.

    Returns dict with note-level metrics.
    """
    from .tabber import STANDARD_TUNING
    from .transcriber import transcribe

    if tuning_pitches is None:
        tuning_pitches = STANDARD_TUNING

    # Get ground truth notes from GuitarSet
    ref_notes = track.notes
    if ref_notes is None:
        return {"error": "No note annotations"}

    ref_intervals = np.array([[n.start, n.end] for n in ref_notes])
    ref_pitches = np.array([n.pitch for n in ref_notes])

    if len(ref_intervals) == 0:
        return {"error": "Empty annotations"}

    # Run our pipeline
    audio_path = track.audio_mic_path or track.audio_mix_path
    if audio_path is None:
        return {"error": "No audio file"}

    try:
        midi_data, note_events = transcribe(audio_path, backend=backend)
    except Exception as e:
        return {"error": f"Transcription failed: {e}"}

    if not note_events:
        return {"error": "No notes detected"}

    # Extract estimated intervals and pitches
    est_intervals = np.array([[ev[0], ev[1]] for ev in note_events])
    est_pitches = np.array([ev[2] for ev in note_events])

    if len(est_intervals) == 0:
        return {"error": "No estimated notes"}

    # Compute note-level metrics using mir_eval
    precision, recall, f1, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals,
        ref_pitches,
        est_intervals,
        est_pitches,
        onset_tolerance=0.05,
        pitch_tolerance=50.0,  # 50 cents = half semitone
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "ref_notes": len(ref_pitches),
        "est_notes": len(est_pitches),
    }


@click.command("benchmark")
@click.option("--backend", default="basic-pitch", help="Transcription backend.")
@click.option("--max-tracks", default=5, type=int, help="Max tracks to evaluate (0=all).")
@click.option("--output", "-o", default="benchmark_results.json", help="Output JSON file.")
def main(backend: str, max_tracks: int, output: str):
    """Run GuitarSet benchmark evaluation."""
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    logger.info("Loading GuitarSet...")
    ds = load_guitarset()
    track_ids = list(ds.track_ids)

    if max_tracks > 0:
        track_ids = track_ids[:max_tracks]

    logger.info("Evaluating %d tracks with backend=%s...", len(track_ids), backend)

    results = {}
    for i, tid in enumerate(track_ids):
        track = ds.track(tid)
        logger.info("[%d/%d] %s", i + 1, len(track_ids), tid)
        result = evaluate_track(track, backend=backend)
        results[tid] = result
        if "f1" in result:
            logger.info(
                "  P=%.3f R=%.3f F1=%.3f (%d ref, %d est)",
                result["precision"],
                result["recall"],
                result["f1"],
                result["ref_notes"],
                result["est_notes"],
            )
        else:
            logger.warning("  %s", result.get("error", "unknown error"))

    # Compute averages
    valid = [r for r in results.values() if "f1" in r]
    if valid:
        avg = {
            "precision": np.mean([r["precision"] for r in valid]),
            "recall": np.mean([r["recall"] for r in valid]),
            "f1": np.mean([r["f1"] for r in valid]),
            "tracks_evaluated": len(valid),
            "tracks_failed": len(results) - len(valid),
        }
        results["_average"] = {k: float(v) if isinstance(v, (float, np.floating)) else v for k, v in avg.items()}
        logger.info(
            "Average: P=%.3f R=%.3f F1=%.3f (%d tracks)", avg["precision"], avg["recall"], avg["f1"], len(valid)
        )

    # Save results
    Path(output).write_text(json.dumps(results, indent=2))
    logger.info("Results saved to %s", output)


if __name__ == "__main__":
    main()
