"""Local technique detection using trained TechniqueNet model.

Replaces the Claude LLM-based technique analysis with a local CNN
that runs in <10ms per note. Trained on Guitar-TECHS dataset.

Usage in pipeline:
    events, annotations = detect_techniques(events, audio_path)
"""

import logging
from pathlib import Path

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Must match training/dataset.py TECHNIQUE_CLASSES exactly
TECHNIQUE_CLASSES = [
    "normal",
    "hammer-on",
    "pull-off",
    "slide",
    "bend",
    "tap",
    "harmonic",
    "palm-mute",
    "vibrato",
    "pinch-harmonic",
    "let-ring",
    "dead-note",
]

# Default model path
DEFAULT_MODEL_PATH = Path("models/technique_net.pt")

# Audio processing params (must match training)
SR = 22050
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
WINDOW_MS = 400.0
WINDOW_SAMPLES = int(WINDOW_MS / 1000 * SR)


def load_model(model_path: str | Path | None = None) -> torch.nn.Module | None:
    """Load the trained TechniqueNet model.

    Returns None if model file doesn't exist (graceful fallback).
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if not path.exists():
        logger.warning("TechniqueNet model not found at %s", path)
        return None

    from training.models import TechniqueNet

    model = TechniqueNet(num_classes=len(TECHNIQUE_CLASSES))
    model.load_state_dict(torch.load(str(path), map_location="cpu", weights_only=True))
    model.eval()
    logger.info("TechniqueNet loaded from %s (%d classes)", path, len(TECHNIQUE_CLASSES))
    return model


def detect_techniques(
    events: list,
    audio_path: str | Path,
    model: torch.nn.Module | None = None,
    model_path: str | Path | None = None,
    confidence_threshold: float = 0.5,
) -> tuple[list, list]:
    """Detect playing techniques for each note using the local CNN model.

    Args:
        events: TabEvent list from fret assignment.
        audio_path: Path to the audio file (for extracting mel spectrograms).
        model: Pre-loaded TechniqueNet model (or None to load from disk).
        model_path: Path to model weights file.
        confidence_threshold: Minimum confidence to assign a technique (else "normal").

    Returns:
        Tuple of (events_with_techniques, annotations).
        Same format as llm_analyzer.analyze_and_refine() for drop-in replacement.
    """
    from .llm_analyzer import TechniqueAnnotation

    if model is None:
        model = load_model(model_path)

    if model is None:
        logger.warning("No TechniqueNet model available — skipping technique detection")
        return events, []

    # Load audio
    audio_path = Path(audio_path)
    if not audio_path.exists():
        logger.warning("Audio file not found for technique detection: %s", audio_path)
        return events, []

    logger.info("Detecting techniques for %d events...", len(events))
    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    audio_duration = len(y) / SR

    annotations = []
    technique_counts: dict[str, int] = {}

    with torch.no_grad():
        for evt_idx, event in enumerate(events):
            if not event.notes:
                continue

            # Extract mel spectrogram window around event onset
            onset_s = event.time
            if onset_s >= audio_duration:
                continue

            onset_sample = int(onset_s * SR)
            start = max(0, onset_sample - int(0.1 * SR))  # 100ms before onset
            end = min(len(y), start + WINDOW_SAMPLES)
            segment = y[start:end]

            if len(segment) < WINDOW_SAMPLES // 2:
                continue

            # Pad if needed
            if len(segment) < WINDOW_SAMPLES:
                segment = np.pad(segment, (0, WINDOW_SAMPLES - len(segment)))

            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=segment.astype(np.float32),
                sr=SR,
                n_mels=N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Inference
            input_tensor = torch.from_numpy(mel_db).unsqueeze(0)  # (1, n_mels, n_frames)
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=-1)
            confidence, predicted = probs.max(dim=-1)

            predicted_class = TECHNIQUE_CLASSES[predicted.item()]
            conf = confidence.item()

            # Only assign non-normal techniques if confident enough
            if predicted_class != "normal" and conf >= confidence_threshold:
                technique = predicted_class
            else:
                technique = "normal"

            technique_counts[technique] = technique_counts.get(technique, 0) + 1

            # Create annotation (same format as llm_analyzer)
            if technique != "normal":
                primary_note_idx = max(
                    range(len(event.notes)),
                    key=lambda i: event.notes[i].midi_pitch,
                )
                annotations.append(
                    TechniqueAnnotation(
                        event_index=evt_idx,
                        note_index=primary_note_idx,
                        technique=technique,
                    )
                )

    logger.info(
        "Technique detection complete: %d annotations, distribution: %s",
        len(annotations),
        technique_counts,
    )
    return events, annotations
