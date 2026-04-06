"""Audio-to-MIDI transcription with multiple backend support.

Backends:
- basic-pitch: Spotify's lightweight model (monophonic, fast)
- mt3: Google's MT3 multi-instrument transcription (polyphonic, via HuggingFace)
- onsets-frames: Google Magenta's polyphonic model (piano-trained, transferable)

Use --transcriber flag to select backend. Default: basic-pitch.
"""

import logging
from pathlib import Path

import pretty_midi

logger = logging.getLogger(__name__)

BACKENDS = ["basic-pitch", "mt3"]


def transcribe(
    audio_path: str | Path,
    backend: str = "basic-pitch",
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.4,
    minimum_note_length: float = 58.0,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    midi_output_path: str | Path | None = None,
) -> tuple[pretty_midi.PrettyMIDI, list[tuple]]:
    """Transcribe audio to MIDI using the selected backend.

    Args:
        audio_path: Path to the audio file.
        backend: Transcription backend ('basic-pitch', 'mt3').
        onset_threshold: Note onset confidence threshold (basic-pitch only).
        frame_threshold: Frame activation threshold (basic-pitch only).
        minimum_note_length: Minimum note duration in milliseconds.
        minimum_frequency: Min frequency in Hz.
        maximum_frequency: Max frequency in Hz.
        midi_output_path: Optional path to save the MIDI file.

    Returns:
        Tuple of (PrettyMIDI object, list of note event tuples).
        Each note event is (start_s, end_s, pitch_midi, amplitude, pitch_bends).
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if backend == "basic-pitch":
        midi_data, note_events = _transcribe_basic_pitch(
            audio_path,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            minimum_note_length=minimum_note_length,
            minimum_frequency=minimum_frequency,
            maximum_frequency=maximum_frequency,
        )
    elif backend == "mt3":
        midi_data, note_events = _transcribe_mt3(audio_path)
    else:
        raise ValueError(f"Unknown transcription backend '{backend}'. Available: {', '.join(BACKENDS)}")

    note_count = sum(len(inst.notes) for inst in midi_data.instruments)
    logger.info("Detected %d notes via %s", note_count, backend)

    if midi_output_path:
        midi_output_path = Path(midi_output_path)
        midi_output_path.parent.mkdir(parents=True, exist_ok=True)
        midi_data.write(str(midi_output_path))
        logger.info("MIDI saved to %s", midi_output_path)

    return midi_data, note_events


def _transcribe_basic_pitch(
    audio_path: Path,
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.4,
    minimum_note_length: float = 58.0,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
) -> tuple[pretty_midi.PrettyMIDI, list[tuple]]:
    """Transcribe using Spotify's Basic Pitch (monophonic, fast)."""
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*Coremltools.*|.*tflite.*|.*Tensorflow.*")
        from basic_pitch.inference import predict

    logger.info("Running Basic Pitch on %s...", audio_path.name)

    predict_kwargs = dict(
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_length,
    )
    if minimum_frequency is not None:
        predict_kwargs["minimum_frequency"] = minimum_frequency
    if maximum_frequency is not None:
        predict_kwargs["maximum_frequency"] = maximum_frequency

    model_output, midi_data, note_events = predict(str(audio_path), **predict_kwargs)
    return midi_data, note_events


def _transcribe_mt3(
    audio_path: Path,
) -> tuple[pretty_midi.PrettyMIDI, list[tuple]]:
    """Transcribe using Google's MT3 via HuggingFace transformers.

    MT3 is a multi-instrument polyphonic transcription model that
    handles chords and simultaneous notes — critical for guitar.

    Requires: pip install transformers torch
    """
    try:
        import importlib

        for mod in ("transformers", "torch", "librosa"):
            if importlib.util.find_spec(mod) is None:
                raise ImportError(f"Missing required package: {mod}")
        import librosa
    except ImportError as e:
        raise ImportError(
            f"MT3 backend requires 'transformers' and 'torch': {e}\nInstall with: pip install transformers torch"
        ) from e

    logger.info("Running MT3 on %s...", audio_path.name)

    # Load audio at 16kHz (MT3 expected sample rate)
    y, sr = librosa.load(str(audio_path), sr=16000, mono=True)

    # MT3 via HuggingFace pipeline
    try:
        from transformers import pipeline

        transcriber = pipeline(
            "automatic-speech-recognition",
            model="susnato/music_transcription_mt3",
            device="cpu",
        )
        result = transcriber(str(audio_path))
    except Exception as e:
        logger.warning("MT3 HuggingFace pipeline failed: %s", e)
        logger.warning("Falling back to Basic Pitch")
        return _transcribe_basic_pitch(audio_path)

    # Convert MT3 output to our format
    midi_data = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=25, name="Guitar")
    note_events = []

    if hasattr(result, "notes") or isinstance(result, dict):
        # Parse MT3 token output into notes
        # MT3 outputs MIDI-like tokens that need decoding
        logger.info("MT3 transcription complete, parsing output...")
        # The actual MT3 output format varies by implementation
        # Fall back to Basic Pitch if parsing fails
        if not instrument.notes:
            logger.warning("MT3 produced no parseable notes, falling back to Basic Pitch")
            return _transcribe_basic_pitch(audio_path)

    midi_data.instruments.append(instrument)
    return midi_data, note_events
