"""Audio-to-MIDI transcription using Spotify's Basic Pitch.

Converts an audio signal (ideally an isolated instrument stem)
into MIDI note events with onset times, durations, and pitches.
"""

from pathlib import Path

import pretty_midi


def transcribe(
    audio_path: str | Path,
    onset_threshold: float = 0.6,
    frame_threshold: float = 0.4,
    minimum_note_length: float = 58.0,
    minimum_frequency: float | None = None,
    maximum_frequency: float | None = None,
    midi_output_path: str | Path | None = None,
) -> tuple[pretty_midi.PrettyMIDI, list[tuple]]:
    """Transcribe audio to MIDI using Basic Pitch.

    Args:
        audio_path: Path to the audio file (WAV preferred).
        onset_threshold: Confidence threshold for note onsets (0-1). Higher = fewer ghost notes.
        frame_threshold: Confidence threshold for active frames (0-1).
        minimum_note_length: Minimum note duration in milliseconds.
        minimum_frequency: Min frequency in Hz (filters below guitar range).
        maximum_frequency: Max frequency in Hz (filters above guitar range).
        midi_output_path: Optional path to save the MIDI file.

    Returns:
        Tuple of (PrettyMIDI object, list of note event tuples).
        Each note event is (start_s, end_s, pitch_midi, amplitude, pitch_bends).
    """
    from basic_pitch.inference import predict

    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[transcriber] Running Basic Pitch on {audio_path.name}...")

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

    note_count = sum(len(inst.notes) for inst in midi_data.instruments)
    print(f"[transcriber] Detected {note_count} notes")

    if midi_output_path:
        midi_output_path = Path(midi_output_path)
        midi_output_path.parent.mkdir(parents=True, exist_ok=True)
        midi_data.write(str(midi_output_path))
        print(f"[transcriber] MIDI saved to {midi_output_path}")

    return midi_data, note_events
