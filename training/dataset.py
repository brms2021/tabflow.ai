"""Training dataset loaders for guitar tablature models.

Provides PyTorch datasets for:
1. Pitch detection: CQT spectrogram -> multi-pitch activation
2. Fret assignment: MIDI note sequence -> (string, fret) sequence
3. Technique classification: mel-spectrogram segment -> technique label
"""

import json
import logging
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Standard tuning MIDI pitches
STANDARD_TUNING = [40, 45, 50, 55, 59, 64]
PITCH_MIN = 40  # E2
PITCH_MAX = 88  # E6 (fret 24 on high E)
NUM_PITCHES = PITCH_MAX - PITCH_MIN + 1  # 49 pitch bins


class PitchDetectionDataset(Dataset):
    """Dataset for training guitar pitch detection.

    Each sample: CQT spectrogram frame window -> multi-pitch binary activation.

    Input: (n_bins, context_frames) CQT magnitude
    Target: (NUM_PITCHES,) binary vector — 1 for each active pitch at center frame
    """

    def __init__(
        self,
        aligned_json_paths: list[str | Path],
        audio_paths: list[str | Path],
        sr: int = 22050,
        hop_length: int = 512,
        n_bins: int = 264,
        bins_per_octave: int = 36,
        context_frames: int = 9,
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.context_frames = context_frames
        self.half_ctx = context_frames // 2

        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._load_all(aligned_json_paths, audio_paths)

    def _load_all(self, json_paths: list, audio_paths: list) -> None:
        for json_path, audio_path in zip(json_paths, audio_paths):
            try:
                self._load_track(Path(json_path), Path(audio_path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", json_path, e)

    def _load_track(self, json_path: Path, audio_path: Path) -> None:
        # Load aligned data
        data = json.loads(json_path.read_text())
        notes = data["notes"]

        # Load audio and compute CQT
        y, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
        cqt = np.abs(
            librosa.cqt(
                y,
                sr=self.sr,
                hop_length=self.hop_length,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
            )
        )
        n_frames = cqt.shape[1]

        # Build frame-level pitch activation matrix
        pitch_matrix = np.zeros((n_frames, NUM_PITCHES), dtype=np.float32)
        for note in notes:
            start_frame = int(note["start_s"] * self.sr / self.hop_length)
            end_frame = int(note["end_s"] * self.sr / self.hop_length)
            pitch_idx = note["pitch_midi"] - PITCH_MIN
            if 0 <= pitch_idx < NUM_PITCHES:
                pitch_matrix[start_frame : min(end_frame, n_frames), pitch_idx] = 1.0

        # Generate samples: one per frame with context window
        for frame in range(self.half_ctx, n_frames - self.half_ctx):
            cqt_window = cqt[:, frame - self.half_ctx : frame + self.half_ctx + 1]
            target = pitch_matrix[frame]
            if cqt_window.shape[1] == self.context_frames:
                self.samples.append((cqt_window.astype(np.float32), target))

        logger.info("Loaded %s: %d frames, %d notes", audio_path.name, n_frames, len(notes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        cqt, target = self.samples[idx]
        return torch.from_numpy(cqt), torch.from_numpy(target)


class FretAssignmentDataset(Dataset):
    """Dataset for training neural fret assignment.

    Each sample: sequence of (pitch, onset_delta, duration) -> sequence of (string, fret).

    Input: (seq_len, 3) — pitch (0-127), onset delta (ms), duration (ms)
    Target: (seq_len, 2) — string (0-5), fret (0-24)
    """

    MAX_SEQ_LEN = 64

    def __init__(
        self,
        ground_truth_paths: list[str | Path],
        tuning: list[int] | None = None,
    ):
        self.tuning = tuning or STANDARD_TUNING
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        self._load_all(ground_truth_paths)

    def _load_all(self, paths: list) -> None:
        for path in paths:
            try:
                self._load_track(Path(path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)

    def _load_track(self, path: Path) -> None:
        data = json.loads(path.read_text())
        notes = data["notes"]

        if not notes:
            return

        # Chunk into sequences of MAX_SEQ_LEN
        for i in range(0, len(notes), self.MAX_SEQ_LEN):
            chunk = notes[i : i + self.MAX_SEQ_LEN]

            inputs = np.zeros((self.MAX_SEQ_LEN, 3), dtype=np.float32)
            targets = np.zeros((self.MAX_SEQ_LEN, 2), dtype=np.int64)

            prev_beat = 0.0
            for j, note in enumerate(chunk):
                pitch = note.get("pitch_midi", 60)
                beat = note.get("beat", 0.0)
                duration = note.get("duration_beats", 1.0)

                inputs[j, 0] = pitch / 127.0  # normalize pitch
                inputs[j, 1] = (beat - prev_beat) * 100  # onset delta in centbeats
                inputs[j, 2] = duration * 100  # duration in centbeats

                targets[j, 0] = note.get("string", 0)
                targets[j, 1] = note.get("fret", 0)
                prev_beat = beat

            self.samples.append((inputs, targets))

        logger.info("Loaded %s: %d notes -> %d sequences", path.name, len(notes), len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = self.samples[idx]
        return torch.from_numpy(inputs), torch.from_numpy(targets)


class TechniqueDataset(Dataset):
    """Dataset for training technique classification.

    Each sample: mel-spectrogram around note onset -> technique label.

    Input: (n_mels, n_frames) mel-spectrogram (200ms window)
    Target: technique class index

    Supports two data sources:
    1. Aligned JSON + audio pairs (from tab alignment pipeline)
    2. Guitar-TECHS pre-extracted .npy segments (from ingest_guitar_techs)
    """

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

    def __init__(
        self,
        aligned_json_paths: list[str | Path] | None = None,
        audio_paths: list[str | Path] | None = None,
        segment_dirs: list[str | Path] | None = None,
        sr: int = 22050,
        n_mels: int = 128,
        window_ms: float = 400.0,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.window_samples = int(window_ms / 1000 * sr)
        self.samples: list[tuple[np.ndarray, int]] = []

        # Load from aligned JSON + audio pairs
        if aligned_json_paths and audio_paths:
            self._load_all(aligned_json_paths, audio_paths)

        # Load from Guitar-TECHS pre-extracted segments
        if segment_dirs:
            self._load_segments(segment_dirs)

    def _load_all(self, json_paths: list, audio_paths: list) -> None:
        for json_path, audio_path in zip(json_paths, audio_paths):
            try:
                self._load_track(Path(json_path), Path(audio_path))
            except Exception as e:
                logger.warning("Failed to load %s: %s", json_path, e)

    def _load_track(self, json_path: Path, audio_path: Path) -> None:
        data = json.loads(json_path.read_text())
        notes = data["notes"]

        y, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)

        for note in notes:
            technique = note.get("technique", "normal")
            if technique not in self.TECHNIQUE_CLASSES:
                technique = "normal"
            label = self.TECHNIQUE_CLASSES.index(technique)

            # Extract window around onset
            onset_sample = int(note["start_s"] * self.sr)
            half_win = self.window_samples // 2
            start = max(0, onset_sample - half_win)
            end = min(len(y), onset_sample + half_win)

            segment = y[start:end]
            if len(segment) < self.window_samples:
                segment = np.pad(segment, (0, self.window_samples - len(segment)))

            # Compute mel spectrogram
            mel = librosa.feature.melspectrogram(
                y=segment,
                sr=self.sr,
                n_mels=self.n_mels,
                n_fft=1024,
                hop_length=256,
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)

            self.samples.append((mel_db.astype(np.float32), label))

    def _load_segments(self, segment_dirs: list[str | Path]) -> None:
        """Load pre-extracted .npy audio segments from Guitar-TECHS.

        Directory structure: {dir}/{player}/{technique}/*.npy
        """
        for seg_dir in segment_dirs:
            seg_dir = Path(seg_dir)
            if not seg_dir.exists():
                logger.warning("Segment dir not found: %s", seg_dir)
                continue

            for technique_dir in sorted(seg_dir.rglob("*")):
                if not technique_dir.is_dir():
                    continue

                technique = technique_dir.name
                if technique not in self.TECHNIQUE_CLASSES:
                    continue

                npy_files = sorted(technique_dir.glob("*.npy"))
                loaded = 0
                for npy_path in npy_files:
                    try:
                        segment = np.load(npy_path)
                        label = self.TECHNIQUE_CLASSES.index(technique)

                        # Pad/trim to standard window size
                        if len(segment) < self.window_samples:
                            segment = np.pad(segment, (0, self.window_samples - len(segment)))
                        else:
                            segment = segment[: self.window_samples]

                        # Compute mel spectrogram
                        mel = librosa.feature.melspectrogram(
                            y=segment.astype(np.float32),
                            sr=self.sr,
                            n_mels=self.n_mels,
                            n_fft=1024,
                            hop_length=256,
                        )
                        mel_db = librosa.power_to_db(mel, ref=np.max)
                        self.samples.append((mel_db.astype(np.float32), label))
                        loaded += 1
                    except Exception as e:
                        logger.debug("Failed to load %s: %s", npy_path.name, e)

                if loaded > 0:
                    logger.info("Loaded %d %s segments from %s", loaded, technique, technique_dir)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        mel, label = self.samples[idx]
        return torch.from_numpy(mel), label
