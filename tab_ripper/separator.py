"""Audio source separation using Demucs (Meta Research).

Separates a mixed audio file into stems: vocals, drums, bass, other.
The 'other' stem captures guitar, keys, synths, etc.
With htdemucs_6s model, a dedicated 'guitar' stem is available.
"""

import subprocess
import sys
from pathlib import Path


def separate(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    model: str = "htdemucs",
    two_stems: str | None = None,
) -> dict[str, Path]:
    """Run Demucs source separation on an audio file.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Where to write stems. Defaults to ./separated/.
        model: Demucs model name. 'htdemucs' (4 stems) or 'htdemucs_6s' (6 stems incl guitar).
        two_stems: If set, only separate into this stem + remainder (e.g. 'vocals').

    Returns:
        Dict mapping stem name -> Path to the separated audio file.
    """
    audio_path = Path(audio_path).resolve()
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if output_dir is None:
        output_dir = audio_path.parent / "separated"
    output_dir = Path(output_dir).resolve()

    cmd = [
        sys.executable, "-m", "demucs",
        "--out", str(output_dir),
        "--name", model,
        str(audio_path),
    ]
    if two_stems:
        cmd.extend(["--two-stems", two_stems])

    print(f"[separator] Running Demucs ({model})...")
    print(f"[separator] Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[separator] STDERR: {result.stderr}")
        raise RuntimeError(f"Demucs failed (exit {result.returncode}): {result.stderr}")

    # Demucs outputs to: output_dir / model / track_name / stem.wav
    track_name = audio_path.stem
    stems_dir = output_dir / model / track_name

    if not stems_dir.exists():
        raise FileNotFoundError(
            f"Expected stems directory not found: {stems_dir}\n"
            f"Demucs stdout: {result.stdout}\n"
            f"Demucs stderr: {result.stderr}"
        )

    stems = {}
    for wav_file in stems_dir.glob("*.wav"):
        stems[wav_file.stem] = wav_file

    print(f"[separator] Stems extracted: {list(stems.keys())}")
    return stems


def get_guitar_stem(stems: dict[str, Path]) -> Path:
    """Pick the best stem for guitar content.

    Prefers 'guitar' (6-stem model), falls back to 'other' (4-stem model).
    """
    if "guitar" in stems:
        return stems["guitar"]
    if "other" in stems:
        print("[separator] No dedicated guitar stem — using 'other' (guitar + keys + etc.)")
        return stems["other"]
    raise KeyError(f"No guitar-like stem found. Available: {list(stems.keys())}")
