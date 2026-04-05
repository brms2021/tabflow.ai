# tab-ripper

Isolate lead guitar from a full mix and generate ASCII tablature.

## Pipeline

1. **Source Separation** (Demucs) — splits audio into stems (vocals, drums, bass, guitar/other)
2. **Pitch Detection** (Basic Pitch) — transcribes isolated guitar audio to MIDI
3. **Tab Generation** — maps MIDI pitches to fretboard positions with playability heuristics

## Install

```bash
cd tab-ripper
pip install -e .
```

> Requires Python 3.10+. Demucs will download model weights (~300MB) on first run.

## Usage

```bash
# Full pipeline: separate + transcribe + tab
tab-ripper path/to/song.mp3

# Custom output directory
tab-ripper song.mp3 -o my-tabs/

# Use 6-stem model (dedicated guitar stem, more accurate)
tab-ripper song.mp3 --model htdemucs_6s

# Higher time resolution (more detailed tab, wider output)
tab-ripper song.mp3 --resolution 0.05

# Skip separation (if you already have an isolated guitar track)
tab-ripper isolated-guitar.wav --skip-separation

# Stricter note detection (fewer ghost notes)
tab-ripper song.mp3 --onset-threshold 0.7 --frame-threshold 0.5
```

## Output

```
tab-ripper-output/
  stems/          # Demucs separated stems (vocals, drums, bass, other)
  song.mid        # Transcribed MIDI
  song.tab.txt    # ASCII tablature
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output, -o` | `./tab-ripper-output` | Output directory |
| `--model, -m` | `htdemucs` | Demucs model (`htdemucs` or `htdemucs_6s`) |
| `--onset-threshold` | `0.5` | Note onset confidence (0-1) |
| `--frame-threshold` | `0.3` | Frame activation threshold (0-1) |
| `--resolution, -r` | `0.1` | Seconds per tab column |
| `--width, -w` | `80` | Tab line width |
| `--skip-separation` | off | Use input directly (skip Demucs) |
