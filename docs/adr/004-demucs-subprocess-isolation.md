# ADR-004: Subprocess Isolation for Demucs

**Status:** Accepted  
**Date:** 2026-04-06

## Context
Demucs (Meta's audio source separation) uses PyTorch with complex CUDA/model loading. Importing it directly into the tab-ripper process risks version conflicts (especially torch/torchaudio), GPU memory issues, and startup time overhead even when `--skip-separation` is used.

## Decision
Run Demucs as a subprocess via `python -m demucs`, communicating through file I/O (input audio file → output stem WAV files).

## Consequences
- **Pro:** Complete isolation — Demucs torch version can't conflict with our code
- **Pro:** No import overhead when separation is skipped
- **Pro:** Crash in Demucs doesn't crash tab-ripper
- **Con:** Subprocess output parsing is fragile (relies on directory structure conventions)
- **Con:** No programmatic access to intermediate results
- **Con:** Windows encoding issues required PYTHONUTF8 workaround
