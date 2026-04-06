# ADR-003: Per-Phrase LLM Analysis Architecture

**Status:** Accepted  
**Date:** 2026-04-06

## Context
LLM technique analysis needs to process potentially thousands of notes. Sending the entire piece in one prompt would exceed context limits and produce unreliable results. Need a chunking strategy.

## Decision
Split notes into musical phrases (pauses ≥0.4s or max 32 notes), send each phrase as an independent Claude API call. Cap at 20 phrases by default (`--llm-max-phrases`).

## Consequences
- **Pro:** Each prompt is focused and manageable (~32 notes)
- **Pro:** Phrase boundaries are musically meaningful (natural pauses)
- **Pro:** Failures on individual phrases don't block the whole analysis
- **Pro:** Phrase cap controls API cost
- **Con:** No cross-phrase context (can't detect position shifts spanning a pause)
- **Con:** Sequential API calls add latency (~2-5 min for 20 phrases)
- **Con:** API costs scale linearly with phrase count
