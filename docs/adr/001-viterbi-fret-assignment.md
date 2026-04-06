# ADR-001: Viterbi DP over Greedy Fret Assignment

**Status:** Accepted  
**Date:** 2026-04-06

## Context
Guitar tablature requires mapping MIDI pitches to (string, fret) positions. Most pitches can be played at multiple positions. A greedy algorithm that picks the closest fret to the previous note produces wildly jumping positions because it has no global view of the sequence.

## Decision
Use Viterbi dynamic programming with beam search (width=200) to find the globally optimal fret assignment sequence. Cost functions model hand position shifts (piecewise: 0-2 frets cheap, 8+ expensive), fret span within a chord, time-gap discounting, and adjacent-string bonuses for fast passages.

## Consequences
- **Pro:** Produces natural hand positions that match how guitarists actually play
- **Pro:** Sweep arpeggios stay in compact fret ranges across strings
- **Pro:** Beam pruning keeps computation tractable even for long pieces
- **Con:** More complex code (~150 lines vs ~20 for greedy)
- **Con:** Cost function tuning requires guitar domain knowledge
- Greedy fallback kept for edge cases where Viterbi fails
