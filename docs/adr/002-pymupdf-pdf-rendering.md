# ADR-002: PyMuPDF for PDF Rendering

**Status:** Accepted (under review for AGPL compliance)  
**Date:** 2026-04-06

## Context
Need to generate professional PDF tablature output. Options considered: LaTeX/LilyPond (complex dependency), reportlab (BSD), PyMuPDF/fitz (AGPL), direct PostScript.

## Decision
Use PyMuPDF (fitz) for direct PDF drawing — string lines, fret numbers with white background boxes, adaptive time-proportional spacing, multi-page support.

## Consequences
- **Pro:** Already installed as a dependency (used for PDF reading elsewhere)
- **Pro:** Simple API for drawing lines, text, and rectangles
- **Pro:** No external tools needed (no LaTeX/LilyPond install)
- **Con:** AGPL license — requires open-sourcing or commercial license for SaaS distribution
- **Con:** Less typographic control than LaTeX
- Spike card exists to evaluate reportlab (BSD) as alternative
