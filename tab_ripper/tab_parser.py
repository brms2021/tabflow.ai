"""Parse guitar tablature PDFs to extract ground truth annotations.

Extracts structured (string, fret, x_position) data from PDF tablature
by detecting tab string lines and mapping fret number positions to strings.

Outputs standardised JSON for training data alignment.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


@dataclass
class TabSystem:
    """A single tab system (one row of 6 string lines across the page)."""

    string_ys: list[float]  # Y positions of each string (high to low, top to bottom)
    x_start: float
    x_end: float
    page: int
    system_idx: int


@dataclass
class GroundTruthNote:
    """A single note extracted from tablature."""

    page: int
    system: int
    string: int  # 0=lowest (E), 5=highest (e)
    fret: int  # 0-24
    x_pos: float  # horizontal position in points
    technique: str = "normal"  # h, p, s, b, T, etc.


@dataclass
class GroundTruthTab:
    """Complete parsed tablature ground truth."""

    title: str = ""
    tuning: str = "standard"
    bpm: float = 120.0
    notes: list[GroundTruthNote] = field(default_factory=list)
    num_strings: int = 6


def parse_tab_pdf(
    pdf_path: str | Path,
    num_strings: int = 6,
    string_spacing_tolerance: float = 2.0,
) -> GroundTruthTab:
    """Parse a tablature PDF into structured ground truth.

    Args:
        pdf_path: Path to the PDF file.
        num_strings: Expected number of strings (6 or 7).
        string_spacing_tolerance: Y tolerance for mapping numbers to strings.

    Returns:
        GroundTruthTab with extracted notes.
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(str(pdf_path))

    result = GroundTruthTab(
        title=pdf_path.stem,
        num_strings=num_strings,
    )

    # Extract metadata from first page
    first_page_text = doc[0].get_text()
    _extract_metadata(first_page_text, result)

    all_notes: list[GroundTruthNote] = []

    for page_idx in range(len(doc)):
        page = doc[page_idx]

        # 1. Find tab string lines (horizontal lines grouped in sets of num_strings)
        systems = _find_tab_systems(page, page_idx, num_strings)
        logger.debug("Page %d: %d tab systems found", page_idx + 1, len(systems))

        # 2. Extract fret numbers with positions
        fret_numbers = _extract_fret_numbers(page)
        logger.debug("Page %d: %d fret numbers found", page_idx + 1, len(fret_numbers))

        # 3. Extract technique annotations
        techniques = _extract_techniques(page)

        # 4. Map fret numbers to strings within each system
        for sys in systems:
            sys_notes = _map_numbers_to_strings(fret_numbers, sys, string_spacing_tolerance)
            # Apply technique annotations
            for note in sys_notes:
                for tech_x, tech_y, tech_label in techniques:
                    if abs(tech_x - note.x_pos) < 8 and sys.string_ys[0] - 15 < tech_y < sys.string_ys[0]:
                        note.technique = tech_label
            all_notes.extend(sys_notes)

    result.notes = all_notes
    num_pages = len(doc)
    doc.close()

    logger.info(
        "Parsed %s: %d notes across %d pages",
        pdf_path.name,
        len(all_notes),
        num_pages,
    )
    return result


def _extract_metadata(text: str, result: GroundTruthTab) -> None:
    """Extract title, tuning, BPM from first page text."""
    lines = text.strip().split("\n")
    if lines:
        result.title = lines[0].strip()

    # BPM detection
    bpm_match = re.search(r"[=♩]\s*(\d+)", text)
    if bpm_match:
        result.bpm = float(bpm_match.group(1))

    # Tuning detection
    text_lower = text.lower()
    if "standard tuning" in text_lower or "standard" in text_lower:
        result.tuning = "standard"
    elif "drop d" in text_lower:
        result.tuning = "drop-d"
    elif "drop a" in text_lower:
        result.tuning = "drop-a7"


def _find_tab_systems(
    page: fitz.Page,
    page_idx: int,
    num_strings: int,
) -> list[TabSystem]:
    """Find groups of horizontal lines that form tab systems."""
    drawings = page.get_drawings()

    # Collect all long horizontal lines
    h_lines: list[tuple[float, float, float]] = []  # (y, x_start, x_end)
    for d in drawings:
        for item in d["items"]:
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                if abs(p1.y - p2.y) < 1 and abs(p1.x - p2.x) > 80:
                    h_lines.append((round(p1.y, 1), min(p1.x, p2.x), max(p1.x, p2.x)))

    if not h_lines:
        return []

    # Deduplicate lines at same Y (multiple segments per string)
    unique_ys: dict[float, tuple[float, float]] = {}
    for y, x1, x2 in h_lines:
        if y in unique_ys:
            unique_ys[y] = (min(unique_ys[y][0], x1), max(unique_ys[y][1], x2))
        else:
            unique_ys[y] = (x1, x2)

    sorted_ys = sorted(unique_ys.keys())

    # Group into systems: consecutive lines with consistent spacing
    systems: list[TabSystem] = []
    i = 0
    sys_count = 0
    while i <= len(sorted_ys) - num_strings:
        candidate = sorted_ys[i : i + num_strings]

        # Check spacing is consistent (tab lines are evenly spaced)
        spacings = [candidate[j + 1] - candidate[j] for j in range(len(candidate) - 1)]
        avg_spacing = sum(spacings) / len(spacings)

        if avg_spacing > 3 and all(abs(s - avg_spacing) < 2.0 for s in spacings):
            # Check this isn't a 5-line music staff (staff spacing ~4.5pt, tab ~6-7pt)
            x_start = min(unique_ys[y][0] for y in candidate)
            x_end = max(unique_ys[y][1] for y in candidate)

            # Tab lines are exactly num_strings, skip if it looks like a staff
            if num_strings == 6 and len(candidate) == 6:
                systems.append(
                    TabSystem(
                        string_ys=candidate,  # top to bottom = high to low string
                        x_start=x_start,
                        x_end=x_end,
                        page=page_idx,
                        system_idx=sys_count,
                    )
                )
                sys_count += 1
                i += num_strings
                continue

        i += 1

    return systems


def _extract_fret_numbers(page: fitz.Page) -> list[tuple[float, float, int]]:
    """Extract all fret numbers with their (x, y) positions.

    Returns list of (x, y, fret_number).
    """
    numbers: list[tuple[float, float, int]] = []
    blocks = page.get_text("dict")["blocks"]

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                # Fret numbers are typically 8-10pt font size
                if span["size"] < 7 or span["size"] > 12:
                    continue
                if re.match(r"^\d{1,2}$", text):
                    fret = int(text)
                    if 0 <= fret <= 24:
                        x, y = span["origin"]
                        numbers.append((x, y, fret))

    return numbers


def _extract_techniques(page: fitz.Page) -> list[tuple[float, float, str]]:
    """Extract technique annotation positions.

    Returns list of (x, y, technique_label).
    """
    techniques: list[tuple[float, float, str]] = []
    blocks = page.get_text("dict")["blocks"]

    technique_map = {
        "T": "tap",
        "H": "hammer-on",
        "P": "pull-off",
        "S": "slide",
        "h": "hammer-on",
        "p": "pull-off",
        "s": "slide",
        "b": "bend",
    }

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                text = span["text"].strip()
                if text in technique_map and span["size"] < 10:
                    x, y = span["origin"]
                    techniques.append((x, y, technique_map[text]))

    return techniques


def _map_numbers_to_strings(
    fret_numbers: list[tuple[float, float, int]],
    system: TabSystem,
    tolerance: float,
) -> list[GroundTruthNote]:
    """Map fret numbers to the nearest string in a tab system."""
    notes: list[GroundTruthNote] = []

    y_min = system.string_ys[0] - tolerance * 2
    y_max = system.string_ys[-1] + tolerance * 2

    for x, y, fret in fret_numbers:
        # Check if this number is within this system's vertical range
        if y < y_min or y > y_max:
            continue
        # Check horizontal range
        if x < system.x_start - 5 or x > system.x_end + 5:
            continue

        # Find nearest string
        best_string = -1
        best_dist = float("inf")
        for i, string_y in enumerate(system.string_ys):
            dist = abs(y - string_y)
            if dist < best_dist:
                best_dist = dist
                best_string = i

        if best_dist <= tolerance * 3 and best_string >= 0:
            # Convert from display order (top=high) to our convention (0=lowest)
            string_idx = len(system.string_ys) - 1 - best_string
            notes.append(
                GroundTruthNote(
                    page=system.page,
                    system=system.system_idx,
                    string=string_idx,
                    fret=fret,
                    x_pos=x,
                )
            )

    # Sort by x position (left to right = time order)
    notes.sort(key=lambda n: n.x_pos)
    return notes


def save_ground_truth(gt: GroundTruthTab, output_path: str | Path) -> Path:
    """Save ground truth to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "title": gt.title,
        "tuning": gt.tuning,
        "bpm": gt.bpm,
        "num_strings": gt.num_strings,
        "note_count": len(gt.notes),
        "notes": [asdict(n) for n in gt.notes],
    }

    output_path.write_text(json.dumps(data, indent=2))
    logger.info("Ground truth saved to %s (%d notes)", output_path, len(gt.notes))
    return output_path
