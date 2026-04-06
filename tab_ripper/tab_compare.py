"""Compare generated tablature against a reference for accuracy measurement.

Parses ASCII tab notation and computes note-level precision/recall.
"""

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TabPosition:
    """A single note position in tablature."""

    string: int  # 0-based, 0=lowest
    fret: int
    column: int  # horizontal position in the tab


def parse_ascii_tab(tab_text: str, num_strings: int = 6) -> list[TabPosition]:
    """Parse ASCII tab notation into note positions.

    Expects lines like:
        e|---5---7---5---|
        B|---5---8---5---|
        ...

    Args:
        tab_text: Multi-line ASCII tab string.
        num_strings: Number of strings in the tab.

    Returns:
        List of TabPosition objects.
    """
    lines = tab_text.strip().split("\n")
    positions: list[TabPosition] = []

    # Find groups of string lines (6 or 7 consecutive lines with | delimiters)
    string_groups: list[list[str]] = []
    current_group: list[str] = []

    for line in lines:
        stripped = line.strip()
        if "|" in stripped and re.match(r"^[A-Ga-g#b]?\|", stripped):
            current_group.append(stripped)
        else:
            if len(current_group) >= num_strings:
                string_groups.append(current_group[:num_strings])
            current_group = []
    if len(current_group) >= num_strings:
        string_groups.append(current_group[:num_strings])

    col_offset = 0
    for group in string_groups:
        # Parse each string line — high string first (index 0 = highest)
        group_width = 0
        for row_idx, line in enumerate(group):
            string_idx = num_strings - 1 - row_idx  # reverse: high string at top
            # Strip string label
            content = line.split("|", 1)[1] if "|" in line else line
            if content.endswith("|"):
                content = content[:-1]

            group_width = max(group_width, len(content))

            # Find fret numbers (1 or 2 digit)
            col = 0
            while col < len(content):
                if content[col].isdigit():
                    # Read full number
                    num_str = content[col]
                    if col + 1 < len(content) and content[col + 1].isdigit():
                        num_str += content[col + 1]
                    positions.append(
                        TabPosition(
                            string=string_idx,
                            fret=int(num_str),
                            column=col_offset + col,
                        )
                    )
                    col += len(num_str)
                else:
                    col += 1

        col_offset += group_width

    return positions


def compare_tabs(
    generated: list[TabPosition],
    reference: list[TabPosition],
    column_tolerance: int = 2,
) -> dict:
    """Compare generated tab positions against reference.

    Args:
        generated: Positions from generated tab.
        reference: Positions from reference tab.
        column_tolerance: How many columns apart notes can be to count as a match.

    Returns:
        Dict with precision, recall, f1, matched, missed, extra counts.
    """
    matched = 0
    ref_matched = set()

    for gen in generated:
        for ref_idx, ref in enumerate(reference):
            if ref_idx in ref_matched:
                continue
            if gen.string == ref.string and gen.fret == ref.fret and abs(gen.column - ref.column) <= column_tolerance:
                matched += 1
                ref_matched.add(ref_idx)
                break

    precision = matched / len(generated) if generated else 0.0
    recall = matched / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": matched,
        "generated_total": len(generated),
        "reference_total": len(reference),
        "extra": len(generated) - matched,
        "missed": len(reference) - matched,
    }
