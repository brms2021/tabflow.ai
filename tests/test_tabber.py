"""Tests for tab_ripper.tabber — tuning, filtering, fret assignment, rendering."""

import pytest

from tab_ripper.tabber import (
    STANDARD_TUNING,
    TUNING_PRESETS,
    NoteEvent,
    TabEvent,
    TabNote,
    assign_frets,
    filter_notes,
    format_tab_header,
    parse_tuning,
    pitch_to_fret_options,
    render_ascii_tab,
    tuning_freq_range,
)

# -----------------------------------------------------------------------
# parse_tuning
# -----------------------------------------------------------------------


class TestParseTuning:
    def test_standard_preset(self):
        pitches, names = parse_tuning("standard")
        assert pitches == [40, 45, 50, 55, 59, 64]
        assert len(names) == 6

    def test_drop_d_preset(self):
        pitches, names = parse_tuning("drop-d")
        assert pitches[0] == 38  # D2
        assert pitches[1:] == [45, 50, 55, 59, 64]

    def test_7string_preset(self):
        pitches, names = parse_tuning("7-string")
        assert len(pitches) == 7
        assert pitches[0] == 35  # B1

    def test_all_presets_exist(self):
        for name in TUNING_PRESETS:
            pitches, labels = parse_tuning(name)
            assert len(pitches) == len(labels)
            assert all(isinstance(p, int) for p in pitches)

    def test_custom_tuning_notes(self):
        pitches, names = parse_tuning("E2,A2,D3,G3,B3,E4")
        assert pitches == [40, 45, 50, 55, 59, 64]

    def test_case_insensitive(self):
        p1, _ = parse_tuning("Standard")
        p2, _ = parse_tuning("STANDARD")
        assert p1 == p2

    def test_invalid_tuning_raises(self):
        with pytest.raises(ValueError):
            parse_tuning("not-a-tuning")

    def test_invalid_note_name_raises(self):
        with pytest.raises(ValueError):
            parse_tuning("X9,Y9,Z9")


# -----------------------------------------------------------------------
# tuning_freq_range
# -----------------------------------------------------------------------


class TestTuningFreqRange:
    def test_standard_range(self):
        min_hz, max_hz = tuning_freq_range(STANDARD_TUNING)
        # E2 ~82 Hz, E4+24 frets = E6 ~1319 Hz (with margin)
        assert min_hz < 82
        assert max_hz > 1300

    def test_lower_tuning_extends_range(self):
        standard_min, _ = tuning_freq_range(STANDARD_TUNING)
        seven_min, _ = tuning_freq_range(TUNING_PRESETS["7-string"][0])
        assert seven_min < standard_min


# -----------------------------------------------------------------------
# pitch_to_fret_options
# -----------------------------------------------------------------------


class TestPitchToFretOptions:
    def test_open_low_e(self):
        options = pitch_to_fret_options(40, STANDARD_TUNING)
        assert (0, 0) in options  # open 6th string

    def test_middle_c(self):
        # C4 = MIDI 60, on standard tuning:
        # string 3 (G3=55) fret 5, string 4 (B3=59) fret 1
        options = pitch_to_fret_options(60, STANDARD_TUNING)
        assert (3, 5) in options
        assert (4, 1) in options

    def test_too_low_pitch(self):
        options = pitch_to_fret_options(30, STANDARD_TUNING)
        assert options == []

    def test_too_high_pitch(self):
        options = pitch_to_fret_options(100, STANDARD_TUNING)
        assert options == []

    def test_all_options_produce_correct_pitch(self):
        for pitch in range(40, 89):
            for string, fret in pitch_to_fret_options(pitch, STANDARD_TUNING):
                assert STANDARD_TUNING[string] + fret == pitch


# -----------------------------------------------------------------------
# filter_notes
# -----------------------------------------------------------------------


def _make_raw_events(notes: list[tuple]) -> list[tuple]:
    """Create raw note event tuples: (start, end, pitch, amplitude, bends)."""
    return [(s, e, p, a, []) for s, e, p, a in notes]


class TestFilterNotes:
    def test_amplitude_filter(self):
        events = _make_raw_events(
            [
                (0.0, 1.0, 60, 0.8),  # keep
                (1.0, 2.0, 62, 0.1),  # drop
            ]
        )
        result = filter_notes(events, amplitude_threshold=0.4)
        assert len(result) == 1
        assert result[0].pitch == 60

    def test_range_filter(self):
        events = _make_raw_events(
            [
                (0.0, 1.0, 60, 0.8),  # in range
                (1.0, 2.0, 30, 0.8),  # below guitar range
                (2.0, 3.0, 100, 0.8),  # above guitar range
            ]
        )
        result = filter_notes(events, amplitude_threshold=0.0)
        assert all(40 <= n.pitch <= 88 for n in result)

    def test_short_note_filter(self):
        events = _make_raw_events(
            [
                (0.0, 1.0, 60, 0.8),  # long enough
                (2.0, 2.01, 62, 0.8),  # 10ms, too short (isolated)
            ]
        )
        result = filter_notes(events, amplitude_threshold=0.0, min_duration_ms=50)
        pitches = [n.pitch for n in result]
        assert 60 in pitches

    def test_deduplication(self):
        # Two near-simultaneous notes at same pitch
        events = _make_raw_events(
            [
                (0.0, 1.0, 60, 0.8),
                (0.02, 1.0, 60, 0.6),  # duplicate
            ]
        )
        result = filter_notes(events, amplitude_threshold=0.0)
        assert len(result) == 1
        assert result[0].amplitude == 0.8  # kept louder one

    def test_empty_input(self):
        assert filter_notes([]) == []

    def test_returns_note_events(self):
        events = _make_raw_events([(0.0, 1.0, 60, 0.8)])
        result = filter_notes(events, amplitude_threshold=0.0)
        assert isinstance(result[0], NoteEvent)


# -----------------------------------------------------------------------
# assign_frets (Viterbi)
# -----------------------------------------------------------------------


class TestAssignFrets:
    def test_single_note(self):
        notes = [NoteEvent(0.0, 1.0, 60, 0.8, 100)]
        events = assign_frets(notes)
        assert len(events) == 1
        assert len(events[0].notes) == 1
        note = events[0].notes[0]
        assert STANDARD_TUNING[note.string] + note.fret == 60

    def test_ascending_scale_stays_in_position(self):
        """An ascending scale should stay in a reasonable fret range."""
        # C major scale: C4-D4-E4-F4-G4 (spans 7 semitones across strings)
        pitches = [60, 62, 64, 65, 67]
        notes = [NoteEvent(i * 0.2, i * 0.2 + 0.15, p, 0.8, 100) for i, p in enumerate(pitches)]
        events = assign_frets(notes)
        frets = [e.notes[0].fret for e in events]
        fret_span = max(frets) - min(frets)
        # Should stay within a playable hand position (not jump wildly)
        assert fret_span <= 8

    def test_all_notes_have_correct_pitch(self):
        """Every assigned (string, fret) must produce the original MIDI pitch."""
        notes = [NoteEvent(i * 0.3, i * 0.3 + 0.2, p, 0.8, 100) for i, p in enumerate([40, 45, 50, 55, 59, 64, 67, 71])]
        events = assign_frets(notes)
        for event in events:
            for note in event.notes:
                assert STANDARD_TUNING[note.string] + note.fret == note.midi_pitch

    def test_empty_input(self):
        assert assign_frets([]) == []

    def test_chord(self):
        """Two simultaneous notes should be on different strings."""
        notes = [
            NoteEvent(0.0, 1.0, 60, 0.8, 100),
            NoteEvent(0.0, 1.0, 64, 0.8, 100),
        ]
        events = assign_frets(notes)
        assert len(events) == 1
        strings = [n.string for n in events[0].notes]
        assert len(set(strings)) == len(strings)  # no duplicates

    def test_7string_tuning(self):
        tuning = TUNING_PRESETS["7-string"][0]
        notes = [NoteEvent(0.0, 1.0, 35, 0.8, 100)]  # B1 = open 7th string
        events = assign_frets(notes, tuning=tuning)
        assert len(events) == 1
        note = events[0].notes[0]
        assert tuning[note.string] + note.fret == 35

    def test_sweep_pattern_stays_compact(self):
        """A fast arpeggio across strings should stay in a tight fret range."""
        # Am arpeggio sweep ascending: A2, C3, E3, A3, C4, E4
        pitches = [45, 48, 52, 57, 60, 64]
        notes = [NoteEvent(i * 0.06, i * 0.06 + 0.05, p, 0.8, 100) for i, p in enumerate(pitches)]
        events = assign_frets(notes)
        frets = [e.notes[0].fret for e in events]
        # All frets should stay within a compact range (natural hand position)
        span = max(frets) - min(frets)
        assert span <= 5, f"Sweep fret span {span} too wide: {frets}"


# -----------------------------------------------------------------------
# render_ascii_tab
# -----------------------------------------------------------------------


class TestRenderAsciiTab:
    def _make_event(self, time, string, fret, pitch=60):
        return TabEvent(
            time=time,
            notes=[TabNote(time=time, duration=0.5, midi_pitch=pitch, string=string, fret=fret, velocity=100)],
        )

    def test_basic_output(self):
        events = [self._make_event(0.0, 5, 5, pitch=69)]  # high e string, fret 5
        result = render_ascii_tab(events)
        assert "5" in result
        assert "|" in result

    def test_string_labels_present(self):
        events = [self._make_event(0.0, 0, 0, pitch=40)]
        result = render_ascii_tab(events)
        lines = [line for line in result.strip().split("\n") if line]
        # Should have string labels e, B, G, D, A, E
        labels = [line.split("|")[0] for line in lines[:6]]
        assert "e" in labels
        assert "E" in labels

    def test_empty_events(self):
        result = render_ascii_tab([])
        assert "no notes" in result.lower()

    def test_fret_number_in_output(self):
        events = [self._make_event(0.5, 2, 12, pitch=62)]
        result = render_ascii_tab(events)
        assert "12" in result


# -----------------------------------------------------------------------
# format_tab_header
# -----------------------------------------------------------------------


class TestFormatTabHeader:
    def test_contains_title(self):
        header = format_tab_header("My Song", 100, 120.0)
        assert "My Song" in header

    def test_contains_note_count(self):
        header = format_tab_header("Test", 42, 60.0)
        assert "42" in header

    def test_contains_duration(self):
        header = format_tab_header("Test", 10, 33.5)
        assert "33.5" in header

    def test_contains_tuning_info(self):
        header = format_tab_header("Test", 10, 30.0, string_names=["E", "A", "D", "G", "B", "e"])
        assert "E A D G B e" in header
