"""Tests for tab_ripper.llm_analyzer — JSON parsing, phrase splitting."""


from tab_ripper.llm_analyzer import _parse_json_response, split_into_phrases
from tab_ripper.tabber import TabEvent, TabNote

# -----------------------------------------------------------------------
# _parse_json_response
# -----------------------------------------------------------------------

class TestParseJsonResponse:
    def test_raw_json_array(self):
        raw = '[{"seq": 0, "technique": "normal"}]'
        result = _parse_json_response(raw)
        assert result == [{"seq": 0, "technique": "normal"}]

    def test_markdown_code_fence(self):
        raw = '```json\n[{"seq": 0}]\n```'
        result = _parse_json_response(raw)
        assert result == [{"seq": 0}]

    def test_markdown_fence_no_lang(self):
        raw = '```\n[{"seq": 0}]\n```'
        result = _parse_json_response(raw)
        assert result == [{"seq": 0}]

    def test_preamble_text_before_json(self):
        raw = 'Here is my analysis:\n[{"seq": 0, "technique": "sweep"}]'
        result = _parse_json_response(raw)
        assert result == [{"seq": 0, "technique": "sweep"}]

    def test_empty_string(self):
        assert _parse_json_response("") is None

    def test_no_json_at_all(self):
        assert _parse_json_response("I cannot analyze this phrase.") is None

    def test_json_object_not_array(self):
        assert _parse_json_response('{"key": "value"}') is None

    def test_malformed_json(self):
        assert _parse_json_response('[{"seq": 0,}]') is None

    def test_multiple_elements(self):
        raw = '[{"seq": 0, "technique": "normal"}, {"seq": 1, "technique": "sweep"}]'
        result = _parse_json_response(raw)
        assert len(result) == 2
        assert result[1]["technique"] == "sweep"

    def test_json_with_trailing_text(self):
        raw = '[{"seq": 0}]\n\nNote: the above analysis...'
        result = _parse_json_response(raw)
        assert result == [{"seq": 0}]


# -----------------------------------------------------------------------
# split_into_phrases
# -----------------------------------------------------------------------

def _make_events(times: list[float]) -> list[TabEvent]:
    return [
        TabEvent(
            time=t,
            notes=[TabNote(time=t, duration=0.1, midi_pitch=60, string=0, fret=5, velocity=100)],
        )
        for t in times
    ]


class TestSplitIntoPhrases:
    def test_no_events(self):
        assert split_into_phrases([]) == []

    def test_single_event(self):
        events = _make_events([0.0])
        phrases = split_into_phrases(events)
        assert phrases == [[0]]

    def test_continuous_phrase(self):
        events = _make_events([0.0, 0.1, 0.2, 0.3])
        phrases = split_into_phrases(events, pause_threshold=0.4)
        assert len(phrases) == 1
        assert phrases[0] == [0, 1, 2, 3]

    def test_split_on_pause(self):
        events = _make_events([0.0, 0.1, 0.2, 1.0, 1.1])
        phrases = split_into_phrases(events, pause_threshold=0.4)
        assert len(phrases) == 2
        assert phrases[0] == [0, 1, 2]
        assert phrases[1] == [3, 4]

    def test_max_notes_per_phrase(self):
        # Create 40 tightly-spaced events — should split at 32
        events = _make_events([i * 0.05 for i in range(40)])
        phrases = split_into_phrases(events, pause_threshold=10.0)
        assert len(phrases) >= 2
        assert len(phrases[0]) == 32
