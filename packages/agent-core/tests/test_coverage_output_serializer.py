"""Additional coverage for OutputSerializer."""
import logging
from unittest.mock import MagicMock

import pytest

from oai_agent_core.processing.output_serializer import OutputSerializer


def test_serialize_passthrough_dict():
    s = OutputSerializer()
    assert s.serialize({"a": 1}) == {"a": 1}


def test_serialize_object_with_dict():
    class Obj:
        def __init__(self):
            self.x = 1

    out = OutputSerializer().serialize(Obj())
    assert out == {"x": 1}


def test_serialize_string_conversion():
    out = OutputSerializer().serialize(123)
    assert out["content"] == "123"
    assert out["type"] == "int"
    assert out["_serialized_from"] == "string_conversion"


def test_write_and_read_session_with_metadata():
    s = OutputSerializer()
    s.write_to_session({"type": "text_delta", "content": "hi"}, "sess", metadata={"k": "v"})
    outs = s.get_session_outputs("sess")
    assert outs[0]["k"] == "v"
    assert "timestamp" in outs[0]


def test_write_uses_existing_timestamp():
    s = OutputSerializer()
    s.write_to_session({"timestamp": "T", "content": "x"}, "sess")
    assert s.get_session_outputs("sess")[0]["timestamp"] == "T"


def test_write_unexpected_error_logged():
    s = OutputSerializer(logger=MagicMock())
    # serialize raising a generic exception -> the broad except branch (104-110)
    s.serialize = MagicMock(side_effect=RuntimeError("boom"))
    s.write_to_session("data", "sess")
    assert s.logger.error.called


def test_clear_session_paths():
    s = OutputSerializer(logger=MagicMock())
    s.write_to_session({"content": "x"}, "sess")
    assert s.clear_session("sess") is True
    assert s.clear_session("missing") is False


def test_get_session_count_and_latest():
    s = OutputSerializer()
    assert s.get_session_count("none") == 0
    assert s.get_latest_output("none") is None
    s.write_to_session({"content": "a"}, "sess")
    s.write_to_session({"content": "b"}, "sess")
    assert s.get_session_count("sess") == 2
    assert s.get_latest_output("sess")["content"] == "b"


def test_filter_and_text_and_tools():
    s = OutputSerializer()
    s.write_to_session({"type": "text_delta", "content": "Hel"}, "sess")
    s.write_to_session({"type": "text_delta", "content": "lo"}, "sess")
    s.write_to_session({"type": "tool_use", "name": "t"}, "sess")
    assert s.get_text_content("sess") == "Hello"
    assert len(s.get_tool_uses("sess")) == 1
    assert s.list_sessions() == ["sess"]


def test_session_summary_empty_and_populated():
    s = OutputSerializer()
    empty = s.get_session_summary("none")
    assert empty["exists"] is False
    s.write_to_session({"type": "text_delta", "content": "hi", "timestamp": "1"}, "sess")
    s.write_to_session({"type": "tool_use", "timestamp": "2"}, "sess")
    summary = s.get_session_summary("sess")
    assert summary["exists"] is True
    assert summary["total_outputs"] == 2
    assert summary["first_timestamp"] == "1"
    assert summary["last_timestamp"] == "2"
    assert summary["tool_use_count"] == 1


def test_export_session_formats():
    s = OutputSerializer()
    s.write_to_session({"type": "text_delta", "content": "hi"}, "sess")
    assert isinstance(s.export_session("sess", "json"), list)
    assert s.export_session("sess", "text") == "hi"
    assert s.export_session("sess", "summary")["exists"] is True
    with pytest.raises(ValueError):
        s.export_session("sess", "xml")


def test_clear_all_and_dunders():
    s = OutputSerializer()
    s.write_to_session({"content": "a"}, "s1")
    s.write_to_session({"content": "b"}, "s2")
    assert len(s) == 2
    assert "s1" in s
    assert "missing" not in s
    assert "OutputSerializer" in repr(s)
    assert s.clear_all_sessions() == 2
    assert len(s) == 0
