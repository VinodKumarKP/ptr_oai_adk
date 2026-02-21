import pytest
import json
import tempfile
import os
from unittest.mock import MagicMock, patch, mock_open
from oai_agent_core.processing.output_serializer import OutputSerializer


class TestOutputSerializer:
    
    @pytest.fixture
    def serializer(self):
        return OutputSerializer()
    
    def test_init(self, serializer):
        assert serializer.session_outputs == {}
        assert serializer.logger is not None
    
    def test_init_default_session(self):
        serializer = OutputSerializer()
        assert serializer.session_outputs == {}
    
    def test_serialize_dict(self, serializer):
        data = {"key": "value", "number": 123}
        result = serializer.serialize(data)
        assert result == data
    
    def test_serialize_string(self, serializer):
        result = serializer.serialize("test string")
        assert result["content"] == "test string"
        assert result["type"] == "str"
    
    def test_serialize_list(self, serializer):
        data = [1, 2, 3]
        result = serializer.serialize(data)
        assert result["content"] == "[1, 2, 3]"
        assert result["type"] == "list"
    
    def test_serialize_none(self, serializer):
        result = serializer.serialize(None)
        assert result["content"] == "None"
        assert result["type"] == "NoneType"
    
    def test_serialize_complex_object(self, serializer):
        class CustomObject:
            def __init__(self):
                self.value = "test"
        
        obj = CustomObject()
        result = serializer.serialize(obj)
        assert result["value"] == "test"
    
    def test_write_to_session(self, serializer):
        serializer.write_to_session("test content", "session1")
        outputs = serializer.get_session_outputs("session1")
        assert len(outputs) == 1
        assert outputs[0]["content"] == "test content"
    
    def test_get_session_outputs(self, serializer):
        serializer.write_to_session("test1", "session1")
        serializer.write_to_session("test2", "session1")
        outputs = serializer.get_session_outputs("session1")
        assert len(outputs) == 2
    
    def test_clear_session(self, serializer):
        serializer.write_to_session("test", "session1")
        result = serializer.clear_session("session1")
        assert result is True
        assert serializer.get_session_outputs("session1") == []
    
    def test_get_session_count(self, serializer):
        serializer.write_to_session("test1", "session1")
        serializer.write_to_session("test2", "session1")
        count = serializer.get_session_count("session1")
        assert count == 2
    
    def test_list_sessions(self, serializer):
        serializer.write_to_session("test1", "session1")
        serializer.write_to_session("test2", "session2")
        sessions = serializer.list_sessions()
        assert "session1" in sessions
        assert "session2" in sessions
    
    def test_get_latest_output(self, serializer):
        serializer.write_to_session("test1", "session1")
        serializer.write_to_session("test2", "session1")
        latest = serializer.get_latest_output("session1")
        assert latest["content"] == "test2"
    
    def test_filter_outputs_by_type(self, serializer):
        serializer.write_to_session({"type": "text_delta", "content": "test1"}, "session1")
        serializer.write_to_session({"type": "tool_use", "content": "test2"}, "session1")
        text_outputs = serializer.filter_outputs_by_type("session1", "text_delta")
        assert len(text_outputs) == 1
        assert text_outputs[0]["content"] == "test1"
    
    def test_get_text_content(self, serializer):
        serializer.write_to_session({"type": "text_delta", "content": "Hello "}, "session1")
        serializer.write_to_session({"type": "text_delta", "content": "World!"}, "session1")
        text = serializer.get_text_content("session1")
        assert text == "Hello World!"
    
    def test_get_session_summary(self, serializer):
        serializer.write_to_session({"type": "text_delta", "content": "test"}, "session1")
        summary = serializer.get_session_summary("session1")
        assert summary["exists"] is True
        assert summary["total_outputs"] == 1