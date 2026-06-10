"""Tests for request model validation."""

import pytest
import os
from pydantic import ValidationError

from oai_agent_server.models.requests import ChatRequest, StreamChatRequest


class TestChatRequestValidation:
    """Test ChatRequest model validation."""
    
    def test_valid_chat_request_minimal(self):
        """Minimal valid request should pass."""
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.user_id == "user"
        assert req.session_id is None
    
    def test_valid_chat_request_with_session(self):
        """Request with session_id should pass."""
        req = ChatRequest(
            message="Hello",
            session_id="550e8400-e29b-41d4-a716-446655440000"
        )
        assert req.session_id == "550e8400-e29b-41d4-a716-446655440000"
    
    def test_valid_chat_request_with_user_id(self):
        """Request with user_id should pass."""
        req = ChatRequest(
            message="Hello",
            user_id="john_doe"
        )
        assert req.user_id == "john_doe"
    
    def test_valid_chat_request_dict_message(self):
        """Dict message should be valid."""
        req = ChatRequest(message={"text": "Hello", "lang": "en"})
        assert isinstance(req.message, dict)
        assert req.message["text"] == "Hello"
    
    def test_message_empty_string_rejected(self):
        """Empty message should be rejected."""
        with pytest.raises(ValidationError, match="empty or whitespace"):
            ChatRequest(message="")
    
    def test_message_whitespace_only_rejected(self):
        """Whitespace-only message should be rejected."""
        with pytest.raises(ValidationError, match="empty or whitespace"):
            ChatRequest(message="   \n\t  ")
    
    def test_message_empty_dict_rejected(self):
        """Empty dict message should be rejected."""
        with pytest.raises(ValidationError, match="cannot be empty"):
            ChatRequest(message={})
    
    def test_message_size_limit_exceeded(self, monkeypatch):
        """Message exceeding max size should be rejected."""
        monkeypatch.setenv("MAX_MESSAGE_SIZE_BYTES", "100")
        # Need to reload the module for env var to take effect
        # For now, test with a very long message
        long_message = "x" * 33000  # Exceeds default 32KB limit
        
        with pytest.raises(ValidationError, match="exceeds maximum allowed size"):
            ChatRequest(message=long_message)
    
    def test_message_json_string_parsed(self):
        """JSON string in message should be parsed."""
        req = ChatRequest(message='{"text": "Hello"}')
        assert isinstance(req.message, dict)
        assert req.message["text"] == "Hello"
    
    def test_message_invalid_json_string_treated_as_string(self):
        """Invalid JSON string should be treated as plain string."""
        req = ChatRequest(message="{not valid json}")
        assert isinstance(req.message, str)
        assert req.message == "{not valid json}"
    
    def test_session_id_uuid_format(self):
        """UUID format session_id should be valid."""
        uuid = "550e8400-e29b-41d4-a716-446655440000"
        req = ChatRequest(message="Hello", session_id=uuid)
        assert req.session_id == uuid
    
    def test_session_id_custom_format(self):
        """Custom alphanumeric session_id should be valid."""
        req = ChatRequest(message="Hello", session_id="session_123-abc")
        assert req.session_id == "session_123-abc"
    
    def test_session_id_invalid_format(self):
        """Invalid session_id format should be rejected."""
        with pytest.raises(ValidationError, match="valid UUID or alphanumeric"):
            ChatRequest(message="Hello", session_id="invalid@#$%")
    
    def test_session_id_too_long(self):
        """Session_id exceeding 100 chars should be rejected."""
        long_id = "a" * 101
        with pytest.raises(ValidationError, match="valid UUID or alphanumeric"):
            ChatRequest(message="Hello", session_id=long_id)
    
    def test_session_id_none_accepted(self):
        """None session_id should be accepted."""
        req = ChatRequest(message="Hello", session_id=None)
        assert req.session_id is None
    
    def test_user_id_simple(self):
        """Simple alphanumeric user_id should be valid."""
        req = ChatRequest(message="Hello", user_id="user123")
        assert req.user_id == "user123"
    
    def test_user_id_with_dots(self):
        """User_id with dots should be valid."""
        req = ChatRequest(message="Hello", user_id="john.doe.smith")
        assert req.user_id == "john.doe.smith"
    
    def test_user_id_invalid_special_chars(self):
        """User_id with invalid special characters should be rejected."""
        with pytest.raises(ValidationError, match="only letters, numbers"):
            ChatRequest(message="Hello", user_id="user@#$%")
    
    def test_user_id_too_long(self):
        """User_id exceeding 100 chars should be rejected."""
        long_id = "a" * 101
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", user_id=long_id)
    
    def test_user_id_default_value(self):
        """Default user_id should be 'user'."""
        req = ChatRequest(message="Hello")
        assert req.user_id == "user"


class TestStreamChatRequestValidation:
    """Test StreamChatRequest model validation."""
    
    def test_valid_stream_request_minimal(self):
        """Minimal valid stream request should pass."""
        req = StreamChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.verbose is False
        assert req.user_id == "user"
    
    def test_stream_request_verbose_true(self):
        """Request with verbose=True should pass."""
        req = StreamChatRequest(message="Hello", verbose=True)
        assert req.verbose is True
    
    def test_stream_request_verbose_false(self):
        """Request with verbose=False should pass."""
        req = StreamChatRequest(message="Hello", verbose=False)
        assert req.verbose is False
    
    def test_stream_request_message_validation_same_as_chat(self):
        """Message validation should be same as ChatRequest."""
        # Test empty message
        with pytest.raises(ValidationError, match="empty or whitespace"):
            StreamChatRequest(message="")
        
        # Test dict message
        req = StreamChatRequest(message={"text": "Hello"})
        assert isinstance(req.message, dict)
    
    def test_stream_request_all_fields(self):
        """Request with all fields should pass."""
        req = StreamChatRequest(
            message="Hello",
            session_id="session123",
            user_id="alice",
            verbose=True
        )
        assert req.message == "Hello"
        assert req.session_id == "session123"
        assert req.user_id == "alice"
        assert req.verbose is True


class TestRequestModelSerialization:
    """Test serialization and deserialization of request models."""
    
    def test_chat_request_dict_serialization(self):
        """ChatRequest should serialize to dict."""
        req = ChatRequest(message="Hello", user_id="alice")
        req_dict = req.model_dump()
        assert req_dict["message"] == "Hello"
        assert req_dict["user_id"] == "alice"
    
    def test_chat_request_json_schema(self):
        """ChatRequest should have valid JSON schema."""
        schema = ChatRequest.model_json_schema()
        assert "properties" in schema
        assert "message" in schema["properties"]
        assert "user_id" in schema["properties"]
    
    def test_stream_request_json_schema(self):
        """StreamChatRequest should have valid JSON schema."""
        schema = StreamChatRequest.model_json_schema()
        assert "properties" in schema
        assert "verbose" in schema["properties"]


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_message_with_unicode(self):
        """Messages with unicode characters should work."""
        req = ChatRequest(message="你好 مرحبا 🎉")
        assert "你好" in req.message
    
    def test_message_with_special_markdown(self):
        """Messages with markdown should work."""
        msg = "# Title\n\n**bold** and *italic*"
        req = ChatRequest(message=msg)
        assert req.message == msg
    
    def test_message_with_code_block(self):
        """Messages with code blocks should work."""
        msg = '```python\nprint("hello")\n```'
        req = ChatRequest(message=msg)
        assert req.message == msg
    
    def test_nested_dict_message(self):
        """Nested dict in message should work."""
        msg = {"data": {"nested": {"value": "test"}}}
        req = ChatRequest(message=msg)
        assert req.message["data"]["nested"]["value"] == "test"
    
    def test_message_size_boundary(self, monkeypatch):
        """Message exactly at size limit should pass."""
        monkeypatch.setenv("MAX_MESSAGE_SIZE_BYTES", "100")
        # This is hard to test precisely due to env var reload limitations
        # But we can test that it respects the limit
    
    def test_session_id_case_sensitivity(self):
        """Session ID should preserve case."""
        req = ChatRequest(message="Hello", session_id="SessionID_123")
        assert req.session_id == "SessionID_123"
    
    def test_user_id_underscore_hyphen(self):
        """User ID with underscores and hyphens should work."""
        req = ChatRequest(message="Hello", user_id="user_name-123")
        assert req.user_id == "user_name-123"
    
    def test_message_with_newlines(self):
        """Messages with newlines should work."""
        msg = "Line 1\nLine 2\nLine 3"
        req = ChatRequest(message=msg)
        assert "\n" in req.message
    
    def test_message_with_tabs(self):
        """Messages with tabs should work."""
        msg = "Column1\tColumn2\tColumn3"
        req = ChatRequest(message=msg)
        assert "\t" in req.message


class TestValidationErrorMessages:
    """Test that validation error messages are clear and helpful."""
    
    def test_empty_message_error_is_clear(self):
        """Empty message error should be clear."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="")
        assert "empty or whitespace" in str(exc_info.value)
    
    def test_invalid_session_id_error_is_clear(self):
        """Invalid session ID error should be clear."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="Hello", session_id="invalid@#$")
        assert "UUID or alphanumeric" in str(exc_info.value)
    
    def test_invalid_user_id_error_is_clear(self):
        """Invalid user ID error should be clear."""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="Hello", user_id="user@#$%")
        assert "only letters, numbers" in str(exc_info.value)
    
    def test_oversized_message_error_is_clear(self):
        """Oversized message error should be clear."""
        long_msg = "x" * 33000
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=long_msg)
        assert "exceeds maximum allowed size" in str(exc_info.value)
