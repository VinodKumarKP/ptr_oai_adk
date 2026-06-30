"""Additional coverage for OpenAI ResultExtractor."""
from unittest.mock import MagicMock

import pytest

from oai_agent_core.openai_core.processing.result_extractor import ResultExtractor


@pytest.fixture
def extractor():
    return ResultExtractor()


def test_extract_text_with_content(extractor):
    """Test extracting text from result."""
    result = {"content": "response text"}
    text = extractor.extract_text(result)
    assert text == "response text"


def test_extract_token_usage_with_usage(extractor):
    """Test extracting token usage."""
    result = {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    usage = extractor.extract_token_usage(result)
    assert usage == {"prompt_tokens": 10, "completion_tokens": 5}


def test_format_response_basic(extractor):
    """Test formatting response."""
    result = {"content": "output"}
    resp = extractor.format_response(result, "sess1", "gpt-4", "openai")
    assert resp["content"]["text"] == "output"
    assert resp["content"]["session_id"] == "sess1"
    assert resp["model"]["model_id"] == "gpt-4"
    assert resp["model"]["model_provider"] == "openai"


def test_format_response_with_usage(extractor):
    """Test formatting response with token usage."""
    result = {
        "content": "output",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}
    }
    resp = extractor.format_response(result, "sess", "gpt-4", "openai")
    assert resp["token_usage"]["prompt_tokens"] == 10
