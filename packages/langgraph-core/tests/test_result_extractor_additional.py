"""Additional coverage for ResultExtractor methods."""
from unittest.mock import MagicMock

import pytest

from oai_agent_core.langgraph_core.processing.result_extractor import ResultExtractor


@pytest.fixture
def extractor():
    return ResultExtractor()


# ---- extract_last_message ----

def test_extract_last_message_with_messages(extractor):
    msg = MagicMock()
    result = {"messages": [msg]}
    last = extractor.extract_last_message(result)
    assert last == msg


def test_extract_last_message_no_messages_key(extractor):
    result = {}
    last = extractor.extract_last_message(result)
    assert last is None


def test_extract_last_message_empty_messages(extractor):
    result = {"messages": []}
    last = extractor.extract_last_message(result)
    assert last is None


# ---- extract_text ----

def test_extract_text_with_message_content(extractor):
    msg = MagicMock()
    msg.content = "response text"
    result = {"messages": [msg]}
    text = extractor.extract_text(result)
    assert text == "response text"


def test_extract_text_with_dict_message(extractor):
    result = {"messages": [{"content": "dict content"}]}
    text = extractor.extract_text(result)
    assert text == "dict content"


def test_extract_text_no_last_message(extractor):
    result = {}
    text = extractor.extract_text(result)
    assert text == ""


# ---- extract_token_usage ----

def test_extract_token_usage_with_usage_metadata(extractor):
    msg = MagicMock()
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    result = {"messages": [msg]}
    usage = extractor.extract_token_usage(result)
    assert usage == {"input_tokens": 10, "output_tokens": 5}


def test_extract_token_usage_dict_message(extractor):
    result = {"messages": [{"usage_metadata": {"tokens": 100}}]}
    usage = extractor.extract_token_usage(result)
    assert usage == {"tokens": 100}


def test_extract_token_usage_no_data(extractor):
    result = {}
    usage = extractor.extract_token_usage(result)
    assert usage == {}


# ---- format_response ----

def test_format_response_basic(extractor):
    msg = MagicMock()
    msg.content = "output"
    result = {"messages": [msg]}
    resp = extractor.format_response(result, "sess1", "gpt-4")
    assert resp["content"]["session_id"] == "sess1"
    assert resp["content"]["text"] == "output"
    assert resp["model"]["model_id"] == "gpt-4"


def test_format_response_with_structured(extractor):
    from pydantic import BaseModel

    class Output(BaseModel):
        value: str

    out_obj = Output(value="test")
    result = {"structured_response": out_obj}
    resp = extractor.format_response(result, "sess2", "model")
    assert "value" in resp["content"]["text"]


def test_format_response_with_raw(extractor):
    msg = MagicMock()
    msg.content = "text"
    result = {"messages": [msg]}
    resp = extractor.format_response(result, "sess", "m", include_raw=True)
    assert "raw_response" in resp


def test_format_response_with_input_message(extractor):
    msg = MagicMock()
    msg.content = "out"
    result = {"messages": [msg]}
    resp = extractor.format_response(result, "s", "m", input_message="in")
    assert resp["input_message"] == "in"


# ---- format_stream_chunk ----

def test_format_stream_chunk_with_content(extractor):
    msg = MagicMock()
    msg.content = "chunk text"
    msg.response_metadata = {}
    msg.tool_calls = []
    result = {"messages": [msg]}
    chunk = extractor.format_stream_chunk(result, "sess", "model")
    assert chunk["content"]["text"] == "chunk text"
    assert chunk["content"]["final"] is False


def test_format_stream_chunk_with_tool_calls(extractor):
    msg = MagicMock()
    msg.content = ""
    msg.response_metadata = {}
    msg.tool_calls = [{"id": "1"}]
    result = {"messages": [msg]}
    chunk = extractor.format_stream_chunk(result, "sess", "model")
    assert "tool_calls" in chunk
    assert chunk["content"]["text"] == "Executing tools"


def test_format_stream_chunk_finish_reason_stop(extractor):
    msg = MagicMock()
    msg.content = "final"
    msg.response_metadata = {"finish_reason": "stop"}
    msg.tool_calls = []
    result = {"messages": [msg]}
    chunk = extractor.format_stream_chunk(result, "sess", "model")
    assert chunk["content"]["final"] is True


def test_format_stream_chunk_no_message(extractor):
    result = {}
    chunk = extractor.format_stream_chunk(result, "sess", "model")
    assert chunk["content"]["text"] == "{}"
    assert chunk["token_usage"] == {}
