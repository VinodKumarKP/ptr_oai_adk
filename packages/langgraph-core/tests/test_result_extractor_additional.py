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


# ---- Stream mode and subgraph support tests ----

def test_extract_messages_values_mode(extractor):
    """Test extract_messages with stream_mode='values'."""
    msg1 = MagicMock()
    msg1.content = "msg1"
    msg2 = MagicMock()
    msg2.content = "msg2"
    result = {"messages": [msg1, msg2]}
    messages = extractor.extract_messages(result, stream_mode="values")
    assert len(messages) == 2
    assert messages[0] == msg1
    assert messages[1] == msg2


def test_extract_messages_messages_mode_single_message(extractor):
    """Test extract_messages with stream_mode='messages' and single message."""
    msg = MagicMock()
    msg.content = "test"
    messages = extractor.extract_messages(msg, stream_mode="messages")
    assert len(messages) == 1
    assert messages[0] == msg


def test_extract_messages_messages_mode_dict(extractor):
    """Test extract_messages with stream_mode='messages' and dict result."""
    msg = MagicMock()
    result = {"messages": [msg]}
    messages = extractor.extract_messages(result, stream_mode="messages")
    assert len(messages) == 1
    assert messages[0] == msg


def test_extract_messages_messages_mode_single_message_key(extractor):
    """Test extract_messages with stream_mode='messages' and 'message' key."""
    msg = MagicMock()
    result = {"message": msg}
    messages = extractor.extract_messages(result, stream_mode="messages")
    assert len(messages) == 1
    assert messages[0] == msg


def test_extract_messages_empty_for_unknown_mode(extractor):
    """Test extract_messages returns None for unknown stream_mode."""
    result = {"messages": []}
    messages = extractor.extract_messages(result, stream_mode="unknown")
    assert messages is None


def test_extract_last_message_with_stream_mode(extractor):
    """Test extract_last_message with explicit stream_mode parameter."""
    msg1 = MagicMock()
    msg1.content = "msg1"
    msg2 = MagicMock()
    msg2.content = "msg2"
    result = {"messages": [msg1, msg2]}
    last = extractor.extract_last_message(result, stream_mode="values")
    assert last == msg2


def test_extract_text_with_stream_mode(extractor):
    """Test extract_text respects stream_mode parameter."""
    msg = MagicMock()
    msg.content = "test content"
    result = {"messages": [msg]}
    text = extractor.extract_text(result, stream_mode="values")
    assert text == "test content"


def test_extract_token_usage_with_stream_mode(extractor):
    """Test extract_token_usage respects stream_mode parameter."""
    msg = MagicMock()
    msg.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
    result = {"messages": [msg]}
    usage = extractor.extract_token_usage(result, stream_mode="values")
    assert usage == {"input_tokens": 10, "output_tokens": 5}


def test_format_response_with_stream_mode(extractor):
    """Test format_response with stream_mode parameter."""
    msg = MagicMock()
    msg.content = "response"
    result = {"messages": [msg]}
    resp = extractor.format_response(
        result, "sess", "model", stream_mode="values"
    )
    assert resp["content"]["text"] == "response"
    assert resp["content"]["session_id"] == "sess"


def test_format_stream_chunk_with_stream_mode_values(extractor):
    """Test format_stream_chunk with stream_mode='values'."""
    msg = MagicMock()
    msg.content = "chunk"
    msg.response_metadata = {}
    msg.tool_calls = []
    chunk = {"messages": [msg]}
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "model", stream_mode="values"
    )
    assert formatted["content"]["text"] == "chunk"
    assert formatted["content"]["session_id"] == "sess"


def test_format_stream_chunk_with_stream_mode_messages(extractor):
    """Test format_stream_chunk with stream_mode='messages'."""
    # Use a dict message for stream_mode="messages" since that's more typical
    msg = {"content": "message"}
    chunk = {"message": msg}
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "model", stream_mode="messages"
    )
    assert formatted["content"]["text"] == "message"


def test_format_stream_chunk_with_subgraphs_true(extractor):
    """Test format_stream_chunk with subgraphs=True."""
    msg1 = MagicMock()
    msg1.content = "main"
    msg2 = MagicMock()
    msg2.content = "subgraph"
    chunk = {"messages": [msg1, msg2]}
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "model", subgraphs=True
    )
    # Should extract the last message (from subgraph)
    assert formatted["content"]["text"] == "subgraph"


# ---- Specific real-world format tests ----

def test_extract_messages_stream_mode_messages_subgraphs_false(extractor):
    """Test stream_mode='messages' subgraphs=False format: (AIMessage, metadata_dict)."""
    msg = MagicMock()
    msg.content = "Hotel search result"
    metadata = {'checkpoint_ns': 'hotel_assistant', 'langgraph_node': 'hotel_assistant'}
    result = (msg, metadata)
    messages = extractor.extract_messages(result, stream_mode="messages", subgraphs=False)
    assert len(messages) == 1
    assert messages[0] == msg
    assert messages[0].content == "Hotel search result"


def test_extract_messages_stream_mode_messages_subgraphs_true(extractor):
    """Test stream_mode='messages' subgraphs=True format: (("node_name",), (AIMessage, metadata))."""
    msg = MagicMock()
    msg.content = "Transfer to assistant"
    metadata = {'checkpoint_ns': 'supervisor:...', 'langgraph_node': 'agent'}
    # Nested format with node names in first element
    result = (('supervisor:65bc3767-8242-b3e0-ba2a-9293b1eeed85',), (msg, metadata))
    messages = extractor.extract_messages(result, stream_mode="messages", subgraphs=True)
    assert len(messages) == 1
    assert messages[0] == msg
    assert messages[0].content == "Transfer to assistant"


def test_extract_messages_stream_mode_values_subgraphs_false(extractor):
    """Test stream_mode='values' subgraphs=False format: {"messages": [...]}."""
    msg = MagicMock()
    msg.content = "User query"
    result = {'messages': [msg]}
    messages = extractor.extract_messages(result, stream_mode="values", subgraphs=False)
    assert len(messages) == 1
    assert messages[0] == msg
    assert messages[0].content == "User query"


def test_extract_messages_stream_mode_values_subgraphs_true(extractor):
    """Test stream_mode='values' subgraphs=True format: ((), {"messages": [...]})."""
    msg = MagicMock()
    msg.content = "User query"
    # Tuple format: empty tuple followed by state dict
    result = ((), {'messages': [msg]})
    messages = extractor.extract_messages(result, stream_mode="values", subgraphs=True)
    assert len(messages) == 1
    assert messages[0] == msg
    assert messages[0].content == "User query"


def test_extract_last_message_stream_messages_subgraphs_false(extractor):
    """Test extract_last_message with stream_mode='messages' subgraphs=False."""
    msg = MagicMock()
    msg.content = "Last message"
    metadata = {}
    result = (msg, metadata)
    last = extractor.extract_last_message(result, stream_mode="messages", subgraphs=False)
    assert last == msg
    assert last.content == "Last message"


def test_extract_last_message_stream_messages_subgraphs_true(extractor):
    """Test extract_last_message with stream_mode='messages' subgraphs=True."""
    msg = MagicMock()
    msg.content = "Last message from subagent"
    metadata = {'langgraph_node': 'supervisor'}
    result = (('supervisor:...',), (msg, metadata))
    last = extractor.extract_last_message(result, stream_mode="messages", subgraphs=True)
    assert last == msg
    assert last.content == "Last message from subagent"


def test_extract_text_stream_messages_subgraphs_false(extractor):
    """Test extract_text with stream_mode='messages' subgraphs=False tuple."""
    msg = MagicMock()
    msg.content = "Tool call output"
    metadata = {}
    result = (msg, metadata)
    text = extractor.extract_text(result, stream_mode="messages", subgraphs=False)
    assert text == "Tool call output"


def test_extract_text_stream_values_subgraphs_true(extractor):
    """Test extract_text with stream_mode='values' subgraphs=True tuple."""
    msg = MagicMock()
    msg.content = "State message"
    result = ((), {'messages': [msg]})
    text = extractor.extract_text(result, stream_mode="values", subgraphs=True)
    assert text == "State message"


def test_format_stream_chunk_real_world_messages_subgraphs_false(extractor):
    """Test format_stream_chunk with real-world stream_mode='messages' subgraphs=False."""
    msg = MagicMock()
    msg.content = "Search result"
    msg.response_metadata = {}
    msg.tool_calls = []
    metadata = {'langgraph_node': 'hotel_assistant'}
    chunk = (msg, metadata)
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "gpt-4", stream_mode="messages", subgraphs=False
    )
    assert formatted["content"]["text"] == "Search result"
    assert formatted["content"]["session_id"] == "sess"


def test_format_stream_chunk_real_world_messages_subgraphs_true(extractor):
    """Test format_stream_chunk with real-world stream_mode='messages' subgraphs=True."""
    msg = MagicMock()
    msg.content = "Transfer decision"
    msg.response_metadata = {}
    msg.tool_calls = []
    metadata = {'langgraph_node': 'supervisor'}
    chunk = (('supervisor:...',), (msg, metadata))
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "gpt-4", stream_mode="messages", subgraphs=True
    )
    assert formatted["content"]["text"] == "Transfer decision"
    assert formatted["model"]["model_id"] == "gpt-4"


def test_format_stream_chunk_real_world_values_subgraphs_true(extractor):
    """Test format_stream_chunk with real-world stream_mode='values' subgraphs=True."""
    msg = MagicMock()
    msg.content = "Full state message"
    msg.response_metadata = {}
    msg.tool_calls = []
    chunk = ((), {'messages': [msg]})
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "gpt-4", stream_mode="values", subgraphs=True
    )
    assert formatted["content"]["text"] == "Full state message"


# ---- Deep Mode Specific Tests ----

def test_extract_messages_deep_mode_values_subgraphs_true_with_files(extractor):
    """Test deep mode: stream_mode='values' subgraphs=True with 'files' key in state."""
    msg = MagicMock()
    msg.content = "Search for hotel"
    # Deep mode includes 'files' key alongside 'messages'
    result = ((), {'files': {}, 'messages': [msg]})
    messages = extractor.extract_messages(result, stream_mode="values", subgraphs=True)
    assert len(messages) == 1
    assert messages[0] == msg
    assert messages[0].content == "Search for hotel"


def test_extract_messages_deep_mode_messages_subgraphs_true_empty_tuple(extractor):
    """Test deep mode: stream_mode='messages' subgraphs=True with empty tuple format.

    Deep mode uses ((), (AIMessage, metadata)) instead of (("node",), (AIMessage, metadata))
    """
    msg = MagicMock()
    msg.content = "Tool call from deep agent"
    metadata = {'langgraph_node': 'model', 'lc_agent_name': 'deep_agent'}
    # Deep mode format: empty tuple followed by (message, metadata)
    result = ((), (msg, metadata))
    messages = extractor.extract_messages(result, stream_mode="messages", subgraphs=True)
    assert len(messages) == 1
    assert messages[0] == msg
    assert messages[0].content == "Tool call from deep agent"


def test_extract_last_message_deep_mode_values_with_files(extractor):
    """Test extract_last_message with deep mode values format containing 'files'."""
    msg = MagicMock()
    msg.content = "Deep agent message"
    result = ((), {'files': {}, 'messages': [msg]})
    last = extractor.extract_last_message(result, stream_mode="values", subgraphs=True)
    assert last == msg
    assert last.content == "Deep agent message"


def test_extract_last_message_deep_mode_messages_empty_tuple(extractor):
    """Test extract_last_message with deep mode messages format with empty tuple."""
    msg = MagicMock()
    msg.content = "Deep agent response"
    metadata = {}
    result = ((), (msg, metadata))
    last = extractor.extract_last_message(result, stream_mode="messages", subgraphs=True)
    assert last == msg
    assert last.content == "Deep agent response"


def test_extract_text_deep_mode_values_with_files(extractor):
    """Test extract_text with deep mode values containing 'files' key."""
    msg = MagicMock()
    msg.content = "Deep mode state text"
    result = ((), {'files': {}, 'messages': [msg]})
    text = extractor.extract_text(result, stream_mode="values", subgraphs=True)
    assert text == "Deep mode state text"


def test_extract_text_deep_mode_messages_empty_tuple(extractor):
    """Test extract_text with deep mode messages format."""
    msg = MagicMock()
    msg.content = "Deep mode message text"
    result = ((), (msg, {}))
    text = extractor.extract_text(result, stream_mode="messages", subgraphs=True)
    assert text == "Deep mode message text"


def test_format_stream_chunk_deep_mode_values_with_files(extractor):
    """Test format_stream_chunk with deep mode values containing 'files'."""
    msg = MagicMock()
    msg.content = "Deep agent state message"
    msg.response_metadata = {}
    msg.tool_calls = []
    chunk = ((), {'files': {}, 'messages': [msg]})
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "nova", stream_mode="values", subgraphs=True
    )
    assert formatted["content"]["text"] == "Deep agent state message"
    assert formatted["model"]["model_id"] == "nova"


def test_format_stream_chunk_deep_mode_messages_empty_tuple(extractor):
    """Test format_stream_chunk with deep mode messages format (empty tuple)."""
    msg = MagicMock()
    msg.content = "Deep agent task instruction"
    msg.response_metadata = {}
    msg.tool_calls = []
    chunk = ((), (msg, {'langgraph_node': 'model', 'lc_agent_name': 'deep_agent'}))
    formatted = extractor.format_stream_chunk(
        chunk, "sess", "nova", stream_mode="messages", subgraphs=True
    )
    assert formatted["content"]["text"] == "Deep agent task instruction"
