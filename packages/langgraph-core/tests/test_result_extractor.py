import pytest
from unittest.mock import MagicMock
from oai_agent_core.langgraph_core.processing.result_extractor import ResultExtractor
from langchain_core.messages import AIMessage

@pytest.fixture
def extractor():
    return ResultExtractor()

def test_extract_last_message_dict(extractor):
    result = {'messages': [{'content': 'msg1'}, {'content': 'msg2'}]}
    last_msg = extractor.extract_last_message(result)
    assert last_msg == {'content': 'msg2'}

def test_extract_last_message_empty(extractor):
    result = {'messages': []}
    last_msg = extractor.extract_last_message(result)
    assert last_msg is None

def test_extract_last_message_invalid(extractor):
    result = {}
    last_msg = extractor.extract_last_message(result)
    assert last_msg is None

def test_extract_token_usage(extractor):
    usage = {'input_tokens': 10, 'output_tokens': 20, 'total_tokens': 30}
    # Create AIMessage and manually set usage_metadata if constructor doesn't support it
    msg = AIMessage(content="test")
    msg.usage_metadata = usage
    
    result = {
        'messages': [msg]
    }
    extracted_usage = extractor.extract_token_usage(result)
    assert extracted_usage == usage

def test_extract_token_usage_dict(extractor):
    usage = {'input_tokens': 10}
    result = {
        'messages': [
            {'content': 'test', 'usage_metadata': usage}
        ]
    }
    extracted_usage = extractor.extract_token_usage(result)
    assert extracted_usage == usage

def test_extract_token_usage_none(extractor):
    result = {'messages': [{'content': 'test'}]}
    extracted_usage = extractor.extract_token_usage(result)
    assert extracted_usage == {}

def test_format_response_string_content(extractor):
    result = {
        'messages': [
            AIMessage(content="Hello world")
        ]
    }
    response = extractor.format_response(result, "sess_1", "gpt-4")
    assert response['content']['text'] == "Hello world"
    assert response['content']['final'] is True
    assert response['content']['session_id'] == "sess_1"
    assert response['model']['model_id'] == "gpt-4"

def test_format_response_list_content(extractor):
    content = [{'type': 'text', 'text': 'Hello'}]
    result = {
        'messages': [
            AIMessage(content=content)
        ]
    }
    response = extractor.format_response(result, "sess_1", "gpt-4")
    assert response['content']['final'] is True
    assert response['content']['session_id'] == "sess_1"

def test_format_response_dict_content(extractor):
    # Test case for dict-based message (not AIMessage object)
    content = [{'type': 'text', 'text': 'Hello'}]
    result = {
        'messages': [
            {'content': content}
        ]
    }
    response = extractor.format_response(result, "sess_1", "gpt-4")
    assert response['content']['final'] is True
    assert response['content']['session_id'] == "sess_1"

def test_format_response_dict_content_simple(extractor):
    # Test case for dict-based message with simple content
    result = {
        'messages': [
            {'content': 'simple'}
        ]
    }
    response = extractor.format_response(result, "sess_1", "gpt-4")
    assert response['content'] == {'final': True, 'session_id': 'sess_1', 'text': 'simple', 'type': 'text'}

def test_format_response_fallback(extractor):
    # Test fallback path for 'else' in content type checks (neither list nor str)
    # Use a mock object since AIMessage enforces types
    mock_msg = MagicMock()
    mock_msg.content = 123
    
    result = {
        'messages': [mock_msg]
    }
    response = extractor.format_response(result, "sess_1", "gpt-4")
    assert response['content'] == {'final': True, 'session_id': 'sess_1', 'text': 123, 'type': 'text'}

def test_format_stream_chunk(extractor):
    chunk = {'messages': [{'content': 'chunk'}]}
    formatted = extractor.format_stream_chunk(chunk, "sess_1", "gpt-4")
    assert formatted['content'] == {'final': False, 'session_id': 'sess_1', 'text': 'chunk', 'type': 'dict'}
    assert formatted['model']['model_id'] == "gpt-4"

def test_format_stream_chunk_aimessage(extractor):
    chunk = {'messages': [AIMessage(content='chunk')]}
    formatted = extractor.format_stream_chunk(chunk, "sess_1", "gpt-4")
    assert formatted['content'] == {'final': False, 'session_id': 'sess_1', 'text': 'chunk', 'type': 'AIMessage'}

def test_format_stream_chunk_usage(extractor):
    usage = {'input_tokens': 5}
    chunk = {'messages': [{'content': 'chunk', 'usage_metadata': usage}]}
    formatted = extractor.format_stream_chunk(chunk, "sess_1", "gpt-4")
    assert formatted['token_usage'] == usage

def test_format_stream_chunk_usage_aimessage(extractor):
    usage = {'input_tokens': 5}
    msg = AIMessage(content='chunk')
    msg.usage_metadata = usage
    chunk = {'messages': [msg]}
    formatted = extractor.format_stream_chunk(chunk, "sess_1", "gpt-4")
    assert formatted['token_usage'] == usage

def test_format_stream_chunk_no_message(extractor):
    chunk = "raw_chunk"
    formatted = extractor.format_stream_chunk(chunk, "sess_1", "gpt-4")
    assert formatted['content'] == {'final': False, 'session_id': 'sess_1', 'text': 'raw_chunk', 'type': 'text'}
    assert formatted['token_usage'] == {}

def test_format_stream_chunk_final(extractor):
    chunk = "raw_chunk"
    formatted = extractor.format_stream_chunk(chunk, "sess_1", "gpt-4", is_final=True)
    assert formatted['content']['final'] is True
