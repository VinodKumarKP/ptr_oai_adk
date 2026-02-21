import pytest
from unittest.mock import MagicMock
from oai_agent_core.openai_core.processing.result_extractor import ResultExtractor

@pytest.fixture
def extractor():
    return ResultExtractor()

def test_extract_text_run_result(extractor):
    mock_result = MagicMock()
    mock_result.final_output = "test output"
    assert extractor.extract_text(mock_result) == "test output"

def test_extract_text_dict(extractor):
    assert extractor.extract_text({'output': 'test'}) == 'test'
    assert extractor.extract_text({'content': 'test'}) == 'test'
    assert extractor.extract_text({'result': 'test'}) == 'test'

def test_extract_text_message(extractor):
    mock_msg = MagicMock()
    mock_msg.content = "test"
    # Ensure the mock doesn't have other attributes that might confuse the extractor
    del mock_msg.final_output
    assert extractor.extract_text(mock_msg) == "test"

def test_extract_from_content_list(extractor):
    content = [
        {'text': 'part1'},
        MagicMock(text='part2'),
        'part3'
    ]
    assert extractor._extract_from_content(content) == "part1part2part3"

def test_extract_token_usage(extractor):
    mock_result = MagicMock()
    mock_result.context_wrapper.usage.input_tokens = 10
    mock_result.context_wrapper.usage.output_tokens = 20
    mock_result.context_wrapper.usage.total_tokens = 30
    
    usage = extractor.extract_token_usage(mock_result)
    assert usage['input_tokens'] == 10
    assert usage['total_tokens'] == 30

def test_format_response(extractor):
    mock_result = MagicMock()
    mock_result.final_output = "output"
    mock_result.context_wrapper.usage.total_tokens = 10
    
    response = extractor.format_response(
        mock_result,
        session_id="sess1",
        model_id="gpt-4",
        include_raw=True
    )
    
    assert response['content']['text'] == "output"
    assert response['model']['model_id'] == "gpt-4"
    assert response['token_usage']['total_tokens'] == 10
    assert 'raw_result' in response

def test_format_streaming_chunk(extractor):
    chunk = extractor.format_streaming_chunk(
        "chunk",
        agent="agent1",
        final=False
    )
    
    assert chunk['content'] == "chunk"
    assert chunk['agent'] == "agent1"
    assert chunk['final'] is False
