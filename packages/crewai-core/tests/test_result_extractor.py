import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
from oai_agent_core.crewai_core.processing.result_extractor import ResultExtractor

@pytest.fixture
def mock_serializer():
    serializer = MagicMock()
    serializer.get_session_outputs.return_value = []
    return serializer

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "gpt-4"
    llm.provider = "openai"
    return llm

@pytest.fixture
def extractor(mock_serializer, mock_llm):
    return ResultExtractor(mock_serializer, mock_llm, MagicMock())

def test_get_response_empty(extractor):
    response = extractor.get_response("sess_1")
    assert response == {}

def test_get_response_with_outputs(extractor, mock_serializer):
    mock_serializer.get_session_outputs.return_value = [{'text': 'msg1'}, {'result': 'msg2'}]
    
    response = extractor.get_response("sess_1")
    
    assert response['result'] == 'msg2'

def test_get_response_with_final_result(extractor, mock_serializer):
    # Mock final_result as an object with token_usage
    final_result = MagicMock()
    final_result.token_usage.__dict__ = {'tokens': 10}
    final_result.raw = "raw_output"

    response = extractor.get_response("sess_1", final_result=final_result)
    
    # Since session outputs are empty (default mock), it goes to the second block
    assert response['result'] == "raw_output"

def test_get_streaming_response(extractor, mock_serializer):
    mock_serializer.get_session_outputs.return_value = [{'text': 'msg1'}, {'result': 'msg2'}]

    chunks = extractor.get_streaming_response("sess_1")
    
    assert len(chunks) == 2
    assert chunks[0]['result'] == 'msg1'
    assert chunks[1]['result'] == 'msg2'

def test_stream_response(extractor, mock_serializer):
    mock_serializer.get_session_outputs.return_value = [{'c': '1'}]
    
    async def run():
        chunks = []
        async for chunk in extractor.stream_response("sess_1"):
            chunks.append(chunk)
            
        assert len(chunks) == 1
        mock_serializer.clear_session.assert_called_once_with("sess_1")
        
    asyncio.run(run())

def test_extract_token_usage(extractor):
    result = MagicMock()
    result.token_usage.__dict__ = {'a': 1}
    usage = extractor.extract_token_usage(result)
    assert usage == {'a': 1}

def test_format_execution_result(extractor):
    res = extractor.format_execution_result("sess_1", "result")
    assert res['session_id'] == "sess_1"
    assert res['result'] == "result"
    assert res['final'] is True

def test_extract_error_details(extractor):
    err = ValueError("oops")
    details = extractor.extract_error_details(err, {'in': 'val'})
    assert details['error_type'] == "ValueError"
    assert details['inputs_provided'] == {'in': 'val'}

def test_get_summary(extractor, mock_serializer):
    mock_serializer.get_session_outputs.return_value = [1, 2]
    summary = extractor.get_summary("sess_1")
    assert summary['step_count'] == 2
    assert summary['has_output'] is True
