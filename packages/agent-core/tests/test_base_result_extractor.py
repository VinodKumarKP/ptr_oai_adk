import pytest
from typing import Any, Dict
from oai_agent_core.processing.base_result_extractor import BaseResultExtractor

class DummyResultExtractor(BaseResultExtractor):
    """Concrete subclass of BaseResultExtractor for testing."""
    def extract_text(self, result: Any) -> str:
        if isinstance(result, dict) and 'text' in result:
            return result['text']
        return str(result)
        
    def extract_token_usage(self, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict) and 'usage' in result:
            return result['usage']
        return {}

def test_base_result_extractor_repr():
    extractor = DummyResultExtractor()
    assert repr(extractor) == "DummyResultExtractor()"

def test_extract_execution_metadata():
    extractor = DummyResultExtractor()
    meta = extractor.extract_execution_metadata("result_string")
    assert meta == {'result_type': 'str'}

def test_format_response_dict():
    extractor = DummyResultExtractor()
    result = {
        "text": "my response",
        "type": "custom_type",
        "tool_calls": [{"name": "get_weather"}],
        "usage": {"input_tokens": 10, "output_tokens": 5}
    }
    
    response = extractor.format_response(
        result=result,
        session_id="session123",
        model_id="gpt-4",
        model_provider="openai",
        include_raw=True,
        input_message="what's the weather?",
        original_message="what's the weather?"
    )
    
    assert response["content"]["text"] == "my response"
    assert response["content"]["type"] == "custom_type"
    assert response["content"]["final"] is True
    assert response["content"]["session_id"] == "session123"
    assert response["model"]["model_id"] == "gpt-4"
    assert response["model"]["model_provider"] == "openai"
    assert response["token_usage"] == {"input_tokens": 10, "output_tokens": 5}
    assert response["raw_result"] == result
    assert response["input_message"] == "what's the weather?"
    assert response["original_message"] == "what's the weather?"
    assert response["tool_calls"] == [{"name": "get_weather"}]
    assert response["metadata"] == {'result_type': 'dict'}

def test_format_streaming_chunk():
    extractor = DummyResultExtractor()
    chunk = extractor.format_streaming_chunk(
        content="hello",
        chunk_type="text",
        agent="writer_agent",
        final=False,
        extra_param=123
    )
    assert chunk["content"] == "hello"
    assert chunk["type"] == "text"
    assert chunk["agent"] == "writer_agent"
    assert chunk["final"] is False
    assert chunk["extra_param"] == 123

def test_extract_from_content():
    extractor = DummyResultExtractor()
    
    # string
    assert extractor._extract_from_content("hello") == "hello"
    
    # list of strings
    assert extractor._extract_from_content(["hello", " ", "world"]) == "hello world"
    
    # list of dicts
    assert extractor._extract_from_content([{"text": "hello"}, {"text": " world"}]) == "hello world"
    
    # list of objects with text attribute
    class Block:
        def __init__(self, text):
            self.text = text
    assert extractor._extract_from_content([Block("hello"), Block(" world")]) == "hello world"
    
    # dict
    assert extractor._extract_from_content({"text": "hello dict"}) == "hello dict"
    
    # fallback
    assert extractor._extract_from_content(123) == "123"

def test_serialize_result():
    extractor = DummyResultExtractor()
    
    # None
    assert extractor._serialize_result(None) is None
    
    # Scalars
    assert extractor._serialize_result(123) == 123
    assert extractor._serialize_result("hello") == "hello"
    
    # List/tuple
    assert extractor._serialize_result([1, "2"]) == [1, "2"]
    
    # Dict
    assert extractor._serialize_result({"key": "val"}) == {"key": "val"}
    
    # Object with __dict__
    class CustomObj:
        def __init__(self):
            self.param = "val"
            self._private = "secret"
    obj = CustomObj()
    assert extractor._serialize_result(obj) == {"param": "val"}
    
    # fallback to str
    assert extractor._serialize_result(1 + 2j) == str(1 + 2j)
