import pytest
from unittest.mock import MagicMock
from datetime import datetime
from oai_agent_server.utils.serialization import make_serializable
from oai_agent_server.utils.response_extractor import ResponseContentExtractor, extract_output_text

def test_make_serializable_basics():
    assert make_serializable(None) is None
    assert make_serializable(1) == 1
    assert make_serializable(1.5) == 1.5
    assert make_serializable(True) is True
    assert make_serializable("s") == "s"

def test_make_serializable_collections():
    assert make_serializable([1, "s"]) == [1, "s"]
    assert make_serializable((1, "s")) == [1, "s"] # Tuples become lists
    assert make_serializable({"a": 1}) == {"a": 1}

def test_make_serializable_objects():
    # Object with __dict__
    class Obj:
        def __init__(self):
            self.x = 1
    assert make_serializable(Obj()) == {"x": 1}
    
    # Object with dict() method
    # The implementation checks hasattr(obj, '__dict__') BEFORE dict()
    # So if the object has __dict__, it uses that.
    # We need to ensure it doesn't have __dict__ or __dict__ is empty/irrelevant if we want to test dict() path?
    # Actually, most python objects have __dict__.
    # If we want to test the dict() path, we might need a class using __slots__ or a mock.
    
    class DictObj:
        __slots__ = [] # No __dict__
        def dict(self): return {"y": 2}
    assert make_serializable(DictObj()) == {"y": 2}
    
    # Object with model_dump() method (Pydantic v2)
    class PydanticObj:
        __slots__ = []
        def model_dump(self): return {"z": 3}
    assert make_serializable(PydanticObj()) == {"z": 3}

def test_make_serializable_special_types():
    # Datetime
    dt = datetime(2023, 1, 1, 12, 0, 0)
    assert make_serializable(dt) == dt.isoformat()
    
    # Bytes
    assert make_serializable(b"hello") == "hello"
    # Bytes that fail decode
    assert make_serializable(b"\xff") == "b'\\xff'"

def test_make_serializable_fallback():
    # Object that fails everything else
    # Needs to NOT have __dict__, dict(), model_dump(), isoformat()
    class FailObj:
        __slots__ = []
        def __str__(self): return "fail_obj"
    
    assert make_serializable(FailObj()) == "fail_obj"

def test_make_serializable_exceptions():
    # Object where dict() raises exception
    class BadDictObj:
        __slots__ = []
        def dict(self): raise Exception("Bad dict")
        def __str__(self): return "bad_dict"
    assert make_serializable(BadDictObj()) == "bad_dict"
    
    # Object where model_dump() raises exception
    class BadModelObj:
        __slots__ = []
        def model_dump(self): raise Exception("Bad model")
        def __str__(self): return "bad_model"
    assert make_serializable(BadModelObj()) == "bad_model"

def test_response_extractor():
    agent = MagicMock()
    # Ensure agent.llm.model_id is set to avoid MagicMock in result
    agent.llm.model_id = "unknown"
    agent.agent_config = {}

    extractor = ResponseContentExtractor(agent)
    
    # Dict response
    resp = {"content": "hello"}
    # The implementation adds model info
    expected = {"content": "hello", "model": {"model_id": "unknown", "provider": "unknown"}}
    assert extractor.extract_content(resp) == expected
    
    # String response
    assert extractor.extract_content("hello") == "hello"

def test_extract_output_text():
    assert extract_output_text("hello") == "hello"
    assert extract_output_text({"content": "hello"}) == "hello"
    assert extract_output_text({"text": "hello"}) == "hello"


# --- Additional ResponseContentExtractor coverage ---

class _FakeAgent:
    def __init__(self, model_id="gpt-x", provider="openai"):
        class _LLM:
            pass
        self.llm = _LLM()
        if model_id is not None:
            self.llm.model_id = model_id
        self.agent_config = {"cloud_provider": provider} if provider else {}


class _AIMessage:
    """Mimics LangChain AIMessage with content + usage_metadata."""
    def __init__(self, content, usage_metadata=None):
        self.content = content
        if usage_metadata is not None:
            self.usage_metadata = usage_metadata


def test_extract_content_object_with_content_attr():
    extractor = ResponseContentExtractor(_FakeAgent())
    msg = _AIMessage("hi there", usage_metadata={"input_tokens": 5, "output_tokens": 2})
    result = extractor.extract_content(msg)
    assert result["content"] == "hi there"
    assert result["token_usage"] == {"input_tokens": 5, "output_tokens": 2}
    assert result["model"]["model_id"] == "gpt-x"
    assert result["model"]["provider"] == "openai"


def test_extract_content_dict_messages_dict_form():
    extractor = ResponseContentExtractor(_FakeAgent())
    resp = {"messages": [{"content": "first"}, {"content": "last reply"}]}
    result = extractor.extract_content(resp)
    assert result["content"] == "last reply"


def test_extract_content_dict_messages_object_form():
    extractor = ResponseContentExtractor(_FakeAgent())
    last_msg = _AIMessage("last reply", usage_metadata={"total_tokens": 7})
    resp = {"messages": [last_msg]}
    result = extractor.extract_content(resp)
    assert result["content"] == "last reply"
    assert result["token_usage"] == {"total_tokens": 7}


def test_extract_content_dict_with_token_usage_key():
    extractor = ResponseContentExtractor(_FakeAgent())
    resp = {"content": "hello", "token_usage": {"total": 4}, "model": "claude"}
    result = extractor.extract_content(resp)
    assert result["content"] == "hello"
    assert result["token_usage"] == {"total": 4}
    assert result["model"] == "claude"


def test_extract_content_list_with_dict_item():
    extractor = ResponseContentExtractor(_FakeAgent())
    resp = [{"content": "first"}, {"content": "last", "usage_metadata": {"total": 10}, "model": "m1"}]
    result = extractor.extract_content(resp)
    assert result["content"] == "last"
    assert result["token_usage"] == {"total": 10}
    assert result["model"] == "m1"


def test_extract_content_list_with_object_item():
    extractor = ResponseContentExtractor(_FakeAgent())
    msg = _AIMessage("from object", usage_metadata={"x": 1})
    resp = [msg]
    result = extractor.extract_content(resp)
    assert result["content"] == "from object"
    assert result["token_usage"] == {"x": 1}


def test_extract_content_fallback_to_string_for_unknown():
    extractor = ResponseContentExtractor(_FakeAgent())
    # Integer has no content/dict/list handling -> str fallback
    assert extractor.extract_content(42) == "42"


def test_extract_content_empty_list_falls_through_to_str():
    extractor = ResponseContentExtractor(_FakeAgent())
    assert extractor.extract_content([]) == "[]"


def test_extract_content_dict_empty_messages_no_content_key():
    extractor = ResponseContentExtractor(_FakeAgent())
    # Dict with neither messages content nor known keys -> falls through to str
    assert extractor.extract_content({"other": "thing"}) == "{'other': 'thing'}"


def test_get_default_model_when_agent_has_no_llm_attr():
    class BareAgent:
        agent_config = {}
    extractor = ResponseContentExtractor(BareAgent())
    msg = _AIMessage("x")
    result = extractor.extract_content(msg)
    assert result["model"]["model_id"] == "unknown"
    assert result["model"]["provider"] == "unknown"


def test_extract_output_text_content_list_of_dicts():
    from oai_agent_server.utils.response_extractor import extract_output_text
    payload = {"content": [{"text": "first"}, {"text": "last"}]}
    assert extract_output_text(payload) == "last"


def test_extract_output_text_content_dict_with_text():
    from oai_agent_server.utils.response_extractor import extract_output_text
    assert extract_output_text({"content": {"text": "nested"}}) == "nested"


def test_extract_output_text_unknown_returns_empty():
    from oai_agent_server.utils.response_extractor import extract_output_text
    assert extract_output_text(123) == ""
    assert extract_output_text({"other": 1}) == ""


def test_extract_chunk_text_string_input():
    from oai_agent_server.utils.response_extractor import extract_chunk_text
    assert extract_chunk_text("hi") == "hi"


def test_extract_chunk_text_dict_with_string_content():
    from oai_agent_server.utils.response_extractor import extract_chunk_text
    assert extract_chunk_text({"content": "hi"}) == "hi"


def test_extract_chunk_text_dict_with_list_content():
    from oai_agent_server.utils.response_extractor import extract_chunk_text
    assert extract_chunk_text({"content": [{"text": "first"}]}) == "first"


def test_extract_chunk_text_unknown_returns_none():
    from oai_agent_server.utils.response_extractor import extract_chunk_text
    assert extract_chunk_text(42) is None
    assert extract_chunk_text({"other": "x"}) is None
