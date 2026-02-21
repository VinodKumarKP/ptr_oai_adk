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
