"""Additional coverage for ResultExtractor."""
from unittest.mock import MagicMock

import pytest

from oai_agent_core.aws_strands_core.processing.result_extractor import (
    ResultExtractor, GraphResult, SwarmResult, AgentResult,
)


@pytest.fixture
def extractor():
    return ResultExtractor()


# ---- extract_text branches ----

def test_extract_text_message_attr(extractor):
    class Obj:
        message = {"content": "msg text"}

    assert extractor.extract_text(Obj()) == "msg text"


def test_extract_text_content_attr(extractor):
    class Obj:
        content = "content text"

    # ensure no 'message' attr
    assert extractor.extract_text(Obj()) == "content text"


def test_extract_text_fallback_str(extractor):
    assert extractor.extract_text(12345) == "12345"


# ---- _safe_get ----

def test_safe_get_none_intermediate(extractor):
    obj = {"a": None}
    assert extractor._safe_get(obj, "a.b", "def") == "def"


def test_safe_get_missing_attr(extractor):
    class Obj:
        pass

    assert extractor._safe_get(Obj(), "x.y", "def") == "def"


# ---- _extract_from_agent_result ----

def test_extract_from_agent_result_structured_output(extractor):
    obj = MagicMock()
    obj.structured_output.model_dump.return_value = {"k": "v"}
    out = extractor._extract_from_agent_result(obj)
    assert out == {"k": "v"}


def test_extract_from_agent_result_message_content(extractor):
    class Obj:
        def __init__(self):
            self.message = type("M", (), {"content": "agent text"})()

    out = extractor._extract_from_agent_result(Obj())
    assert out == "agent text"


def test_extract_from_agent_result_fallback_str(extractor):
    class Obj:
        message = type("M", (), {"content": None})()

    out = extractor._extract_from_agent_result(Obj())
    assert "Obj" in out


def test_extract_from_agent_result_exception(extractor):
    bad = MagicMock()
    bad.structured_output.model_dump.side_effect = RuntimeError("boom")
    out = extractor._extract_from_agent_result(bad)
    assert isinstance(out, str)


# ---- _extract_from_graph_result ----

def test_extract_from_graph_structured_output(extractor):
    result = MagicMock(spec=GraphResult)
    step = MagicMock()
    step.result.result.structured_output.model_dump_json.return_value = '{"x": 1}'
    result.execution_order = [step]
    out = extractor._extract_from_graph_result(result)
    assert out == '{"x": 1}'


def test_extract_from_graph_no_execution_order(extractor):
    result = MagicMock(spec=GraphResult)
    result.execution_order = []
    out = extractor._extract_from_graph_result(result)
    assert isinstance(out, str)


def test_extract_from_graph_fallback_str_result(extractor):
    result = MagicMock(spec=GraphResult)
    step = MagicMock()
    result.execution_order = [step]

    def safe_get(obj, path, default=None):
        if path == "result.result.structured_output":
            return None
        if path in ("result.result.message", "result.message", "message"):
            return None
        if path == "result":
            return "plain string result"
        return default

    extractor._safe_get = safe_get
    out = extractor._extract_from_graph_result(result)
    assert out == "plain string result"


def test_extract_from_graph_exception(extractor):
    result = MagicMock(spec=GraphResult)
    type(result).execution_order = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))
    out = extractor._extract_from_graph_result(result)
    assert isinstance(out, str)


# ---- _extract_from_swarm_result ----

def test_extract_from_swarm_structured_output(extractor):
    result = MagicMock(spec=SwarmResult)
    item = MagicMock()
    item.result.structured_output.model_dump.return_value = {"a": 1}
    result.results = {"node1": item}
    out = extractor._extract_from_swarm_result(result)
    assert "a" in out


def test_extract_from_swarm_node_history(extractor):
    result = MagicMock(spec=SwarmResult)
    result.results = {}
    node = MagicMock()
    node.executor.messages = [{"content": [{"text": "swarm node text"}]}]
    result.node_history = [node]
    out = extractor._extract_from_swarm_result(result)
    assert out == "swarm node text"


def test_extract_from_swarm_str_fallback(extractor):
    result = MagicMock(spec=SwarmResult)
    result.results = {}
    node = MagicMock()
    node.executor.messages = None
    result.node_history = [node]
    out = extractor._extract_from_swarm_result(result)
    assert isinstance(out, str)


def test_extract_from_swarm_exception(extractor):
    result = MagicMock(spec=SwarmResult)
    type(result).results = property(lambda self: (_ for _ in ()).throw(RuntimeError("x")))
    out = extractor._extract_from_swarm_result(result)
    assert isinstance(out, str)


# ---- _extract_from_message ----

def test_extract_from_message_dict(extractor):
    assert extractor._extract_from_message({"content": "dict msg"}) == "dict msg"


def test_extract_from_message_str(extractor):
    assert extractor._extract_from_message("plain") == "plain"


# ---- extract_token_usage non-dict ----

def test_extract_token_usage_non_dict(extractor):
    obj = MagicMock()
    obj.accumulated_usage = "notadict"
    assert extractor.extract_token_usage(obj) == {}
