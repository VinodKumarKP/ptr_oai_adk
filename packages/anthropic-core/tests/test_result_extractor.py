"""Unit tests for the anthropic_core ResultExtractor."""

from types import SimpleNamespace

import pytest

from oai_agent_core.anthropic_core.processing.result_extractor import ResultExtractor


@pytest.fixture
def extractor():
    return ResultExtractor()


# ── extract_text ──────────────────────────────────────────────────────────────

def test_extract_text_none_returns_empty(extractor):
    assert extractor.extract_text(None) == ""


def test_extract_text_plain_string(extractor):
    assert extractor.extract_text("hello world") == "hello world"


def test_extract_text_dict_with_text_key(extractor):
    assert extractor.extract_text({"text": "answer", "raw": object()}) == "answer"


def test_extract_text_dict_empty_text_falls_back_to_raw(extractor):
    raw = SimpleNamespace(result="from-raw")
    assert extractor.extract_text({"text": "", "raw": raw}) == "from-raw"


def test_extract_text_dict_missing_text(extractor):
    # No "text" key and no "raw" -> empty string
    assert extractor.extract_text({}) == ""


def test_extract_text_raw_message_object(extractor):
    # A non-str, non-dict object is routed through _from_raw
    msg = SimpleNamespace(result="final-result")
    assert extractor.extract_text(msg) == "final-result"


# ── _from_raw ─────────────────────────────────────────────────────────────────

def test_from_raw_none(extractor):
    assert extractor._from_raw(None) == ""


def test_from_raw_result_attribute(extractor):
    msg = SimpleNamespace(result="done")
    assert extractor._from_raw(msg) == "done"


def test_from_raw_content_string(extractor):
    msg = SimpleNamespace(content="plain content")
    assert extractor._from_raw(msg) == "plain content"


def test_from_raw_content_list_of_text_blocks(extractor):
    blocks = [SimpleNamespace(text="line1"), {"type": "text", "text": "line2"}]
    msg = SimpleNamespace(content=blocks)
    assert extractor._from_raw(msg) == "line1\nline2"


def test_from_raw_content_list_ignores_non_text_dict(extractor):
    blocks = [{"type": "image", "url": "x"}, {"type": "text", "text": "kept"}]
    msg = SimpleNamespace(content=blocks)
    # non-text blocks (no .text attr, type != "text") are skipped entirely
    assert extractor._from_raw(msg) == "kept"


def test_from_raw_fallback_str(extractor):
    # Object with neither result nor content -> str()
    assert extractor._from_raw(123) == "123"


# ── format_response ───────────────────────────────────────────────────────────

def test_format_response_basic_shape(extractor):
    resp = extractor.format_response("hi", session_id="s1", model_id="claude-x")
    # Nested platform-standard schema
    assert resp["content"]["text"] == "hi"
    assert resp["content"]["type"] == "AIMessage"
    assert resp["content"]["final"] is True
    assert resp["content"]["session_id"] == "s1"
    assert resp["model"] == {"model_id": "claude-x", "model_provider": "anthropic"}
    assert resp["metadata"]["result_type"] == "str"
    assert "raw_result" not in resp
    assert "input_message" not in resp
    assert "original_message" not in resp


def test_format_response_custom_provider(extractor):
    resp = extractor.format_response("hi", session_id="s", model_id="m", model_provider="bedrock")
    assert resp["model"]["model_provider"] == "bedrock"


def test_format_response_include_raw(extractor):
    raw_obj = {"text": "hi"}
    resp = extractor.format_response(raw_obj, session_id="s", model_id="m", include_raw=True)
    assert resp["raw_result"] is raw_obj


def test_format_response_includes_token_usage(extractor):
    raw = SimpleNamespace(usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    resp = extractor.format_response({"text": "hi", "raw": raw}, session_id="s", model_id="m")
    assert resp["token_usage"] == {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}


def test_format_response_omits_token_usage_when_absent(extractor):
    resp = extractor.format_response({"text": "hi", "raw": None}, session_id="s", model_id="m")
    assert "token_usage" not in resp


def test_format_response_includes_optional_messages(extractor):
    resp = extractor.format_response(
        "hi", session_id="s", model_id="m",
        input_message="augmented", original_message="orig",
    )
    assert resp["input_message"] == "augmented"
    assert resp["original_message"] == "orig"


def test_format_response_extracts_text_from_result(extractor):
    resp = extractor.format_response({"text": "extracted"}, session_id="s", model_id="m")
    assert resp["content"]["text"] == "extracted"


# ── format_streaming_chunk ────────────────────────────────────────────────────

def test_format_streaming_chunk_defaults(extractor):
    chunk = extractor.format_streaming_chunk("partial")
    assert chunk["content"] == {"text": "partial", "type": "AIMessage"}
    assert chunk["chunk_type"] == "text_delta"
    assert chunk["final"] is False
    assert "agent" not in chunk


def test_format_streaming_chunk_with_agent_and_final(extractor):
    chunk = extractor.format_streaming_chunk("done", chunk_type="message_stop", agent="travel", final=True)
    assert chunk["agent"] == "travel"
    assert chunk["final"] is True
    assert chunk["chunk_type"] == "message_stop"
