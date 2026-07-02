"""Simple targeted coverage for openai-core components."""
from unittest.mock import MagicMock

import pytest

from oai_agent_core.openai_core.processing.result_extractor import ResultExtractor
from oai_agent_core.openai_core.components.configuration.model_config import OpenAIModelConfigurationManager


@pytest.fixture
def extractor():
    return ResultExtractor()


# ---- ResultExtractor coverage ----

def test_extract_text_with_dict(extractor):
    result = {"content": "text"}
    assert extractor.extract_text(result) == "text"


def test_extract_text_empty(extractor):
    # An empty dict has no 'output'/'content'/'result' key, so falls through
    # to str(result) which returns '{}'
    assert extractor.extract_text({}) == "{}"


def test_extract_token_usage_with_data(extractor):
    result = {"usage": {"prompt_tokens": 5}}
    usage = extractor.extract_token_usage(result)
    assert usage["prompt_tokens"] == 5


def test_extract_token_usage_empty(extractor):
    assert extractor.extract_token_usage({}) == {}


def test_format_response_content(extractor):
    result = {"content": "resp"}
    formatted = extractor.format_response(result, "s1", "m1", "p1")
    assert formatted["content"]["text"] == "resp"
    assert formatted["content"]["session_id"] == "s1"


def test_format_response_with_raw(extractor):
    result = {"content": "resp"}
    formatted = extractor.format_response(result, "s1", "m1", "p1", include_raw=True)
    assert formatted["raw_result"] is not None


# ---- ModelConfig simple coverage ----

def test_model_manager_get_model():
    """Test basic model manager initialization."""
    # BaseModelConfigurationManager.__init__ takes default_config and logger, not config_root
    manager = OpenAIModelConfigurationManager(default_config=None, logger=MagicMock())
    # Just test it initializes
    assert manager is not None
