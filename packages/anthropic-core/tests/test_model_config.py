"""Unit tests for AnthropicModelConfigurationManager."""

import sys
from unittest.mock import MagicMock

import pytest

from oai_agent_core.anthropic_core.components.configuration.model_config import (
    AnthropicModelConfigurationManager,
)


def _mgr(**overrides):
    cfg = {"model_id": "anthropic/claude-opus-4-5-20251101"}
    cfg.update(overrides)
    return AnthropicModelConfigurationManager(default_config=cfg)


# ── get_model_info (pure) ─────────────────────────────────────────────────────

def test_get_model_info_default_provider_anthropic():
    info = _mgr().get_model_info()
    assert info["model_id"] == "anthropic/claude-opus-4-5-20251101"
    assert info["provider"] == "anthropic"
    assert info["via_proxy"] is False


def test_get_model_info_bedrock_provider():
    info = _mgr(model_id="bedrock/anthropic.claude-sonnet-4-5").get_model_info()
    assert info["provider"] == "bedrock"


def test_get_model_info_vertex_provider():
    info = _mgr(model_id="vertex_ai/claude-sonnet-4-5").get_model_info()
    assert info["provider"] == "vertex_ai"


def test_get_model_info_via_proxy_from_raw_config():
    info = _mgr(litellm_proxy_url="http://localhost:4000").get_model_info()
    assert info["via_proxy"] is True


# ── create_model ──────────────────────────────────────────────────────────────

def test_create_model_requires_claude_agent_sdk(monkeypatch):
    # Make `import claude_agent_sdk` fail
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    with pytest.raises(ImportError):
        _mgr().create_model()


def test_create_model_strips_litellm_prefix(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", MagicMock())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    kwargs = _mgr(model_id="anthropic/claude-opus-4-5-20251101").create_model()
    assert kwargs["model"] == "claude-opus-4-5-20251101"
    # no auth/proxy configured -> no env
    assert "env" not in kwargs


def test_create_model_sets_api_key_and_proxy_env(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", MagicMock())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    kwargs = _mgr(api_key="sk-test", litellm_proxy_url="http://localhost:4000").create_model()
    assert kwargs["env"]["ANTHROPIC_API_KEY"] == "sk-test"
    assert kwargs["env"]["ANTHROPIC_BASE_URL"] == "http://localhost:4000"


def test_create_model_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", MagicMock())
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_URL", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    kwargs = _mgr().create_model()  # no api_key in raw config
    assert kwargs["env"]["ANTHROPIC_API_KEY"] == "env-key"


def test_create_model_bare_model_id_unchanged(monkeypatch):
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", MagicMock())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_PROXY_API_KEY", raising=False)
    kwargs = _mgr(model_id="claude-sonnet-4-5").create_model()
    assert kwargs["model"] == "claude-sonnet-4-5"
