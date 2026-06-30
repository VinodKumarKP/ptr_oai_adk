"""Additional coverage for LangfuseObservabilityManager."""
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest

from oai_agent_core.components.observability.langfuse_observability_manager import (
    LangfuseObservabilityManager,
)


@pytest.fixture
def lf_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_HOST", "http://localhost:3000")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    # Skip instrumentation side effects by default
    LangfuseObservabilityManager._instrumented = True
    yield
    LangfuseObservabilityManager._instrumented = False


def _make(monkeypatch, framework=None):
    return LangfuseObservabilityManager("agent", logger=MagicMock(), framework=framework)


def test_check_env_missing(monkeypatch):
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    LangfuseObservabilityManager._instrumented = True
    mgr = LangfuseObservabilityManager("agent", logger=MagicMock())
    assert mgr.is_enabled is False
    LangfuseObservabilityManager._instrumented = False


def test_initialize_client_success(lf_env):
    mgr = _make(None)
    assert mgr.is_enabled is True


def test_initialize_callback_handler_path(lf_env, monkeypatch):
    mgr = _make(monkeypatch, framework="langchain")
    # callback handler initialised via mocked langfuse modules
    assert mgr.framework == "langchain"


def test_trace_generation_disabled():
    LangfuseObservabilityManager._instrumented = True
    mgr = LangfuseObservabilityManager("agent", logger=MagicMock())
    mgr.client = None
    with mgr.trace_generation("hi", "u", "s") as span:
        assert span is None
    LangfuseObservabilityManager._instrumented = False


def test_trace_generation_enabled(lf_env):
    mgr = _make(None)
    fake_span = MagicMock()

    @contextmanager
    def fake_obs(**kwargs):
        yield fake_span

    mgr.client = MagicMock()
    mgr.client.start_as_current_observation = fake_obs
    with mgr.trace_generation("hi", "u", "s", metadata={"x": 1}, name_suffix="-stream") as span:
        assert span is fake_span
    assert mgr.client.flush.called


def test_update_trace_enabled(lf_env):
    mgr = _make(None)
    mgr.client = MagicMock()
    span = MagicMock()
    mgr.update_trace(span, "in", "out", "u", "s", tags=["t"], metadata={"m": 1})
    assert span.update_trace.called


def test_update_trace_disabled_or_no_span(lf_env):
    mgr = _make(None)
    mgr.client = None
    mgr.update_trace(None, "in", "out", "u", "s")  # no error


def test_log_error_enabled(lf_env):
    mgr = _make(None)
    fake_span = MagicMock()

    @contextmanager
    def fake_obs(**kwargs):
        yield fake_span

    mgr.client = MagicMock()
    mgr.client.start_as_current_observation = fake_obs
    mgr.log_error("user msg", ValueError("boom"), metadata={"k": "v"})
    assert fake_span.update.called
    assert mgr.client.flush.called


def test_log_error_handles_exception(lf_env):
    mgr = _make(None)
    mgr.client = MagicMock()
    mgr.client.start_as_current_observation = MagicMock(side_effect=RuntimeError("x"))
    mgr.log_error("msg", ValueError("boom"))  # swallowed
    assert mgr.logger.debug.called


def test_log_error_disabled(lf_env):
    mgr = _make(None)
    mgr.client = None
    mgr.log_error("msg", ValueError("boom"))  # no-op


def test_flush_handles_exception(lf_env):
    mgr = _make(None)
    mgr.client = MagicMock()
    mgr.client.flush = MagicMock(side_effect=RuntimeError("x"))
    mgr.flush()  # swallowed
    assert mgr.logger.debug.called


def test_setup_instrumentation_langfuse_only(lf_env, monkeypatch):
    import openlit

    LangfuseObservabilityManager._instrumented = False
    monkeypatch.setattr(openlit, "init", MagicMock())
    mgr = LangfuseObservabilityManager("agent", logger=MagicMock())
    assert LangfuseObservabilityManager._instrumented is True
    LangfuseObservabilityManager._instrumented = False


def test_setup_instrumentation_with_otel_endpoint(lf_env, monkeypatch):
    from oai_agent_core.components.observability import tracing

    LangfuseObservabilityManager._instrumented = False
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    monkeypatch.setattr(tracing, "configure_tracing", MagicMock())
    monkeypatch.setattr(tracing, "add_otlp_exporter", MagicMock())
    monkeypatch.setattr(tracing, "instrument_openlit", MagicMock())
    mgr = LangfuseObservabilityManager("agent", logger=MagicMock())
    assert tracing.configure_tracing.called
    LangfuseObservabilityManager._instrumented = False
