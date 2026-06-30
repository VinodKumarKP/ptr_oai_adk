"""Additional coverage for the prometheus metrics module."""
import importlib

import oai_agent_core.components.observability.metrics as metrics


def _reset(module):
    module._configure_attempted = False
    module._metrics_enabled = False
    module._own_server_started = False
    module._instruments = {}


def test_truthy():
    assert metrics._truthy("1")
    assert metrics._truthy("TRUE")
    assert metrics._truthy("yes")
    assert not metrics._truthy("0")
    assert not metrics._truthy("")


def test_record_operation_noop_when_disabled():
    _reset(metrics)
    # Should simply return without raising
    metrics.record_operation("op", agent="a", seconds=1.0)
    assert metrics.is_metrics_enabled() is False


def test_configure_not_available(monkeypatch):
    _reset(metrics)
    monkeypatch.setattr(metrics, "_PROM_AVAILABLE", False)
    assert metrics.configure_metrics() is False


def test_configure_disabled_env(monkeypatch):
    _reset(metrics)
    monkeypatch.setattr(metrics, "_PROM_AVAILABLE", True)
    monkeypatch.delenv("PROMETHEUS_ENABLED", raising=False)
    assert metrics.configure_metrics() is False
    assert metrics._configure_attempted is True


def test_configure_second_call_returns_cached(monkeypatch):
    _reset(metrics)
    monkeypatch.setattr(metrics, "_PROM_AVAILABLE", True)
    metrics._configure_attempted = True
    metrics._metrics_enabled = True
    assert metrics.configure_metrics() is True


def test_configure_enabled_records(monkeypatch):
    if not metrics._PROM_AVAILABLE:
        return
    _reset(metrics)
    monkeypatch.setenv("PROMETHEUS_ENABLED", "true")
    monkeypatch.delenv("PROMETHEUS_PORT", raising=False)
    assert metrics.configure_metrics(default_service_name="svc") is True
    assert metrics.is_metrics_enabled() is True
    # exercise the recording path including instrument creation and duration
    metrics.record_operation("op1", agent="agent", status="ok", seconds=0.1)
    metrics.record_operation("op1", agent="agent", status="error")
    _reset(metrics)


def test_configure_enabled_with_bad_port(monkeypatch):
    if not metrics._PROM_AVAILABLE:
        return
    _reset(metrics)
    monkeypatch.setenv("PROMETHEUS_ENABLED", "1")
    monkeypatch.setenv("PROMETHEUS_PORT", "not-a-port")
    # Should swallow the exception when starting the server
    assert metrics.configure_metrics() is True
    _reset(metrics)
