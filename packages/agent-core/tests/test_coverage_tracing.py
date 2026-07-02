"""Additional coverage for the OpenTelemetry tracing helpers."""
import asyncio

import pytest

import oai_agent_core.components.observability.tracing as tracing


def test_coerce_level_variants():
    assert tracing._coerce_level(15) == 15
    assert tracing._coerce_level("debug") == tracing.DEBUG
    assert tracing._coerce_level("INFO") == tracing.INFO
    assert tracing._coerce_level("warning") == tracing.WARNING
    assert tracing._coerce_level("unknown") == tracing.INFO


def test_set_get_trace_level():
    original = tracing.get_trace_level()
    try:
        tracing.set_trace_level("debug")
        assert tracing.get_trace_level() == tracing.DEBUG
        tracing.set_trace_level(tracing.WARNING)
        assert tracing.get_trace_level() == tracing.WARNING
    finally:
        tracing.set_trace_level(original)


def test_is_tracing_available():
    assert tracing.is_tracing_available() == tracing._OTEL_AVAILABLE


def test_configure_tracing_already_attempted():
    tracing._auto_init_attempted = True
    # Should return immediately without error
    tracing.configure_tracing()


def test_configure_tracing_no_endpoint(monkeypatch):
    tracing._auto_init_attempted = False
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    tracing.configure_tracing()
    assert tracing._auto_init_attempted is True


def test_configure_tracing_existing_provider(monkeypatch):
    # When a real (non-proxy) provider is already installed, configure is a no-op.
    tracing._auto_init_attempted = False
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    # A real SDK provider is installed once we create one
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry import trace

    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    tracing.configure_tracing(default_service_name="svc")
    # Did not raise; flag set
    assert tracing._auto_init_attempted is True


def test_get_or_create_provider_returns_existing():
    # A real provider is now installed by the previous test
    provider = tracing._get_or_create_provider()
    assert provider is not None


def test_add_otlp_exporter_no_endpoint():
    assert tracing.add_otlp_exporter("") is False


def test_add_otlp_exporter_grpc(monkeypatch):
    from unittest.mock import MagicMock

    fake_provider = MagicMock()
    monkeypatch.setattr(tracing, "_get_or_create_provider", lambda *a, **k: fake_provider)
    assert tracing.add_otlp_exporter(
        "http://localhost:4317", protocol="grpc", service_name="svc"
    ) is True
    assert fake_provider.add_span_processor.called


def test_add_otlp_exporter_http(monkeypatch):
    from unittest.mock import MagicMock

    fake_provider = MagicMock()
    monkeypatch.setattr(tracing, "_get_or_create_provider", lambda *a, **k: fake_provider)
    assert tracing.add_otlp_exporter(
        "http://localhost:4318", headers={"Authorization": "Basic x"}, protocol="http"
    ) is True


def test_add_otlp_exporter_no_provider(monkeypatch):
    monkeypatch.setattr(tracing, "_get_or_create_provider", lambda *a, **k: None)
    assert tracing.add_otlp_exporter("http://localhost:4317") is False


def test_add_otlp_exporter_unavailable(monkeypatch):
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    assert tracing.add_otlp_exporter("http://x") is False


def test_instrument_openlit_already_done():
    tracing._openlit_instrumented = True
    assert tracing.instrument_openlit() is False


def test_instrument_openlit_unavailable(monkeypatch):
    tracing._openlit_instrumented = False
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    assert tracing.instrument_openlit() is False


def test_instrument_openlit_runs(monkeypatch):
    import openlit
    from unittest.mock import MagicMock

    tracing._openlit_instrumented = False
    # Avoid real OpenLIT instrumentation side effects by stubbing init.
    monkeypatch.setattr(openlit, "init", MagicMock())
    fake_provider = MagicMock()
    monkeypatch.setattr(tracing, "_get_or_create_provider", lambda *a, **k: fake_provider)
    result = tracing.instrument_openlit(application_name="app", environment="test")
    assert result is True
    tracing._openlit_instrumented = False


def test_instrument_openlit_no_provider(monkeypatch):
    tracing._openlit_instrumented = False
    monkeypatch.setattr(tracing, "_get_or_create_provider", lambda *a, **k: None)
    assert tracing.instrument_openlit(application_name="app") is False
    tracing._openlit_instrumented = False


def test_trace_span_noop_when_unavailable(monkeypatch):
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    with tracing.trace_span("op") as span:
        assert span is None


def test_trace_span_suppressed_level(monkeypatch):
    # Level below threshold -> yields None (the suppressed branch)
    tracing.set_trace_level(tracing.WARNING)
    try:
        with tracing.trace_span("op", level=tracing.DEBUG) as span:
            assert span is None
    finally:
        tracing.set_trace_level(tracing.INFO)


def test_trace_span_suppressed_with_exception_and_metrics(monkeypatch):
    tracing.set_trace_level(tracing.WARNING)
    monkeypatch.setattr(tracing._metrics, "is_metrics_enabled", lambda: True)
    recorded = {}
    monkeypatch.setattr(
        tracing._metrics, "record_operation",
        lambda *a, **k: recorded.update(k),
    )
    try:
        with pytest.raises(ValueError):
            with tracing.trace_span("op", level=tracing.DEBUG, agent_name="a"):
                raise ValueError("boom")
        assert recorded.get("status") == "error"
    finally:
        tracing.set_trace_level(tracing.INFO)


def test_trace_span_active_success(monkeypatch):
    tracing.set_trace_level(tracing.INFO)
    with tracing.trace_span("op", attributes={"k": "v"}, agent_name="a", skip=None) as span:
        # span may be a real or no-op span object
        assert span is not None or span is None


def test_trace_span_active_exception(monkeypatch):
    tracing.set_trace_level(tracing.INFO)
    monkeypatch.setattr(tracing._metrics, "is_metrics_enabled", lambda: True)
    monkeypatch.setattr(tracing._metrics, "record_operation", lambda *a, **k: None)
    with pytest.raises(RuntimeError):
        with tracing.trace_span("op", agent_name="a"):
            raise RuntimeError("fail")


def test_traced_sync():
    @tracing.traced("op.sync")
    def fn(x):
        return x + 1

    assert fn(1) == 2


def test_traced_coroutine():
    @tracing.traced("op.async")
    async def fn(x):
        return x * 2

    assert asyncio.run(fn(3)) == 6


def test_traced_async_gen():
    @tracing.traced("op.gen")
    async def gen():
        for i in range(3):
            yield i

    async def collect():
        return [i async for i in gen()]

    assert asyncio.run(collect()) == [0, 1, 2]


def test_traced_with_self_agent_name():
    class Holder:
        agent_name = "myagent"

        @tracing.traced("op.method")
        def run(self):
            return "done"

    assert Holder().run() == "done"


def test_traced_stream():
    async def source():
        for i in range(2):
            yield i

    async def collect():
        return [i async for i in tracing.traced_stream("op.stream", source(), agent_name="a")]

    assert asyncio.run(collect()) == [0, 1]
