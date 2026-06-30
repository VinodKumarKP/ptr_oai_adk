import pytest
from unittest.mock import MagicMock, patch
import sys
import types
import time

# Pre-populate sys.modules for opentelemetry exporters that might not be installed
mock_jaeger_exporter_class = MagicMock()
jaeger_module = types.ModuleType('opentelemetry.exporter.jaeger.thrift')
jaeger_module.JaegerExporter = mock_jaeger_exporter_class
sys.modules['opentelemetry.exporter.jaeger.thrift'] = jaeger_module

mock_otlp_exporter_class = MagicMock()
otlp_module = types.ModuleType('opentelemetry.exporter.otlp.proto.grpc.trace_exporter')
otlp_module.OTLPSpanExporter = mock_otlp_exporter_class
sys.modules['opentelemetry.exporter.otlp.proto.grpc.trace_exporter'] = otlp_module

# Mock FastAPIInstrumentor
fastapi_instrumentor_mock = MagicMock()
fastapi_inst_module = types.ModuleType('opentelemetry.instrumentation.fastapi')
fastapi_inst_module.FastAPIInstrumentor = fastapi_instrumentor_mock
sys.modules['opentelemetry.instrumentation.fastapi'] = fastapi_inst_module

import oai_agent_server.utils.observability as obs
from oai_agent_server.utils.observability import (
    ObservabilityManager, ObservabilityConfig, instrument_function
)

@pytest.mark.asyncio
async def test_observability_manager_otel_jaeger_grpc():
    config = ObservabilityConfig()
    config.ENABLE_TRACING = True
    config.JAEGER_ENDPOINT = "http://localhost:14250"
    
    mock_jaeger_exporter_class.reset_mock()
    mock_tracer_provider = MagicMock()
    mock_resource = MagicMock()
    
    obs._OTEL_AVAILABLE = True
    obs._JAEGER_AVAILABLE = True
    obs._OTLP_AVAILABLE = False
    
    with patch('opentelemetry.sdk.trace.TracerProvider', return_value=mock_tracer_provider), \
         patch('opentelemetry.sdk.resources.Resource.create', return_value=mock_resource), \
         patch('opentelemetry.trace.set_tracer_provider'), \
         patch('opentelemetry.trace.get_tracer') as mock_get_tracer:
         
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        mock_jaeger_exporter_class.assert_called_once_with(collector_endpoint="http://localhost:14250")
        assert mock_get_tracer.call_count >= 1

@pytest.mark.asyncio
async def test_observability_manager_otel_jaeger_udp():
    config = ObservabilityConfig()
    config.ENABLE_TRACING = True
    config.JAEGER_ENDPOINT = None
    config.JAEGER_AGENT_HOST = "my-host"
    config.JAEGER_AGENT_PORT = 6831
    
    mock_jaeger_exporter_class.reset_mock()
    
    obs._OTEL_AVAILABLE = True
    obs._JAEGER_AVAILABLE = True
    obs._OTLP_AVAILABLE = False
    
    with patch('opentelemetry.sdk.trace.TracerProvider'), \
         patch('opentelemetry.sdk.resources.Resource.create'), \
         patch('opentelemetry.trace.set_tracer_provider'), \
         patch('opentelemetry.trace.get_tracer'):
         
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        mock_jaeger_exporter_class.assert_called_once_with(
            agent_host_name="my-host",
            agent_port=6831
        )

@pytest.mark.asyncio
async def test_observability_manager_otel_jaeger_failure_otlp_fallback():
    config = ObservabilityConfig()
    config.ENABLE_TRACING = True
    
    mock_jaeger_exporter_class.side_effect = Exception("Jaeger failed")
    mock_otlp_exporter_class.reset_mock()
    
    obs._OTEL_AVAILABLE = True
    obs._JAEGER_AVAILABLE = True
    obs._OTLP_AVAILABLE = True
    
    with patch('opentelemetry.sdk.trace.TracerProvider'), \
         patch('opentelemetry.sdk.resources.Resource.create'), \
         patch('opentelemetry.trace.set_tracer_provider'), \
         patch('opentelemetry.trace.get_tracer'):
         
        manager = ObservabilityManager(config=config, logger=MagicMock())
        mock_otlp_exporter_class.assert_called_once()
    
    mock_jaeger_exporter_class.side_effect = None

@pytest.mark.asyncio
async def test_observability_manager_otel_all_exporters_failed():
    config = ObservabilityConfig()
    config.ENABLE_TRACING = True
    
    logger_mock = MagicMock()
    mock_jaeger_exporter_class.side_effect = Exception("Jaeger failed")
    mock_otlp_exporter_class.side_effect = Exception("OTLP failed")
    
    obs._OTEL_AVAILABLE = True
    obs._JAEGER_AVAILABLE = True
    obs._OTLP_AVAILABLE = True
         
    manager = ObservabilityManager(config=config, logger=logger_mock)
    assert manager.tracer is None
    logger_mock.warning.assert_called()

    mock_jaeger_exporter_class.side_effect = None
    mock_otlp_exporter_class.side_effect = None

def test_trace_operation_with_attributes():
    config = ObservabilityConfig()
    config.ENABLE_TRACING = True
    
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    obs._OTEL_AVAILABLE = True
    
    with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        with manager.trace_operation("test_op", attributes={"attr1": "val1", "attr_err": MagicMock(side_effect=Exception)}) as span:
            assert span == mock_span
            
        mock_span.set_attribute.assert_any_call("attr1", "val1")

def test_metrics_recording():
    config = ObservabilityConfig()
    config.ENABLE_METRICS = True
    
    manager = ObservabilityManager(config=config, logger=MagicMock())
    
    # Pre-populate mock metric instruments
    mock_req_dur = MagicMock()
    mock_req_count = MagicMock()
    mock_err_count = MagicMock()
    mock_db_dur = MagicMock()
    mock_llm_dur = MagicMock()
    mock_stream = MagicMock()
    mock_pool = MagicMock()
    
    manager._prom = {
        'request_duration': mock_req_dur,
        'request_count': mock_req_count,
        'error_count': mock_err_count,
        'db_query_duration': mock_db_dur,
        'llm_judge_duration': mock_llm_dur,
        'streaming_events': mock_stream,
        'db_pool_utilization': mock_pool
    }
    
    # 1. record_request
    manager.record_request("GET", "/chat", 200, 0.5)
    mock_req_dur.labels.assert_called_with(method="GET", path="/chat", status_code="200")
    
    # record error request
    manager.record_request("POST", "/chat", 400, 0.5)
    mock_err_count.labels.assert_called_with(status_code="400", error_type="HTTP_400")
    
    # 2. record_db_query
    manager.record_db_query("SELECT", 0.1, True)
    mock_db_dur.labels.assert_called_with(operation="SELECT", success="True")
    
    # 3. record_llm_judge_evaluation
    manager.record_llm_judge_evaluation(1.2, False)
    mock_llm_dur.labels.assert_called_with(success="False")
    
    # 4. record_streaming_event
    manager.record_streaming_event("message", True)
    mock_stream.labels.assert_called_with(event_type="message", success="True")
    
    # 5. set_pool_utilization
    manager.set_pool_utilization(75.5)
    mock_pool.set.assert_called_with(75.5)

def test_instrument_function_decorator():
    config = ObservabilityConfig()
    config.ENABLE_TRACING = True
    
    mock_tracer = MagicMock()
    mock_span = MagicMock()
    mock_tracer.start_as_current_span.return_value.__enter__.return_value = mock_span
    
    with patch('opentelemetry.trace.get_tracer', return_value=mock_tracer):
        manager = ObservabilityManager(config=config, logger=MagicMock())
        
        @instrument_function(manager)
        def target_func(x):
            return x * 3
            
        @instrument_function(manager)
        def target_error_func():
            raise ValueError("bad val")
            
        res = target_func(5)
        assert res == 15
        
        with pytest.raises(ValueError):
            target_error_func()
