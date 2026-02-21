import pytest
import os
from unittest.mock import patch
from oai_agent_core.components.observability.langfuse_observability_manager import LangfuseObservabilityManager


class TestLangfuseObservabilityManager:
    
    @pytest.fixture
    def manager(self):
        return LangfuseObservabilityManager(agent_name="test_agent")
    
    def test_init_disabled(self, manager):
        assert manager.is_enabled is False
        assert manager.client is None
    
    @patch.dict(os.environ, {
        'LANGFUSE_PUBLIC_KEY': 'test_public_key',
        'LANGFUSE_SECRET_KEY': 'test_secret_key',
        'LANGFUSE_HOST': 'https://test.langfuse.com'
    })
    def test_init_enabled(self):
        with patch('langfuse.get_client'):
            manager = LangfuseObservabilityManager(agent_name="test_agent")
            assert manager.is_enabled is True
    
    def test_is_enabled_false(self, manager):
        assert manager.is_enabled is False
    
    def test_trace_generation_disabled(self, manager):
        with manager.trace_generation("test", "user1", "session1") as span:
            assert span is None
    
    def test_log_error_disabled(self, manager):
        manager.log_error("test message", Exception("test error"))
        # Should not raise any exception
    
    def test_flush_disabled(self, manager):
        manager.flush()
        # Should not raise any exception
    
    @patch.dict(os.environ, {
        'LANGFUSE_PUBLIC_KEY': 'test_key',
        'LANGFUSE_SECRET_KEY': 'test_secret',
        'LANGFUSE_HOST': 'https://test.langfuse.com'
    })
    def test_flush_enabled(self):
        with patch('langfuse.get_client') as mock_get_client:
            mock_client = mock_get_client.return_value
            manager = LangfuseObservabilityManager(agent_name="test_agent")
            manager.flush()
            mock_client.flush.assert_called_once()