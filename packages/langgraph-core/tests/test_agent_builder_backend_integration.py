"""Integration tests for agent_builder backend configuration."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from oai_agent_core.langgraph_core.builders.agent_builder import AgentBuilder
from oai_agent_core.langgraph_core.config import BackendConfig, FilesystemBackendConfig


class TestAgentBuilderBackendIntegration:
    """Test agent_builder integration with Pydantic backend config."""

    @pytest.fixture
    def agent_builder(self):
        """Create an AgentBuilder instance."""
        with patch('oai_agent_core.langgraph_core.builders.agent_builder.BaseAgentBuilder.__init__'):
            builder = AgentBuilder(
                llm=MagicMock(),
                tool_registry=MagicMock(),
                config_root="/tmp",
                model_manager=MagicMock(),
                skill_registry=MagicMock(),
                structured_output_model_registry=MagicMock(),
                logger=MagicMock(),
            )
            # Manually set attributes
            builder.llm = MagicMock()
            builder.tool_registry = MagicMock()
            builder.config_root = "/tmp"
            builder.model_manager = MagicMock()
            builder.logger = MagicMock()
            builder.document_loader = None
            builder.vector_store = None
            builder.skill_registry = MagicMock()
            builder.structured_output_model_registry = MagicMock()
            builder._tool_lock = asyncio.Lock()
            builder._kb_lock = asyncio.Lock()
            return builder

    def test_create_deep_agent_backend_state(self, agent_builder):
        """State backend should return None."""
        result = agent_builder._create_deep_agent_backend("state")
        assert result is None

    def test_create_deep_agent_backend_state_dict(self, agent_builder):
        """State backend dict should return None."""
        result = agent_builder._create_deep_agent_backend({"type": "state"})
        assert result is None

    def test_create_deep_agent_backend_none(self, agent_builder):
        """None should return None."""
        result = agent_builder._create_deep_agent_backend(None)
        assert result is None

    def test_create_deep_agent_backend_empty_dict(self, agent_builder):
        """Empty dict should default to state and return None."""
        result = agent_builder._create_deep_agent_backend({})
        assert result is None

    def test_create_deep_agent_backend_invalid_type_raises(self, agent_builder):
        """Invalid backend type should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            agent_builder._create_deep_agent_backend(123)

    def test_create_deep_agent_backend_invalid_backend_type_raises(self, agent_builder):
        """Invalid backend type string should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid backend"):
            agent_builder._create_deep_agent_backend("invalid_backend")

    def test_create_deep_agent_backend_filesystem_without_config_raises(self, agent_builder):
        """Filesystem type without config should raise ValueError."""
        config = {"type": "filesystem"}  # No 'filesystem' key
        with pytest.raises(ValueError, match="filesystem backend selected"):
            agent_builder._create_deep_agent_backend(config)

    def test_create_deep_agent_backend_store_without_config_raises(self, agent_builder):
        """Store type without config should raise ValueError."""
        config = {"type": "store"}  # No 'store' key
        with pytest.raises(ValueError, match="store backend selected"):
            agent_builder._create_deep_agent_backend(config)

    def test_create_deep_agent_backend_pydantic_validation(self, agent_builder):
        """Invalid Pydantic config should raise ValueError with validation error."""
        config = {"type": "invalid_type"}
        with pytest.raises(ValueError, match="Invalid backend"):
            agent_builder._create_deep_agent_backend(config)
