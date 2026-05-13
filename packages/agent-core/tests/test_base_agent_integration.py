"""Integration tests for BaseAgent concurrent initialization and core functionality.

Tests cover:
- Concurrent initialization of tools, KB, memory, and guardrails
- Configuration management and deep copy behavior
- State consistency under concurrent operations
- Error handling and graceful degradation
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

from oai_agent_core.core.base_agent import BaseAgent


class ConcreteAgent(BaseAgent):
    """Minimal concrete implementation of BaseAgent for testing."""

    async def initialize(self):
        """Simple initialization for testing."""
        self._initialized = True

    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Echo implementation for testing."""
        return {"content": user_message, "final": True}

    async def astream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Simple streaming implementation."""
        yield {"content": user_message, "final": True}

    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Sync wrapper for testing."""
        import asyncio
        return asyncio.run(self.ainvoke(user_message, config))

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Sync wrapper for testing."""
        async for chunk in self.astream(user_message, config):
            yield chunk


class TestBaseAgentInitialization:
    """Test BaseAgent initialization and configuration management."""

    @pytest.fixture
    def base_config(self):
        """Basic agent configuration for testing."""
        return {
            "type": "custom",
            "name": "test_agent",
            "model": {"name": "gpt-4"},
            "description": "Test agent",
        }

    @pytest.fixture
    def agent(self, base_config):
        """Create a test agent instance."""
        return ConcreteAgent(
            agent_name="test_agent",
            agent_config=base_config,
            llm=MagicMock(),
        )

    def test_agent_initialization(self, agent, base_config):
        """Test basic agent initialization."""
        assert agent.agent_name == "test_agent"
        assert agent.agent_config == base_config
        assert agent.session_id == "default"
        assert agent.user_id == "default"
        assert not agent.is_initialized

    def test_initialization_lock_created(self, agent):
        """Test that initialization lock is created."""
        assert hasattr(agent, "_initialization_lock")
        assert agent._initialization_lock is not None

    @pytest.mark.asyncio
    async def test_initialize_sets_flag(self, agent):
        """Test that initialize() sets the _initialized flag."""
        assert not agent.is_initialized
        await agent.initialize()
        assert agent.is_initialized

    def test_config_deep_copy_behavior(self, agent):
        """Test that config updates don't corrupt nested structures."""
        original_config = {
            "type": "custom",
            "nested": {"level1": {"level2": "value"}},
        }
        agent.agent_config = original_config.copy()

        # Update nested config
        agent.update_config({"nested": {"level1": {"level3": "new_value"}}})

        # Original nested value should still exist (deep merge)
        assert "level2" in agent.agent_config["nested"]["level1"]
        assert agent.agent_config["nested"]["level1"]["level2"] == "value"
        assert agent.agent_config["nested"]["level1"]["level3"] == "new_value"

    def test_augment_system_prompt_deep_copy(self, agent):
        """Test that _augment_system_prompt uses deep copy and doesn't mutate original."""
        config = {
            "system_prompt": "Test prompt",
            "agent_list": [
                {"agent1": {"system_prompt": "Agent 1 prompt", "nested": {"key": "value"}}}
            ],
        }

        # Mock MacroProcessor to just return input
        with patch("oai_agent_core.core.base_agent.MacroProcessor") as mock_processor:
            mock_processor.return_value.process = MagicMock(side_effect=lambda x: x)
            result = agent._augment_system_prompt(config)

        # Original should not be modified
        assert config is not result
        assert config["agent_list"][0]["agent1"]["nested"]["key"] == "value"

    def test_config_value_get_with_dot_notation(self, agent):
        """Test getting config values using dot notation."""
        agent.agent_config = {
            "model": {"temperature": 0.7, "max_tokens": 2000},
            "name": "test",
        }

        assert agent.get_config_value("model.temperature") == 0.7
        assert agent.get_config_value("model.max_tokens") == 2000
        assert agent.get_config_value("name") == "test"
        assert agent.get_config_value("nonexistent", default="default_val") == "default_val"

    def test_config_value_set_with_dot_notation(self, agent):
        """Test setting config values using dot notation."""
        agent.agent_config = {"model": {"temperature": 0.7}}

        agent.set_config_value("model.temperature", 0.9)
        assert agent.agent_config["model"]["temperature"] == 0.9

        agent.set_config_value("model.new_param", "new_value")
        assert agent.agent_config["model"]["new_param"] == "new_value"

    def test_validate_config_requires_type(self, agent):
        """Test that validate_config requires 'type' field."""
        agent.agent_config = {"name": "test_agent"}

        with pytest.raises(ValueError):
            agent.validate_config()

    def test_validate_config_passes_with_type(self, agent, base_config):
        """Test that validate_config passes with required 'type' field."""
        agent.agent_config = base_config
        # Should not raise
        agent.validate_config()


class TestConcurrentInitialization:
    """Test concurrent initialization of tools, KB, memory, and other components."""

    @pytest.mark.asyncio
    async def test_concurrent_component_initialization_no_race(self):
        """Test that concurrent initialization doesn't corrupt state."""
        config = {
            "type": "custom",
            "tools": {"test_tool": {"module": "test_module"}},  # Non-empty to trigger load
            "knowledge_base": [],
            "memory": {},
            "guardrails": {},
            "environment": {"TEST_VAR": "test_value"},
            "skills": {"skill_dir": "./skills"},  # Non-empty to trigger discovery
            "structured_output": {},
        }

        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config=config,
            llm=MagicMock(),
            config_root=str(Path.cwd()),
        )

        # Mock components that would be loaded
        agent.tool_registry = MagicMock()
        agent.tool_registry.load_tools_from_config = MagicMock()
        agent.tool_registry.load_mcp_config = MagicMock()
        agent.tool_registry.update_env = MagicMock(return_value={})
        agent.tool_registry.tools = {}

        agent.skill_registry = MagicMock()
        agent.skill_registry.discover_skills = MagicMock()

        agent.output_model_registry = MagicMock()
        agent.output_model_registry.discover_output_models = MagicMock()

        # Run concurrent initialization
        await agent._load_tools_and_kb_and_memory()

        # Verify components were initialized (no race condition)
        agent.tool_registry.load_tools_from_config.assert_called_once()
        agent.skill_registry.discover_skills.assert_called_once()

    @pytest.mark.asyncio
    async def test_memory_store_initialization(self):
        """Test memory store initialization during concurrent load."""
        config = {
            "type": "custom",
            "memory": {
                "vector_store": {"type": "chroma"},
                "embedding": {"model_id": "test-model"},
                "settings": {"max_recent_turns": 5},
            },
        }

        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config=config,
            llm=MagicMock(),
        )

        # Initialize tool_registry to avoid early return
        agent.tool_registry = MagicMock()
        agent.tool_registry.update_env = MagicMock(return_value={})
        agent.tool_registry.tools = {}

        # Mock the BaseMemoryStore at the import location
        with patch("oai_agent_core.core.base_memory_store.BaseMemoryStore") as mock_memory:
            mock_memory.return_value = MagicMock()
            await agent._load_tools_and_kb_and_memory()

            # Verify memory store was attempted to be created
            mock_memory.assert_called_once()

    @pytest.mark.asyncio
    async def test_component_failure_graceful_degradation(self):
        """Test that individual component failures don't prevent agent initialization."""
        config = {
            "type": "custom",
            "knowledge_base": [],  # KB config (will fail, but gracefully)
            "memory": {},  # Memory config (will fail, but gracefully)
        }

        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config=config,
            llm=MagicMock(),
        )

        # Mock tool registry
        agent.tool_registry = MagicMock()
        agent.tool_registry.load_tools_from_config = MagicMock()
        agent.tool_registry.load_mcp_config = MagicMock()
        agent.tool_registry.update_env = MagicMock(return_value={})

        agent.skill_registry = MagicMock()
        agent.skill_registry.discover_skills = MagicMock()
        agent.output_model_registry = MagicMock()
        agent.output_model_registry.discover_output_models = MagicMock()

        # Should not raise even though components fail
        await agent._load_tools_and_kb_and_memory()

        # Agent should still be usable (graceful degradation)
        assert agent is not None
        assert agent.memory_store is None  # Not created due to error
        assert agent.global_kb_factory is None  # Not created due to error


class TestGuardrails:
    """Test input/output guardrail functionality."""

    @pytest.fixture
    def base_config(self):
        """Basic agent configuration for testing."""
        return {
            "type": "custom",
            "name": "test_agent",
            "model": {"name": "gpt-4"},
            "description": "Test agent",
        }

    @pytest.fixture
    def agent_with_guardrails(self, base_config):
        """Create agent with mocked guardrails manager."""
        return ConcreteAgent(
            agent_name="test_agent",
            agent_config=base_config,
            llm=MagicMock(),
        )

    def test_guardrail_input_with_manager(self, agent_with_guardrails):
        """Test input guardrails when manager is present."""
        mock_manager = MagicMock()
        mock_manager.validate_input = MagicMock(return_value="cleaned_input")
        agent_with_guardrails.guardrails_manager = mock_manager

        result = agent_with_guardrails._guardrail_input_message("test input")

        assert result == "cleaned_input"
        mock_manager.validate_input.assert_called_once_with("test input")

    def test_guardrail_input_without_manager(self, agent_with_guardrails):
        """Test input guardrails when manager is not present."""
        agent_with_guardrails.guardrails_manager = None

        result = agent_with_guardrails._guardrail_input_message("test input")

        assert result == "test input"

    def test_guardrail_output_with_manager(self, agent_with_guardrails):
        """Test output guardrails when manager is present."""
        mock_manager = MagicMock()
        mock_manager.validate_output = MagicMock(return_value="cleaned_output")
        agent_with_guardrails.guardrails_manager = mock_manager

        result = agent_with_guardrails._guardrail_output_message("test output")

        assert result == "cleaned_output"
        mock_manager.validate_output.assert_called_once_with("test output")

    def test_guardrail_output_without_manager(self, agent_with_guardrails):
        """Test output guardrails when manager is not present."""
        agent_with_guardrails.guardrails_manager = None

        result = agent_with_guardrails._guardrail_output_message("test output")

        assert result == "test output"


class TestConfigManagement:
    """Test configuration management and merging."""

    @pytest.fixture
    def config_dict(self):
        """Complex configuration for merging tests."""
        return {
            "type": "custom",
            "model": {"name": "gpt-4", "temperature": 0.7},
            "tools": ["tool1", "tool2"],
            "nested": {"level1": {"level2": {"key": "value"}}},
        }

    def test_update_config_merge(self, config_dict):
        """Test that update_config performs deep merge."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config=config_dict,
            llm=MagicMock(),
        )

        updates = {"model": {"temperature": 0.9}}
        agent.update_config(updates)

        # Original model name should persist
        assert agent.agent_config["model"]["name"] == "gpt-4"
        # Updated value should be present
        assert agent.agent_config["model"]["temperature"] == 0.9

    def test_update_config_with_type_change(self, config_dict):
        """Test that type change is tracked when updated."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config=config_dict,
            agent_type="custom",
            llm=MagicMock(),
        )

        agent.update_config({"type": "crewai"})

        assert agent.agent_config["type"] == "crewai"
        assert agent.agent_type == "crewai"


class TestConfigValidation:
    """Test comprehensive configuration validation."""

    def test_validate_config_missing_type(self):
        """Test that validation fails without 'type' field."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={"model": {"name": "gpt-4"}},
            llm=MagicMock(),
        )

        with pytest.raises(ValueError) as exc_info:
            agent.validate_config()

        assert "Missing required field 'type'" in str(exc_info.value)

    def test_validate_config_invalid_type(self):
        """Test that validation fails with invalid type."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={"type": "invalid_type"},
            llm=MagicMock(),
        )

        with pytest.raises(ValueError) as exc_info:
            agent.validate_config()

        assert "type" in str(exc_info.value).lower()

    def test_validate_config_missing_model_and_llm(self):
        """Test that validation fails without model config or llm."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={"type": "custom"},
            llm=None,
        )

        with pytest.raises(ValueError) as exc_info:
            agent.validate_config()

        assert "model" in str(exc_info.value).lower()

    def test_validate_config_model_missing_name(self):
        """Test that model config must have 'name' field."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={
                "type": "custom",
                "model": {"temperature": 0.7}  # Missing 'name'
            },
            llm=None,
        )

        with pytest.raises(ValueError) as exc_info:
            agent.validate_config()

        assert "name" in str(exc_info.value).lower()

    def test_validate_config_with_llm_provided(self):
        """Test that validation passes when llm is provided."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={"type": "custom"},
            llm=MagicMock(),
        )

        # Should not raise
        agent.validate_config()

    def test_validate_config_kb_missing_vector_store(self):
        """Test that KB config must have vector_store."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={
                "type": "custom",
                "knowledge_base": [
                    {"type": "pdf", "path": "./docs"}  # Missing vector_store
                ]
            },
            llm=MagicMock(),
        )

        with pytest.raises(ValueError) as exc_info:
            agent.validate_config()

        assert "vector_store" in str(exc_info.value).lower()

    def test_validate_config_memory_missing_vector_store(self):
        """Test that memory config must have vector_store."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={
                "type": "custom",
                "memory": {"settings": {"max_recent_turns": 5}}
            },
            llm=MagicMock(),
        )

        with pytest.raises(ValueError) as exc_info:
            agent.validate_config()

        assert "vector_store" in str(exc_info.value).lower()

    def test_validate_config_valid_kb_config(self):
        """Test that valid KB config passes validation."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={
                "type": "custom",
                "knowledge_base": [
                    {
                        "type": "pdf",
                        "path": "./docs",
                        "vector_store": {"type": "chroma"}
                    }
                ]
            },
            llm=MagicMock(),
        )

        # Should not raise
        agent.validate_config()

    def test_validate_config_valid_memory_config(self):
        """Test that valid memory config passes validation."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={
                "type": "custom",
                "memory": {
                    "vector_store": {"type": "chroma"},
                    "embedding": {"model_id": "test"}
                }
            },
            llm=MagicMock(),
        )

        # Should not raise
        agent.validate_config()

    def test_validate_config_invalid_tools_type(self):
        """Test that tools config must be dict."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={
                "type": "custom",
                "tools": ["tool1", "tool2"]  # Should be dict
            },
            llm=MagicMock(),
        )

        with pytest.raises(ValueError) as exc_info:
            agent.validate_config()

        assert "tools" in str(exc_info.value).lower()
        assert "dictionary" in str(exc_info.value).lower()

    def test_validate_config_all_required_fields_present(self):
        """Test that validation passes with all required fields."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={
                "type": "custom",
                "model": {"name": "gpt-4", "temperature": 0.7},
                "tools": {"search": {"module": "tools"}},
                "knowledge_base": [
                    {
                        "type": "pdf",
                        "path": "./docs",
                        "vector_store": {"type": "chroma"}
                    }
                ],
                "memory": {
                    "vector_store": {"type": "chroma"},
                    "embedding": {"model_id": "test"}
                }
            },
            llm=None,
        )

        # Should not raise
        agent.validate_config()


class TestLogging:
    """Test logging behavior and lazy formatting."""

    def test_logger_initialization(self):
        """Test that logger is properly initialized."""
        agent = ConcreteAgent(
            agent_name="test_agent",
            agent_config={"type": "custom"},
            llm=MagicMock(),
        )

        assert agent.logger is not None
        assert hasattr(agent.logger, "info")
        assert hasattr(agent.logger, "error")
        assert hasattr(agent.logger, "warning")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
