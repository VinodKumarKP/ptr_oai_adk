import pytest
import logging
from unittest.mock import MagicMock, patch, AsyncMock

from oai_agent_core.services import (
    ConfigResolverService,
    KnowledgeBaseService,
    ToolService,
    SkillService,
    ModelService,
    ObservabilityService,
    MemoryService,
)
from oai_agent_core.utils.exceptions import ConfigurationError

# ============================================================================
# ConfigResolverService Tests
# ============================================================================

class TestConfigResolverService:
    def test_init_and_cache(self):
        service = ConfigResolverService(config_root="/my/root", enable_cache=True)
        assert service.config_root == "/my/root"
        assert service.enable_cache is True
        
        # Test clear cache
        service._config_cache["agentA"] = {"type": "test"}
        service.clear_cache()
        assert service._config_cache == {}

    @patch("oai_agent_core.components.configuration.model_config.ConfigManager")
    def test_load_agent_config(self, mock_config_manager_cls):
        mock_manager = MagicMock()
        mock_manager.load_agent_config.return_value = {"type": "test_agent"}
        mock_config_manager_cls.return_value = mock_manager

        service = ConfigResolverService(enable_cache=True)
        
        # First load (disk)
        config = service.load_agent_config("agentA")
        assert config == {"type": "test_agent"}
        mock_manager.load_agent_config.assert_called_once_with("agentA")
        
        # Second load (cache)
        config_cached = service.load_agent_config("agentA")
        assert config_cached == {"type": "test_agent"}
        # Should not call config_manager load again
        assert mock_manager.load_agent_config.call_count == 1

    @patch("oai_agent_core.components.configuration.model_config.ConfigManager")
    def test_load_agent_config_failure(self, mock_config_manager_cls):
        mock_manager = MagicMock()
        mock_manager.load_agent_config.side_effect = Exception("File missing")
        mock_config_manager_cls.return_value = mock_manager

        service = ConfigResolverService()
        with pytest.raises(ConfigurationError):
            service.load_agent_config("nonexistent")

    @patch("oai_agent_core.macros.MacroProcessor")
    def test_resolve_macros(self, mock_processor_cls):
        mock_proc = MagicMock()
        mock_proc.process.return_value = "processed prompt"
        mock_processor_cls.return_value = mock_proc

        service = ConfigResolverService()
        config = {"system_prompt": "hello ${VAR}"}
        
        res = service.resolve_macros(config)
        assert res["system_prompt"] == "processed prompt"
        
        # Test exception fallback
        mock_proc.process.side_effect = Exception("failed macro")
        res_fallback = service.resolve_macros({"system_prompt": "hello ${VAR}"})
        assert res_fallback["system_prompt"] == "hello ${VAR}"

    def test_validate_config(self):
        service = ConfigResolverService()
        
        # Valid config
        assert service.validate_config({"type": "crewai"}) is True
        
        # Missing type
        assert service.validate_config({"model": {}}) is False
        
        # Invalid model type
        assert service.validate_config({"type": "langchain", "model": "not_a_dict"}) is False
        
        # Invalid tools type
        assert service.validate_config({"type": "langchain", "tools": "not_a_dict"}) is False

# ============================================================================
# KnowledgeBaseService Tests
# ============================================================================

class TestKnowledgeBaseService:
    def test_knowledge_base_methods(self):
        service = KnowledgeBaseService()
        
        # normalize_kb_config
        config = {"type": "chroma"}
        normalized = service.normalize_kb_config(config)
        assert normalized == config
        
        # create_knowledge_base_factory
        mock_factory_cls = MagicMock()
        service.create_knowledge_base_factory(mock_factory_cls, config)
        mock_factory_cls.assert_called_once_with(
            knowledge_base_config=config,
            logger=service.logger,
            project_root=None,
            llm=None,
            document_loader=None,
            vector_store=None
        )

        # get_knowledge_base
        mock_factory_cls.return_value.get_knowledge_base.return_value = "my_kb"
        assert service.get_knowledge_base("sourceA") == "my_kb"
        mock_factory_cls.return_value.get_knowledge_base.assert_called_once_with("sourceA")

# ============================================================================
# ToolService Tests
# ============================================================================

class TestToolService:
    @patch("oai_agent_core.core.base_tool_registry.BaseToolRegistry")
    def test_create_registry(self, mock_registry_cls):
        service = ToolService()
        
        # Base framework creation is a noop (returns None)
        assert service.create_registry("base") is None
        
        # Unsupported framework raises ToolLoadingError
        with pytest.raises(Exception):
            service.create_registry("unsupported")

    def test_load_and_get_tools(self):
        service = ToolService()
        registry = MagicMock()
        registry.get_tools.return_value = ["toolA"]
        registry.mcp_configs = {"mcp1": {}}
        registry.get_tool.return_value = "toolA"
        
        # get_tool
        assert service.get_tool(registry, "toolA") == "toolA"
        
        # load_tools
        assert service.load_tools(registry, {"my_tool": {}}) == 1
        registry.load_tools_from_config.assert_called_once_with({"my_tool": {}})

        # load_mcp_tools
        assert service.load_mcp_tools(registry, {"mcp1": {}}) == 1
        registry.load_mcp_config.assert_called_once_with({"mcp1": {}})

# ============================================================================
# SkillService Tests
# ============================================================================

class TestSkillService:
    @patch("oai_agent_core.components.skills.skill_registry.SkillRegistry")
    def test_create_skill_registry(self, mock_registry_cls):
        service = SkillService(project_root="/root")
        config = {"skills": {}}
        service.create_skill_registry(config)
        mock_registry_cls.assert_called_once_with(
            logger=service.logger,
            project_root="/root",
            registry_url=None,
            auth_token=None,
            skills_cache_dir=None
        )

    @pytest.mark.asyncio
    async def test_discover_and_pull(self):
        service = SkillService()
        registry = MagicMock()
        registry.skills = {"skillA": MagicMock()}
        registry.pull_skill = AsyncMock(return_value=True)
        
        # discover_skills
        assert service.discover_skills(registry, "/path") == 1
        registry.discover_skills.assert_called_once_with(skills_dir="/path")

        # pull_required_skills
        assert await service.pull_required_skills(registry, ["skillA"]) == 1
        registry.pull_skill.assert_called_once_with("skillA")

# ============================================================================
# ModelService Tests
# ============================================================================

class TestModelService:
    def test_model_service_flows(self):
        service = ModelService()
        
        # validate_model_config
        assert service.validate_model_config({"name": "gpt-4", "provider": "openai"}) is True
        assert service.validate_model_config({"name": "gpt-4"}) is False
        
        # set_model_manager
        manager = MagicMock()
        service.set_model_manager(manager)
        assert service._model_manager == manager

        # create_model
        model_config = {"name": "gpt-4", "provider": "openai"}
        manager.create_model.return_value = "my_model"
        assert service.create_model(model_config) == "my_model"
        manager.create_model.assert_called_once_with(model_config=model_config)

# ============================================================================
# ObservabilityService Tests
# ============================================================================

class TestObservabilityService:
    @patch("oai_agent_core.components.observability.langfuse_observability_manager.LangfuseObservabilityManager")
    def test_observability_flows(self, mock_langfuse_cls):
        service = ObservabilityService(agent_name="my_agent", framework="crewai")
        
        mock_mgr = MagicMock()
        mock_langfuse_cls.return_value = mock_mgr
        
        # create_langfuse_manager
        assert service.create_langfuse_manager() == mock_mgr
        mock_langfuse_cls.assert_called_once_with(
            agent_name="my_agent",
            logger=service.logger,
            framework="crewai"
        )

        # track_operation
        mock_mgr.trace_operation.return_value = "span"
        assert service.track_operation("op1") == "span"
        mock_mgr.trace_operation.assert_called_once_with("op1")

        # track_error
        err = ValueError("bad value")
        service.track_error("op1", err)
        mock_mgr.track_error.assert_called_once_with("op1", err)

        # track_tokens
        service.track_tokens(10, 20)
        mock_mgr.track_tokens.assert_called_once_with(input_tokens=10, output_tokens=20)

# ============================================================================
# MemoryService Tests
# ============================================================================

class TestMemoryService:
    def test_memory_flows(self):
        service = MemoryService(project_root="/root")
        assert service.project_root == "/root"
        
        # None config
        assert service.create_memory_store(None) is None
        
        # Valid config creation (pass mock vector_store to skip Bedrock embeddings setup)
        config = {"type": "simple"}
        store = service.create_memory_store(config, llm=MagicMock())
        assert store is not None
        assert store.project_root == "/root"
        assert store.max_recent_turns == 5
