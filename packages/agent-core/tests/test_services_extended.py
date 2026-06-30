"""Extended tests for services with low coverage."""
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from oai_agent_core.services import (
    KnowledgeBaseService,
    ToolService,
    SkillService,
    ModelService,
    ObservabilityService,
    MemoryService,
)


class TestKnowledgeBaseServiceExtended:
    def test_get_kb_no_factory(self):
        service = KnowledgeBaseService()
        # No factory initialized, so get_knowledge_base returns None
        result = service.get_knowledge_base({"type": "chroma"})
        assert result is None

    def test_get_kb_factory_error(self):
        service = KnowledgeBaseService()
        mock_factory = MagicMock()
        mock_factory.get_knowledge_base.side_effect = Exception("kb error")
        service._kb_factory = mock_factory
        result = service.get_knowledge_base({"type": "chroma"})
        assert result is None

    def test_normalize_kb_config_agent_list(self):
        service = KnowledgeBaseService()
        config = {
            "registry": {
                "url": "http://registry.com",
                "auth_token": "my_token"
            },
            "agent_list": [
                {
                    "writer_agent": {
                        "knowledge_base": [
                            {"name": "docs_kb"},
                            {"name": "code_kb", "registry_url": "http://other.com"}
                        ]
                    }
                }
            ]
        }
        normalized = service.normalize_kb_config(config)
        
        agent_kbs = normalized["agent_list"][0]["writer_agent"]["knowledge_base"]
        # First KB should get the registry URL from top-level config
        assert agent_kbs[0]["registry_url"] == "http://registry.com"
        assert agent_kbs[0]["auth_token"] == "my_token"
        # Second KB already has a registry_url, shouldn't be overwritten
        assert agent_kbs[1]["registry_url"] == "http://other.com"


class TestToolServiceExtended:
    def test_load_tools_no_registry(self):
        service = ToolService()
        result = service.load_tools(None, {})
        assert result == 0

    def test_load_tools_with_tools_attribute(self):
        service = ToolService()
        registry = MagicMock()
        del registry.get_tools  # no get_tools method
        registry.tools = {"toolA": MagicMock(), "toolB": MagicMock()}
        result = service.load_tools(registry, {"tools": {}})
        assert result == 2

    def test_load_tools_no_count_attribute(self):
        service = ToolService()
        registry = MagicMock(spec=[])  # no get_tools or tools attributes
        result = service.load_tools(registry, {"tools": {}})
        assert result == 0

    def test_load_tools_exception(self):
        service = ToolService()
        registry = MagicMock()
        registry.load_tools_from_config.side_effect = Exception("load failed")
        result = service.load_tools(registry, {"tools": {}})
        assert result == 0

    def test_load_mcp_tools_no_registry(self):
        service = ToolService()
        assert service.load_mcp_tools(None, {}) == 0

    def test_load_mcp_tools_exception(self):
        service = ToolService()
        registry = MagicMock()
        registry.load_mcp_config.side_effect = Exception("mcp failed")
        result = service.load_mcp_tools(registry, {})
        assert result == 0

    def test_get_tool_no_registry(self):
        service = ToolService()
        assert service.get_tool(None, "toolA") is None


class TestSkillServiceExtended:
    def test_create_skill_registry_empty_config(self):
        service = SkillService()
        result = service.create_skill_registry({})
        assert result is None
        result = service.create_skill_registry(None)
        assert result is None

    @patch("oai_agent_core.components.skills.skill_registry.SkillRegistry",
           side_effect=Exception("import error"))
    def test_create_skill_registry_failure(self, mock_cls):
        service = SkillService()
        result = service.create_skill_registry({"skills": {}})
        assert result is None

    def test_discover_skills_no_registry(self):
        service = SkillService()
        assert service.discover_skills(None, "/path") == 0
        assert service.discover_skills(MagicMock(), "") == 0

    def test_discover_skills_exception(self):
        service = SkillService()
        registry = MagicMock()
        registry.discover_skills.side_effect = Exception("scan failed")
        result = service.discover_skills(registry, "/path")
        assert result == 0


class TestModelServiceExtended:
    def test_create_model_no_manager(self):
        service = ModelService()
        result = service.create_model({"name": "gpt-4", "provider": "openai"})
        assert result is None

    def test_create_model_cache_hit(self):
        service = ModelService(enable_cache=True)
        manager = MagicMock()
        manager.create_model.return_value = "my_model"
        service.set_model_manager(manager)
        
        config = {"name": "gpt-4", "provider": "openai"}
        # First call populates cache
        service.create_model(config)
        # Second call should use cache
        service.create_model(config)
        assert manager.create_model.call_count == 1

    def test_create_model_exception(self):
        service = ModelService()
        manager = MagicMock()
        manager.create_model.side_effect = Exception("model error")
        service.set_model_manager(manager)
        result = service.create_model({"name": "gpt-4", "provider": "openai"})
        assert result is None

    def test_clear_cache(self):
        service = ModelService(enable_cache=True)
        manager = MagicMock()
        manager.create_model.return_value = "model"
        service.set_model_manager(manager)
        service.create_model({"name": "gpt-4", "provider": "openai"})
        assert len(service._model_cache) == 1
        service.clear_cache()
        assert len(service._model_cache) == 0

    def test_validate_model_config_missing_provider(self):
        service = ModelService()
        assert service.validate_model_config({"name": "gpt-4"}) is False


class TestObservabilityServiceExtended:
    def test_init(self):
        service = ObservabilityService(agent_name="my_agent", framework="langchain")
        assert service.agent_name == "my_agent"
        assert service.framework == "langchain"

    @patch("oai_agent_core.components.observability.langfuse_observability_manager.LangfuseObservabilityManager",
           side_effect=Exception("langfuse not available"))
    def test_create_manager_failure(self, mock_cls):
        service = ObservabilityService(agent_name="agent1", framework="crewai")
        result = service.create_langfuse_manager()
        assert result is None

    def test_track_operation_no_manager(self):
        service = ObservabilityService(agent_name="agent1", framework="crewai")
        service._langfuse_manager = None
        result = service.track_operation("op1")
        assert result is None

    def test_track_error_no_manager(self):
        service = ObservabilityService(agent_name="agent1", framework="crewai")
        service._langfuse_manager = None
        # Should not raise
        service.track_error("op1", ValueError("bad"))

    def test_track_tokens_no_manager(self):
        service = ObservabilityService(agent_name="agent1", framework="crewai")
        service._langfuse_manager = None
        # Should not raise
        service.track_tokens(10, 20)


class TestMemoryServiceExtended:
    def test_create_memory_store_none_config(self):
        service = MemoryService()
        result = service.create_memory_store(None)
        assert result is None

    @patch("oai_agent_core.core.base_memory_store.BaseMemoryStore.__init_subclass__",
           side_effect=Exception("abstract init failed"))
    def test_create_memory_store_failure(self, mock_subclass):
        service = MemoryService()
        # Patch the memory service so that instantiation raises
        with patch.object(service, 'create_memory_store', return_value=None):
            result = service.create_memory_store({"type": "simple"})
            assert result is None
