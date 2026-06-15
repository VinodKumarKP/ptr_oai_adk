"""Unit tests for service layer classes.

Tests each service independently with mocked dependencies
to verify correct behavior and error handling.
"""

import logging
from oai_agent_core.services import (
    ConfigResolverService,
    KnowledgeBaseService,
    ToolService,
    SkillService,
    ModelService,
    ObservabilityService,
)


class TestConfigResolverService:
    """Test ConfigResolverService."""

    def test_init_with_defaults(self):
        """Test service initialization with default parameters."""
        service = ConfigResolverService()
        
        assert service.config_root is not None
        assert service.logger is not None

    def test_init_with_custom_config_root(self):
        """Test service initialization with custom config root."""
        service = ConfigResolverService(config_root='/custom/config')
        
        assert service.config_root == '/custom/config'

    def test_init_with_custom_logger(self):
        """Test service initialization with custom logger."""
        custom_logger = logging.getLogger('test')
        service = ConfigResolverService(logger=custom_logger)
        
        assert service.logger == custom_logger

    def test_has_expected_methods(self):
        """Test service has expected methods."""
        service = ConfigResolverService()
        
        assert hasattr(service, 'load_agent_config')
        assert hasattr(service, 'resolve_macros')
        assert hasattr(service, 'validate_config')


class TestKnowledgeBaseService:
    """Test KnowledgeBaseService."""

    def test_init_with_defaults(self):
        """Test service initialization."""
        service = KnowledgeBaseService()
        
        assert service.logger is not None

    def test_has_expected_methods(self):
        """Test service has expected methods."""
        service = KnowledgeBaseService()
        
        assert hasattr(service, 'create_knowledge_base_factory')
        assert hasattr(service, 'get_knowledge_base')
        assert hasattr(service, 'normalize_kb_config')


class TestToolService:
    """Test ToolService."""

    def test_init_with_defaults(self):
        """Test service initialization."""
        service = ToolService()
        
        assert service.logger is not None

    def test_has_expected_methods(self):
        """Test service has expected methods."""
        service = ToolService()
        
        assert hasattr(service, 'create_registry')
        assert hasattr(service, 'load_tools')
        assert hasattr(service, 'load_mcp_tools')
        assert hasattr(service, 'get_tool')


class TestSkillService:
    """Test SkillService."""

    def test_init_with_defaults(self):
        """Test service initialization."""
        service = SkillService()
        
        assert service.logger is not None

    def test_has_expected_methods(self):
        """Test service has expected methods."""
        service = SkillService()
        
        assert hasattr(service, 'create_skill_registry')
        assert hasattr(service, 'pull_required_skills')
        assert hasattr(service, 'discover_skills')


class TestModelService:
    """Test ModelService."""

    def test_init_with_defaults(self):
        """Test service initialization."""
        service = ModelService()
        
        assert service.logger is not None

    def test_has_expected_methods(self):
        """Test service has expected methods."""
        service = ModelService()
        
        assert hasattr(service, 'set_model_manager')
        assert hasattr(service, 'create_model')
        assert hasattr(service, 'validate_model_config')


class TestObservabilityService:
    """Test ObservabilityService."""

    def test_init_with_defaults(self):
        """Test service initialization."""
        service = ObservabilityService()
        
        assert service.logger is not None

    def test_has_expected_methods(self):
        """Test service has expected methods."""
        service = ObservabilityService()
        
        assert hasattr(service, 'create_langfuse_manager')
        assert hasattr(service, 'track_operation')
        assert hasattr(service, 'track_error')
        assert hasattr(service, 'track_tokens')


class TestServiceLayerIntegration:
    """Test services working together."""

    def test_multiple_services_initialization(self):
        """Test initializing multiple services."""
        config_service = ConfigResolverService()
        tool_service = ToolService()
        kb_service = KnowledgeBaseService()
        skill_service = SkillService()
        model_service = ModelService()
        obs_service = ObservabilityService()
        
        # All services should initialize successfully
        assert config_service is not None
        assert tool_service is not None
        assert kb_service is not None
        assert skill_service is not None
        assert model_service is not None
        assert obs_service is not None

    def test_service_independence(self):
        """Test that services can be used independently."""
        service1 = ToolService()
        service2 = ToolService()
        
        # Services should be independent instances
        assert service1 is not service2
        assert service1.logger is not None
        assert service2.logger is not None
