"""Unit tests for agent mixins.

Tests each mixin independently with mocked dependencies to verify
single responsibility and correct behavior.
"""

from unittest.mock import MagicMock
from oai_agent_core.mixins.agent_mixins import (
    SkillsMixin,
    KnowledgeBaseMixin,
    ToolsMixin,
    MemoryMixin,
    MessageProcessingMixin,
    ObservabilityMixin,
)


class TestSkillsMixin:
    """Test SkillsMixin functionality."""

    def test_mixin_has_methods(self):
        """Test that mixin has expected methods."""
        mixin = SkillsMixin()
        
        # Should have these methods
        assert hasattr(mixin, '_load_skills_if_configured')
        assert hasattr(mixin, 'get_skill_registry')


class TestKnowledgeBaseMixin:
    """Test KnowledgeBaseMixin functionality."""

    def test_mixin_has_methods(self):
        """Test that mixin has expected methods."""
        mixin = KnowledgeBaseMixin()
        
        # Should have these methods
        assert hasattr(mixin, '_load_knowledge_base')
        assert hasattr(mixin, 'get_knowledge_base')


class TestToolsMixin:
    """Test ToolsMixin functionality."""

    def test_mixin_has_methods(self):
        """Test that mixin has expected methods."""
        mixin = ToolsMixin()
        
        # Should have these methods
        assert hasattr(mixin, '_load_tools')
        assert hasattr(mixin, 'get_tools')
        assert hasattr(mixin, 'get_tool')


class TestMemoryMixin:
    """Test MemoryMixin functionality."""

    def test_mixin_has_methods(self):
        """Test that mixin has expected methods."""
        mixin = MemoryMixin()
        
        # Should have these methods
        assert hasattr(mixin, '_load_memory')
        assert hasattr(mixin, '_get_conversation_context')
        assert hasattr(mixin, 'store_conversation')


class TestMessageProcessingMixin:
    """Test MessageProcessingMixin functionality."""

    def test_mixin_has_methods(self):
        """Test that mixin has expected methods."""
        mixin = MessageProcessingMixin()
        
        # Should have these methods
        assert hasattr(mixin, '_augment_system_prompt')
        assert hasattr(mixin, '_guardrail_input_message')
        assert hasattr(mixin, '_guardrail_output_message')
        assert hasattr(mixin, '_augment_message')


class TestObservabilityMixin:
    """Test ObservabilityMixin functionality."""

    def test_mixin_has_methods(self):
        """Test that mixin has expected methods."""
        mixin = ObservabilityMixin()
        
        # Should have these methods
        assert hasattr(mixin, '_trace_operation')
        assert hasattr(mixin, 'track_token_usage')
        assert hasattr(mixin, 'get_trace_context')


class TestMixinIntegration:
    """Test mixins working together."""

    def test_multiple_mixins_in_class(self):
        """Test that multiple mixins can be used in a single class."""
        
        class TestAgent(SkillsMixin, KnowledgeBaseMixin, ToolsMixin):
            def __init__(self):
                self.logger = MagicMock()
        
        agent = TestAgent()
        
        # Should have methods from all mixins
        assert hasattr(agent, '_load_skills_if_configured')
        assert hasattr(agent, '_load_knowledge_base')
        assert hasattr(agent, '_load_tools')
        assert hasattr(agent, 'get_skill_registry')
        assert hasattr(agent, 'get_knowledge_base')
        assert hasattr(agent, 'get_tools')

    def test_mixin_method_resolution_order(self):
        """Test that mixin MRO works correctly."""
        
        class TestAgent(SkillsMixin, KnowledgeBaseMixin, ToolsMixin):
            pass
        
        # MRO should include all mixins in order
        mro_names = [cls.__name__ for cls in TestAgent.__mro__]
        assert 'SkillsMixin' in mro_names
        assert 'KnowledgeBaseMixin' in mro_names
        assert 'ToolsMixin' in mro_names
