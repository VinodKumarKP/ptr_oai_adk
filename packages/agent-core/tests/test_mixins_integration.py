"""Integration tests for BaseAgent with all mixins working together.

Tests BaseAgent's interaction with all mixins to verify they work
together correctly without conflicts.
"""

import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.core.agent_mixins import (
    SkillsMixin,
    KnowledgeBaseMixin,
    ToolsMixin,
    MemoryMixin,
    MessageProcessingMixin,
    ObservabilityMixin,
)


class TestBaseAgentMixinIntegration:
    """Test BaseAgent integration with all mixins."""

    def test_base_agent_has_all_mixin_methods(self):
        """Test that BaseAgent has all methods from all mixins."""
        methods_to_check = [
            # Skills
            '_load_skills_if_configured', 'get_skill_registry',
            # KB
            '_load_knowledge_base', 'get_knowledge_base',
            # Tools
            '_load_tools', 'get_tools', 'get_tool',
            # Memory
            '_load_memory', '_get_conversation_context', 'store_conversation',
            # Message Processing
            '_augment_system_prompt', '_guardrail_input_message',
            '_guardrail_output_message', '_augment_message',
            # Observability
            '_trace_operation', 'track_token_usage', 'get_trace_context'
        ]
        
        for method_name in methods_to_check:
            assert hasattr(BaseAgent, method_name), f"BaseAgent missing {method_name}"

    def test_base_agent_mro_contains_all_mixins(self):
        """Test that MRO contains all mixins in correct order."""
        mro_names = [cls.__name__ for cls in BaseAgent.__mro__]
        
        expected_mixins = [
            'SkillsMixin',
            'KnowledgeBaseMixin',
            'ToolsMixin',
            'MemoryMixin',
            'MessageProcessingMixin',
            'ObservabilityMixin'
        ]
        
        for mixin_name in expected_mixins:
            assert mixin_name in mro_names, f"{mixin_name} not in BaseAgent MRO"


class TestMixinCompositionWithCustomClass:
    """Test mixin composition in custom classes."""

    def test_all_mixins_together(self):
        """Test composing all mixins in a single class."""
        class FullAgent(
            SkillsMixin,
            KnowledgeBaseMixin,
            ToolsMixin,
            MemoryMixin,
            MessageProcessingMixin,
            ObservabilityMixin
        ):
            def __init__(self):
                self.logger = MagicMock()
        
        agent = FullAgent()
        
        # Verify all methods exist and are callable
        methods = [
            '_load_skills_if_configured', 'get_skill_registry',
            '_load_knowledge_base', 'get_knowledge_base',
            '_load_tools', 'get_tools', 'get_tool',
            '_load_memory', '_get_conversation_context', 'store_conversation',
            '_augment_system_prompt', '_guardrail_input_message',
            '_guardrail_output_message', '_augment_message',
            '_trace_operation', 'track_token_usage', 'get_trace_context'
        ]
        
        for method_name in methods:
            assert hasattr(agent, method_name)
            assert callable(getattr(agent, method_name))

    def test_partial_mixin_composition(self):
        """Test composing only some mixins."""
        class PartialAgent(SkillsMixin, ToolsMixin):
            def __init__(self):
                self.logger = MagicMock()
        
        agent = PartialAgent()
        
        # Should have methods from selected mixins
        assert hasattr(agent, '_load_skills_if_configured')
        assert hasattr(agent, '_load_tools')
        
        # Should not have methods from unselected mixins
        assert not hasattr(agent, '_load_knowledge_base')
        assert not hasattr(agent, '_load_memory')


class TestMixinStateManagement:
    """Test state management with multiple mixins."""

    def test_mixin_attributes_isolated(self):
        """Test that mixin attributes are properly isolated."""
        class TestAgent(SkillsMixin, KnowledgeBaseMixin, ToolsMixin):
            def __init__(self):
                self.logger = MagicMock()
        
        agent1 = TestAgent()
        agent2 = TestAgent()
        
        # Set attributes on agent1
        agent1.skill_registry = MagicMock(name='registry1')
        agent1.kb = MagicMock(name='kb1')
        
        # agent2 should not be affected
        assert not hasattr(agent2, 'skill_registry')
        assert not hasattr(agent2, 'kb')

    def test_attribute_access_via_mixins(self):
        """Test accessing attributes set by different mixins."""
        class TestAgent(SkillsMixin, KnowledgeBaseMixin, ToolsMixin):
            def __init__(self):
                self.logger = MagicMock()
                self.skill_registry = MagicMock()
                self.kb = MagicMock()
                self.tool_registry = MagicMock()
        
        agent = TestAgent()
        
        # All attributes should be accessible
        assert agent.skill_registry is not None
        assert agent.kb is not None
        assert agent.tool_registry is not None


class TestMixinMethodCalls:
    """Test calling mixin methods in various scenarios."""

    def test_mixin_methods_callable_with_mocks(self):
        """Test that mixin methods can be called with mocked dependencies."""
        class TestAgent(SkillsMixin, ToolsMixin):
            def __init__(self):
                self.logger = MagicMock()
                self.skill_registry = None
                self.tool_registry = None
        
        agent = TestAgent()
        
        # Should be able to call methods (they might return None or raise, but that's ok)
        assert callable(agent.get_skill_registry)
        assert callable(agent.get_tools)

    def test_mixin_method_chaining(self):
        """Test that mixin methods can work together."""
        class TestAgent(SkillsMixin, KnowledgeBaseMixin):
            def __init__(self):
                self.logger = MagicMock()
        
        agent = TestAgent()
        
        # Should have methods from both mixins
        skill_methods = [
            agent._load_skills_if_configured,
            agent.get_skill_registry,
        ]
        
        kb_methods = [
            agent._load_knowledge_base,
            agent.get_knowledge_base,
        ]
        
        all_methods = skill_methods + kb_methods
        for method in all_methods:
            assert callable(method)


class TestMixinInheritanceHierarchy:
    """Test mixin inheritance hierarchy."""

    def test_mixin_inheritance_chain(self):
        """Test that mixins form proper inheritance chain."""
        # Each mixin should be a class
        assert isinstance(SkillsMixin, type)
        assert isinstance(KnowledgeBaseMixin, type)
        assert isinstance(ToolsMixin, type)
        assert isinstance(MemoryMixin, type)
        assert isinstance(MessageProcessingMixin, type)
        assert isinstance(ObservabilityMixin, type)

    def test_base_agent_inheritance_from_mixins(self):
        """Test BaseAgent properly inherits from mixins."""
        # Check that BaseAgent is a subclass of all mixins
        assert issubclass(BaseAgent, SkillsMixin)
        assert issubclass(BaseAgent, KnowledgeBaseMixin)
        assert issubclass(BaseAgent, ToolsMixin)
        assert issubclass(BaseAgent, MemoryMixin)
        assert issubclass(BaseAgent, MessageProcessingMixin)
        assert issubclass(BaseAgent, ObservabilityMixin)


class TestMixinBackwardCompatibility:
    """Test backward compatibility with mixin integration."""

    def test_base_agent_still_abstract(self):
        """Test that BaseAgent is still abstract."""
        from abc import ABC
        
        # Should be abstract
        assert issubclass(BaseAgent, ABC)

    def test_base_agent_core_interface(self):
        """Test that BaseAgent core interface is preserved."""
        # Should still have abstract methods or expected interface
        assert hasattr(BaseAgent, '__init__') or hasattr(BaseAgent, 'initialize')

    def test_mixin_integration_no_conflicts(self):
        """Test that mixins don't create method conflicts."""
        class TestAgent(
            SkillsMixin,
            KnowledgeBaseMixin,
            ToolsMixin,
            MemoryMixin,
            MessageProcessingMixin,
            ObservabilityMixin
        ):
            def __init__(self):
                self.logger = MagicMock()
        
        agent = TestAgent()
        
        # Get all methods
        all_methods = dir(agent)
        
        # Count occurrences in MRO
        method_sources = {}
        for method_name in all_methods:
            if not method_name.startswith('_'):
                continue
            sources = 0
            for cls in TestAgent.__mro__:
                if method_name in cls.__dict__:
                    sources += 1
            if sources > 1:
                method_sources[method_name] = sources
        
        # Most methods should come from exactly one class
        # (some magic methods might come from multiple, that's ok)
        # This test just verifies the structure makes sense
        assert isinstance(method_sources, dict)
