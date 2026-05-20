"""OpenAI Core — agent implementation built on top of the openai-agents library."""

from oai_agent_core.openai_core.agents.openai_agent import OpenAIAgent
from oai_agent_core.openai_core.builders.agent_builder import AgentBuilder
from oai_agent_core.openai_core.components.tools.tool_registry import OpenAIToolRegistry
from oai_agent_core.openai_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
from oai_agent_core.openai_core.components.configuration.model_config import OpenAIModelConfigurationManager
from oai_agent_core.openai_core.processing.result_extractor import ResultExtractor

__all__ = [
    "OpenAIAgent",
    "AgentBuilder",
    "OpenAIToolRegistry",
    "KnowledgeBaseFactory",
    "OpenAIModelConfigurationManager",
    "ResultExtractor",
]
