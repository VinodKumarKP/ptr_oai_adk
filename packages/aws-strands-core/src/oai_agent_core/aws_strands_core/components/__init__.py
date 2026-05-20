"""Components module for AWS Strands agent tools, knowledge base, and configuration."""

from oai_agent_core.aws_strands_core.components.registry.tool_registry import AWSStrandsToolRegistry
from oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
from oai_agent_core.aws_strands_core.components.configuration.model_config import StrandsModelConfigurationManager

__all__ = [
    "AWSStrandsToolRegistry",
    "KnowledgeBaseFactory",
    "StrandsModelConfigurationManager",
]