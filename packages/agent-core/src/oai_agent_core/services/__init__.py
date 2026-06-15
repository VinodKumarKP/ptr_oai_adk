"""Service Layer for Agent-Core Architecture

This package provides focused service classes for major agent operations,
enabling dependency injection and easier testing. Each service lives in its own
module and is re-exported here so ``from oai_agent_core.services import X`` keeps
working.

## Services

- **ConfigResolverService** (``config_resolver_service``): Configuration loading
  and resolution
- **KnowledgeBaseService** (``knowledge_base_service``): Knowledge base
  initialization and access
- **ToolService** (``tool_service``): Tool registry and loading
- **SkillService** (``skill_service``): Skill discovery and management
- **ModelService** (``model_service``): LLM provider initialization
- **ObservabilityService** (``observability_service``): Tracing and monitoring
- **ConfigValidator** (``config_validator``): Configuration validation/schema
- **MemoryService** (``memory_service``): Memory store initialization
- **GuardrailsService** (``guardrails_service``): Guardrails manager initialization
"""

from oai_agent_core.services.config_resolver_service import ConfigResolverService
from oai_agent_core.services.knowledge_base_service import KnowledgeBaseService
from oai_agent_core.services.tool_service import ToolService
from oai_agent_core.services.skill_service import SkillService
from oai_agent_core.services.model_service import ModelService
from oai_agent_core.services.observability_service import ObservabilityService
from oai_agent_core.services.config_validator import ConfigValidator
from oai_agent_core.services.memory_service import MemoryService
from oai_agent_core.services.guardrails_service import GuardrailsService

__all__ = [
    "ConfigResolverService",
    "KnowledgeBaseService",
    "ToolService",
    "SkillService",
    "ModelService",
    "ObservabilityService",
    "ConfigValidator",
    "MemoryService",
    "GuardrailsService",
]
