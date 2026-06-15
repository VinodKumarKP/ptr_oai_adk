"""Knowledge base initialization and management service."""

import logging
from typing import Dict, Any, Optional


class KnowledgeBaseService:
    """Service for knowledge base initialization and management.

    Handles:
    - Creating knowledge base factories
    - Initializing knowledge bases from config
    - Managing KB credentials and registry access
    - Propagating top-level KB settings to individual KBs
    """

    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the KB service.

        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self._kb_factory = None

    def create_knowledge_base_factory(self,
                                      kb_factory_class: Any,
                                      knowledge_base_config: Any,
                                      llm: Optional[Any] = None,
                                      document_loader: Optional[Any] = None,
                                      vector_store: Optional[Any] = None) -> Any:
        """Create a knowledge base factory instance.

        Constructs a concrete, framework-specific knowledge base factory from the
        resolved KB configuration. The returned object is the factory instance
        itself (exposing ``search_custom_knowledge_base``, ``create_tool`` etc.),
        not a wrapper or callable.

        Args:
            kb_factory_class: Concrete KnowledgeBaseFactory subclass to instantiate
                (supplied by the framework-specific agent).
            knowledge_base_config: Resolved KB sources (registry credentials already
                merged in).
            llm: Optional LLM instance for embeddings.
            document_loader: Optional custom document loader.
            vector_store: Optional custom vector store.

        Returns:
            KnowledgeBaseFactory instance.

        Raises:
            Exception: Propagates construction errors (e.g. ImportError for missing
                optional dependencies) so the caller can decide degradation policy.
        """
        factory = kb_factory_class(
            knowledge_base_config=knowledge_base_config,
            logger=self.logger,
            project_root=self.project_root,
            llm=llm,
            document_loader=document_loader,
            vector_store=vector_store,
        )
        self._kb_factory = factory
        self.logger.debug("Created knowledge base factory")
        return factory

    def get_knowledge_base(self, kb_config: Dict[str, Any]) -> Any:
        """Get a knowledge base instance for the given configuration.

        Args:
            kb_config: Knowledge base configuration

        Returns:
            Knowledge base instance, or None if factory not available
        """
        if not self._kb_factory:
            self.logger.warning("KB factory not initialized")
            return None

        try:
            kb = self._kb_factory.get_knowledge_base(kb_config)
            self.logger.debug("Created knowledge base instance")
            return kb
        except Exception as e:
            self.logger.error(f"Failed to create KB instance: {e}")
            return None

    def normalize_kb_config(self, kb_config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize knowledge base configuration.

        Handles legacy formats and propagates top-level settings to individual KBs.

        Args:
            kb_config: Raw KB configuration

        Returns:
            Normalized KB configuration
        """
        # If agent_list is present, propagate top-level registry to all agents
        if isinstance(kb_config, dict) and 'agent_list' in kb_config:
            registry_url = kb_config.get('registry', {}).get('url')
            auth_token = kb_config.get('registry', {}).get('auth_token')

            for agent_entry in kb_config.get('agent_list', []):
                if isinstance(agent_entry, dict):
                    for agent_name, agent_config in agent_entry.items():
                        kb_entries = agent_config.get('knowledge_base', [])
                        if isinstance(kb_entries, list):
                            for entry in kb_entries:
                                if registry_url and 'registry_url' not in entry:
                                    entry['registry_url'] = registry_url
                                if auth_token and 'auth_token' not in entry:
                                    entry['auth_token'] = auth_token

        return kb_config
