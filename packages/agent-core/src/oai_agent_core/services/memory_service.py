"""Memory store initialization and management service."""

import logging
from typing import Dict, Any, Optional


class MemoryService:
    """Service for memory store initialization and management.

    Handles:
    - Creating memory stores from configuration
    - Validating memory configuration
    - Supporting multiple memory backends (Redis, MongoDB, DiskCache, etc.)

    **Phase 3.6**: Wraps memory store initialization with service layer.
    """

    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the memory service.

        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)

    def create_memory_store(self, memory_config: Dict[str, Any],
                           llm: Optional[Any] = None) -> Optional[Any]:
        """Create a memory store from configuration.

        **Phase 3.6**: Initializes memory store based on config.

        Args:
            memory_config: Memory configuration
            llm: Optional LLM instance

        Returns:
            Memory store instance, or None if creation failed
        """
        if not memory_config:
            return None

        try:
            from oai_agent_core.core.base_memory_store import BaseMemoryStore

            # Create concrete implementation
            class ConfigurableMemoryStore(BaseMemoryStore):
                pass

            store = ConfigurableMemoryStore(
                memory_config=memory_config,
                logger=self.logger,
                project_root=self.project_root,
                llm=llm
            )

            self.logger.debug("Created memory store from configuration")
            return store
        except Exception as e:
            self.logger.error(f"Failed to create memory store: {e}")
            return None

    def validate_memory_config(self, memory_config: Dict[str, Any]) -> bool:
        """Validate memory configuration.

        **Phase 3.6**: Checks for required memory settings.

        Args:
            memory_config: Memory configuration

        Returns:
            True if valid, False otherwise
        """
        if not memory_config:
            return True  # Empty config is ok

        memory_type = memory_config.get('type', 'simple')

        # Supported types
        supported_types = ['redis', 'mongodb', 'diskcache', 'simple', 'pinecone']
        if memory_type not in supported_types:
            self.logger.warning(f"Unknown memory type: {memory_type}")
            return False

        self.logger.debug("Memory configuration validation passed")
        return True
