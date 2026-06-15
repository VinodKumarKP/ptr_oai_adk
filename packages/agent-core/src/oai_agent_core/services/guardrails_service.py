"""Guardrails manager initialization and management service."""

import logging
from typing import Dict, Any, Optional


class GuardrailsService:
    """Service for guardrails manager initialization and management.

    Handles:
    - Creating guardrails managers from configuration
    - Validating guardrails configuration
    - Managing guardrails prompts and validation

    **Phase 3.6**: Wraps guardrails manager initialization with service layer.
    """

    def __init__(self, project_root: Optional[str] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize the guardrails service.

        Args:
            project_root: Project root directory
            logger: Optional logger instance
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)

    def create_guardrails_manager(self, guardrails_config: Dict[str, Any]) -> Optional[Any]:
        """Create a guardrails manager from configuration.

        **Phase 3.6**: Initializes guardrails manager based on config.

        Args:
            guardrails_config: Guardrails configuration

        Returns:
            GuardrailsManager instance, or None if creation failed
        """
        if not guardrails_config:
            return None

        try:
            from oai_agent_core.components.guardrails.guardrails_manager import GuardrailManager

            manager = GuardrailManager(
                self.project_root,
                guardrails_config,
                logger=self.logger
            )

            self.logger.debug("Created guardrails manager from configuration")
            return manager
        except Exception as e:
            self.logger.error(f"Failed to create guardrails manager: {e}")
            return None

    def validate_guardrails_config(self, guardrails_config: Dict[str, Any]) -> bool:
        """Validate guardrails configuration.

        **Phase 3.6**: Checks for required guardrails settings.

        Args:
            guardrails_config: Guardrails configuration

        Returns:
            True if valid, False otherwise
        """
        if not guardrails_config:
            return True  # Empty config is ok

        # Check for guard definition
        if not guardrails_config.get('definition_file') and not guardrails_config.get('guard'):
            self.logger.warning("Guardrails configuration missing 'definition_file' or 'guard'")
            return False

        self.logger.debug("Guardrails configuration validation passed")
        return True

    def get_guardrails_prompt(self, manager: Optional[Any]) -> str:
        """Get guardrails prompt from manager.

        **Phase 3.6**: Retrieves guardrails instructions for system prompt.

        Args:
            manager: GuardrailsManager instance

        Returns:
            Guardrails prompt string
        """
        if not manager:
            return ""

        try:
            return manager.get_guardrails_prompt()
        except Exception as e:
            self.logger.warning(f"Failed to get guardrails prompt: {e}")
            return ""
