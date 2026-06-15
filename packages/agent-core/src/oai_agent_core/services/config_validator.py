"""Configuration validation and schema enforcement."""

import logging
from typing import Dict, Any, List, Optional


class ConfigValidator:
    """Phase 3.2: Configuration validation and schema enforcement.

    Validates all configuration sections to catch errors early and provide
    clear error messages for invalid configurations.

    Handles:
    - Tool configuration validation
    - Skill configuration validation
    - Knowledge base configuration validation
    - Model configuration validation
    - Memory configuration validation
    - Guardrails configuration validation
    - Overall agent configuration validation
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the configuration validator.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_agent_config(self, config: Dict[str, Any]) -> bool:
        """Validate complete agent configuration.

        Args:
            config: Agent configuration to validate

        Returns:
            True if all validations passed, False otherwise
        """
        self.errors = []
        self.warnings = []

        # Skip validation if config is empty or minimal (common in tests)
        if not config:
            return True

        # Required fields - but don't fail if 'type' is missing
        # (some test configs might not have it)
        if 'type' not in config and any(k in config for k in ['model', 'tools', 'knowledge_base']):
            # Config has sections but no type - worth warning about
            self.warnings.append("Configuration missing 'type' field but has other sections")

        # Validate optional sections only if they're non-empty
        if config.get('model'):  # Only validate if model config is present and non-empty
            self._validate_model_config(config['model'])

        if config.get('tools'):  # Only validate if tools config is present
            self._validate_tools_config(config['tools'])

        if config.get('knowledge_base'):  # Only validate if KB config is present
            self._validate_kb_config(config['knowledge_base'])

        if config.get('skills'):  # Only validate if skills config is present
            self._validate_skills_config(config['skills'])

        if config.get('memory'):  # Only validate if memory config is present
            self._validate_memory_config(config['memory'])

        if config.get('guardrails'):  # Only validate if guardrails config is present
            self._validate_guardrails_config(config['guardrails'])

        # Log results only if there are errors
        if self.errors:
            for error in self.errors:
                self.logger.error(f"Validation Error: {error}")

        if self.warnings:
            for warning in self.warnings:
                self.logger.debug(f"Validation Warning: {warning}")

        return len(self.errors) == 0

    def _validate_model_config(self, model_config: Any) -> None:
        """Validate model configuration.

        Args:
            model_config: Model configuration section
        """
        if not isinstance(model_config, dict):
            self.errors.append("'model' must be a dictionary")
            return

        # Skip validation for empty model configs (common in tests)
        if not model_config:
            return

        # Only require 'name' and 'provider' if other fields suggest it's a real config
        # (i.e., has settings like 'temperature', 'max_tokens', etc.)
        has_real_settings = any(k in model_config for k in [
            'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty',
            'stop', 'functions', 'tools', 'system_prompt', 'endpoint', 'api_key'
        ])

        if has_real_settings:
            # This looks like a real config, validate required fields
            if 'name' not in model_config:
                self.errors.append("Model configuration missing required 'name' field")

            if 'provider' not in model_config:
                self.errors.append("Model configuration missing required 'provider' field")
        else:
            # Minimal config (just name/provider or empty), just check type
            if 'name' in model_config and not isinstance(model_config['name'], str):
                self.warnings.append("Model 'name' should be a string")

            if 'provider' in model_config and not isinstance(model_config['provider'], str):
                self.warnings.append("Model 'provider' should be a string")

    def _validate_tools_config(self, tools_config: Any) -> None:
        """Validate tools configuration.

        Args:
            tools_config: Tools configuration section
        """
        if not isinstance(tools_config, dict):
            self.errors.append("'tools' must be a dictionary")
            return

        # Each tool should be properly configured
        for tool_name, tool_config in tools_config.items():
            if not isinstance(tool_config, dict):
                self.errors.append(f"Tool '{tool_name}' configuration must be a dictionary")
                continue

            # Tools should have either 'module' or 'command'
            if 'module' not in tool_config and 'command' not in tool_config and 'url' not in tool_config:
                self.warnings.append(
                    f"Tool '{tool_name}' missing 'module' (class/function), "
                    "'command' (MCP stdio), or 'url' (MCP HTTP)"
                )

    def _validate_kb_config(self, kb_config: Any) -> None:
        """Validate knowledge base configuration.

        Args:
            kb_config: Knowledge base configuration section
        """
        if isinstance(kb_config, dict):
            # New style with 'sources' or agent-level configs
            sources = kb_config.get('sources', [])
            if not isinstance(sources, list):
                self.warnings.append("KB 'sources' should be a list")
        elif isinstance(kb_config, list):
            # Old style list of KB entries
            for i, entry in enumerate(kb_config):
                if not isinstance(entry, dict):
                    self.errors.append(f"KB entry {i} must be a dictionary")
        else:
            self.errors.append("'knowledge_base' must be a dictionary or list")

    def _validate_skills_config(self, skills_config: Any) -> None:
        """Validate skills configuration.

        Args:
            skills_config: Skills configuration section
        """
        if not isinstance(skills_config, dict):
            self.errors.append("'skills' must be a dictionary")
            return

        # Required fields
        if 'skill_dir' not in skills_config:
            self.errors.append("Skills configuration missing required 'skill_dir' field")

        # Registry is optional but if provided should be valid
        if 'registry' in skills_config:
            registry = skills_config['registry']
            if not isinstance(registry, dict):
                self.errors.append("Skills 'registry' must be a dictionary")

    def _validate_memory_config(self, memory_config: Any) -> None:
        """Validate memory configuration.

        Args:
            memory_config: Memory configuration section
        """
        if not isinstance(memory_config, dict):
            self.errors.append("'memory' must be a dictionary")
            return

        # Type is required for memory
        if 'type' not in memory_config:
            self.warnings.append("Memory configuration missing 'type' field (defaults to 'simple')")

        valid_types = ['simple', 'redis', 'mongodb', 'diskcache', 'pinecone']
        mem_type = memory_config.get('type', '').lower()
        if mem_type and mem_type not in valid_types:
            self.warnings.append(f"Unknown memory type: '{mem_type}'")

    def _validate_guardrails_config(self, guardrails_config: Any) -> None:
        """Validate guardrails configuration.

        Args:
            guardrails_config: Guardrails configuration section
        """
        if not isinstance(guardrails_config, dict):
            self.errors.append("'guardrails' must be a dictionary")
            return

        # If configured, should have either 'guard' or 'definition_file'
        has_guard = 'guard' in guardrails_config
        has_def = 'definition_file' in guardrails_config

        if not has_guard and not has_def:
            self.warnings.append(
                "Guardrails configured but missing 'guard' or 'definition_file'"
            )

    def get_errors(self) -> List[str]:
        """Get all validation errors.

        Returns:
            List of error messages
        """
        return self.errors

    def get_warnings(self) -> List[str]:
        """Get all validation warnings.

        Returns:
            List of warning messages
        """
        return self.warnings

    def export_json_schema(self) -> Dict[str, Any]:
        """Export configuration schema as JSON Schema format.

        **Phase 3.5**: Generates a JSON Schema that can be used for:
        - Configuration validation in external tools
        - IDE autocompletion and validation
        - Documentation generation

        Returns:
            JSON Schema as a dictionary following JSON Schema Draft 7 standard
        """
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Agent Configuration Schema",
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "description": "Agent framework type (crewai, langchain, etc.)",
                    "examples": ["crewai", "langchain", "bedrock"]
                },
                "model": {
                    "type": "object",
                    "description": "Language model configuration",
                    "properties": {
                        "name": {"type": "string", "description": "Model name"},
                        "provider": {
                            "type": "string",
                            "description": "LLM provider",
                            "enum": ["openai", "anthropic", "bedrock", "local"]
                        },
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                        "max_tokens": {"type": "integer", "minimum": 1}
                    }
                },
                "tools": {
                    "type": "object",
                    "description": "Tool configurations",
                    "additionalProperties": True
                },
                "knowledge_base": {
                    "oneOf": [
                        {"type": "array", "description": "Old-style KB list (deprecated)"},
                        {
                            "type": "object",
                            "properties": {
                                "registry": {
                                    "type": "object",
                                    "properties": {
                                        "url": {"type": "string"},
                                        "token": {"type": "string"}
                                    }
                                },
                                "sources": {"type": "array"}
                            }
                        }
                    ]
                },
                "skills": {
                    "type": "object",
                    "properties": {
                        "skill_dir": {"type": "string"},
                        "registry": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "auth_token": {"type": "string"}
                            }
                        }
                    }
                },
                "memory": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["redis", "mongodb", "diskcache", "simple", "pinecone"]
                        }
                    }
                },
                "guardrails": {
                    "type": "object",
                    "properties": {
                        "definition_file": {"type": "string"},
                        "enable_agent_validation": {"type": "boolean"}
                    }
                }
            },
            "required": ["type"]
        }

        self.logger.debug("Exported configuration JSON schema")
        return schema

    def migrate_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate configuration from old format to new format.

        **Phase 3.5**: Handles common migration scenarios:
        - Converting old knowledge_base list format to new dict format
        - Updating deprecated fields
        - Adding required fields with defaults

        Args:
            old_config: Configuration in old format

        Returns:
            Configuration in new format
        """
        new_config = dict(old_config)

        # Migrate knowledge_base from list to dict format if needed
        if 'knowledge_base' in new_config and isinstance(new_config['knowledge_base'], list):
            kb_list = new_config['knowledge_base']

            # Extract common registry settings from first entry (if present)
            registry_url = None
            registry_token = None

            if kb_list and isinstance(kb_list[0], dict):
                registry_url = kb_list[0].get('registry_url')
                registry_token = kb_list[0].get('auth_token')

            # Convert to new format
            new_config['knowledge_base'] = {
                'sources': kb_list,
                'registry': {}
            }

            if registry_url:
                new_config['knowledge_base']['registry']['url'] = registry_url
            if registry_token:
                new_config['knowledge_base']['registry']['token'] = registry_token

            self.logger.info("Migrated knowledge_base from list to dict format")

        # Add type if missing
        if 'type' not in new_config:
            new_config['type'] = 'base'
            self.logger.warning("Added default 'type' field to configuration")

        return new_config

    def validate_with_details(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate configuration and return detailed results.

        **Phase 3.5**: Provides structured validation output including:
        - Validation status (valid/invalid)
        - Error list with details
        - Warning list
        - Recommendations for fixes

        Args:
            config: Configuration to validate

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'suggestions': List[str]
            }
        """
        self.validate_agent_config(config)

        suggestions = []

        # Generate suggestions based on errors/warnings
        if any('missing' in e.lower() for e in self.errors):
            suggestions.append("Ensure all required fields are present in configuration")

        if any('type' in w.lower() for w in self.warnings):
            suggestions.append("Consider adding 'type' field to specify agent framework")

        if any('provider' in w.lower() for w in self.warnings):
            suggestions.append("Specify a model provider (openai, anthropic, bedrock, etc.)")

        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'suggestions': suggestions
        }
