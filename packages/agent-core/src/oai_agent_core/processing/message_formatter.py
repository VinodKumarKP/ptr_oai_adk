"""Message formatting and template variable substitution for agent inputs."""

import logging
import re
from typing import Dict, Any, List, Set, Optional


class MessageFormatter:
    """Handles message formatting and template variable substitution.

    Supports:
    - Template variable extraction from text ({variable})
    - Variable substitution in messages
    - Input extraction from configuration
    - Default value handling

    Attributes:
        logger: Logger instance
    """

    VARIABLE_PATTERN = re.compile(r'\{(\w+)\}')

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the message formatter.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def format_message(
            self,
            message: str,
            inputs: Dict[str, Any]
    ) -> str:
        """Format message by replacing {variable} patterns with values.

        Args:
            message: Message template with {variable} placeholders
            inputs: Dictionary mapping variable names to values

        Returns:
            Formatted message with variables replaced

        Example:
            >>> formatter = MessageFormatter()
            >>> msg = "Hello {name}, welcome to {city}!"
            >>> result = formatter.format_message(msg, {'name': 'Alice', 'city': 'NYC'})
            >>> print(result)
            Hello Alice, welcome to NYC!
        """
        if not inputs:
            return message

        formatted = message
        for key, value in inputs.items():
            placeholder = f"{{{key}}}"
            formatted = formatted.replace(placeholder, str(value))

        # Log any unfilled placeholders
        remaining = self.extract_variables(formatted)
        if remaining:
            self.logger.warning(
                f"Unfilled template variables in message: {remaining}"
            )

        return formatted

    def extract_variables(self, text: str) -> Set[str]:
        """Extract all {variable} placeholders from text.

        Args:
            text: Text containing template variables

        Returns:
            Set of variable names found

        Example:
            >>> formatter = MessageFormatter()
            >>> text = "Process {input} and save to {output}"
            >>> vars = formatter.extract_variables(text)
            >>> print(vars)
            {'input', 'output'}
        """
        return set(self.VARIABLE_PATTERN.findall(text))

    def extract_variables_from_config(
            self,
            agent_config: Dict[str, Any]
    ) -> Set[str]:
        """Extract all template variables from agent configuration.

        Searches through:
        - Agent system prompts and backstories
        - Task descriptions
        - Crew-level task descriptions

        Args:
            agent_config: Complete agent configuration dictionary

        Returns:
            Set of all unique variable names found
        """
        variables = set()

        # Extract from agent-level configurations
        for agent_cfg in agent_config.get('agent_list', []):
            if isinstance(agent_cfg, dict):
                agent_key = list(agent_cfg.keys())[0]
                agent_data = agent_cfg[agent_key]
            else:
                agent_key = agent_cfg
                agent_data = {}

            # Extract from system prompt
            system_prompt = agent_data.get('system_prompt', agent_data.get('backstory', ''))
            variables.update(self.extract_variables(system_prompt))

            # Extract from agent tasks
            for task_cfg in agent_data.get('tasks', []):
                task_key = list(task_cfg.keys())[0]
                task_data = task_cfg[task_key]
                description = task_data.get('description', '')
                variables.update(self.extract_variables(description))

        # Extract from crew-level tasks
        for task_cfg in agent_config.get('task_list', []):
            task_key = list(task_cfg.keys())[0]
            task_data = task_cfg[task_key]
            description = task_data.get('description', '')
            variables.update(self.extract_variables(description))

        self.logger.debug(f"Extracted {len(variables)} template variables: {variables}")
        return variables

    def get_inputs(
            self,
            user_message: str,
            config: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare and format user message with inputs.

        Args:
            user_message: Raw user message
            config: Optional configuration with inputs

        Returns:
            Formatted message string
        """
        # Get inputs from config or extract defaults
        if config and 'inputs' in config:
            inputs = config['inputs']
        else:
            # Extract template variables from configuration
            variables = self.extract_variables_from_config(
                config
            )
            inputs = self.create_default_inputs(
                user_message,
                variables
            )

        # Format the message
        return inputs

    def create_default_inputs(
            self,
            user_message: Any,
            variables: Set[str]
    ) -> Dict[str, Any]:
        """Create default input dictionary from user message and variables.

        Maps user message to extracted variables with sensible defaults.

        Args:
            user_message: User input (string or dict)
            variables: Set of variable names that need values

        Returns:
            Dictionary mapping variables to values

        Example:
            >>> formatter = MessageFormatter()
            >>> inputs = formatter.create_default_inputs(
            ...     "Write a blog post",
            ...     {'topic', 'style'}
            ... )
            >>> print(inputs)
            {'topic': 'Write a blog post', 'style': 'Write a blog post'}
        """
        inputs = {}
        variables_list = sorted(list(variables))

        if not variables_list:
            return inputs

        if isinstance(user_message, dict):
            # User provided structured input
            inputs = user_message.copy()
        elif isinstance(user_message, str) and user_message:
            # Map string to all variables (user can override via config)
            for var in variables_list:
                inputs[var] = user_message
        else:
            self.logger.warning("No user message provided for template variables")

        return inputs

    def merge_inputs(
            self,
            default_inputs: Dict[str, Any],
            override_inputs: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge default and override inputs, preferring overrides.

        Args:
            default_inputs: Default input values
            override_inputs: Override values (can be None)

        Returns:
            Merged input dictionary
        """
        if not override_inputs:
            return default_inputs.copy()

        merged = default_inputs.copy()
        merged.update({
            k: v for k, v in override_inputs.items()
            if v is not None
        })

        return merged

    def validate_inputs(
            self,
            inputs: Dict[str, Any],
            required_variables: Set[str]
    ) -> tuple[bool, List[str]]:
        """Validate that all required variables have values.

        Args:
            inputs: Input dictionary to validate
            required_variables: Set of required variable names

        Returns:
            Tuple of (is_valid, missing_variables)

        Example:
            >>> formatter = MessageFormatter()
            >>> is_valid, missing = formatter.validate_inputs(
            ...     {'name': 'Alice'},
            ...     {'name', 'age'}
            ... )
            >>> print(is_valid, missing)
            False ['age']
        """
        missing = []

        for var in required_variables:
            if var not in inputs or inputs[var] is None or inputs[var] == '':
                missing.append(var)

        is_valid = len(missing) == 0

        if not is_valid:
            self.logger.warning(f"Missing required variables: {missing}")

        return is_valid, missing

    def format_with_validation(
            self,
            message: str,
            inputs: Dict[str, Any],
            strict: bool = False
    ) -> str:
        """Format message with optional strict validation.

        Args:
            message: Message template
            inputs: Input values
            strict: If True, raise error for missing variables

        Returns:
            Formatted message

        Raises:
            ValueError: If strict=True and variables are missing
        """
        required_vars = self.extract_variables(message)
        is_valid, missing = self.validate_inputs(inputs, required_vars)

        if not is_valid and strict:
            raise ValueError(
                f"Missing required template variables: {missing}. "
                f"Provided inputs: {list(inputs.keys())}"
            )

        return self.format_message(message, inputs)

    def get_template_info(self, text: str) -> Dict[str, Any]:
        """Get information about template variables in text.

        Args:
            text: Text to analyze

        Returns:
            Dictionary with template information
        """
        variables = self.extract_variables(text)

        return {
            'variable_count': len(variables),
            'variables': sorted(list(variables)),
            'has_variables': len(variables) > 0,
            'template_text': text
        }

    def __repr__(self) -> str:
        """String representation of the formatter."""
        return "MessageFormatter()"
