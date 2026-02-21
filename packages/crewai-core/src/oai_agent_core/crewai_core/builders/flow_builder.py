"""FlowBuilder for CrewAI Flow instances.

This module handles the construction and configuration of CrewAI Flow instances
from YAML configuration files.
"""

import os
import sys
import importlib.util
from typing import Dict, Any, Optional
from pathlib import Path
from crewai import Flow


class FlowBuilder:
    """Builds and configures CrewAI Flow instances from configuration.

    This class handles:
    - Loading Python flow scripts from disk
    - Importing and instantiating flow classes
    - Injecting tools and LLM into flow instances
    - Validating flow configuration

    Attributes:
        config: Agent configuration dictionary
        tool_registry: Registry of available tools
        llm: Language model instance
        logger: Logger instance
    """

    def __init__(self, config: Dict[str, Any], tool_registry, llm, logger, project_root:None):
        """Initialize the FlowBuilder.

        Args:
            config: Agent configuration dictionary
            tool_registry: ToolRegistry instance with loaded tools
            llm: LLM instance to inject into flows
            logger: Logger instance for debugging
            project_root: Root directory of the project
        """
        self.config = config
        self.tool_registry = tool_registry
        self.llm = llm
        self.logger = logger
        self.flow_config = config.get('flow_config', {})
        self.project_root = project_root

    def build_flow(self, session_id: str, inputs: Optional[Dict[str, Any]] = None) -> Flow:
        """Build and configure a Flow instance.

        Args:
            session_id: Session identifier for tracking
            inputs: Optional initial inputs for the flow

        Returns:
            Configured Flow instance ready for kickoff

        Raises:
            ValueError: If flow configuration is invalid or script cannot be loaded
        """
        # Validate configuration first
        validation = self.validate_flow_configuration()
        if not validation['valid']:
            raise ValueError(
                f"Invalid flow configuration: {validation.get('errors', [])}"
            )

        # Extract flow configuration
        script_path = self.flow_config.get('script_path')
        flow_class_name = self.flow_config.get('flow_class')

        # Load the flow class from script
        flow_class = self._load_flow_class(script_path, flow_class_name)

        # Prepare tools as a dictionary of {name: function}
        tools_dict = self._prepare_tools()

        # Prepare inputs
        flow_inputs = inputs or {}

        # Instantiate the flow with injected dependencies
        try:
            flow_instance = flow_class(
                tools=tools_dict,
                llm=self.llm,
                inputs=flow_inputs
            )

            self.logger.info(
                f"Successfully built flow '{flow_class_name}' from {script_path}"
            )

            return flow_instance

        except Exception as e:
            self.logger.error(f"Error instantiating flow class: {e}")
            raise ValueError(
                f"Failed to instantiate flow class '{flow_class_name}': {e}\n"
                f"Ensure the class constructor accepts 'tools', 'llm', and 'inputs' parameters."
            )

    def _load_flow_class(self, script_path: str, class_name: str):
        """Load a Flow class from a Python script.

        Args:
            script_path: Path to the Python script
            class_name: Name of the Flow class to import

        Returns:
            Flow class object

        Raises:
            ValueError: If script or class cannot be loaded
        """
        # Resolve the full path
        if self.project_root and script_path and script_path.startswith('.'):
            script_path = Path(str(script_path).replace('.', self.project_root, 1))
        else:
            script_path = Path(script_path).expanduser()

        if not os.path.isabs(script_path):
            # If relative path, make it relative to config root or current directory
            config_root = self.config.get('config_root', '.')
            script_path = os.path.join(config_root, script_path)

        script_path = os.path.abspath(script_path)

        # Check if file exists
        if not os.path.exists(script_path):
            raise ValueError(f"Flow script not found: {script_path}")

        # Load the module
        try:
            module_name = Path(script_path).stem
            spec = importlib.util.spec_from_file_location(module_name, script_path)

            if spec is None or spec.loader is None:
                raise ValueError(f"Cannot load module spec from {script_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            self.logger.info(f"Successfully loaded module from {script_path}")

        except Exception as e:
            raise ValueError(f"Error loading flow script {script_path}: {e}")

        # Get the class from the module
        if not hasattr(module, class_name):
            available_classes = [
                name for name in dir(module)
                if not name.startswith('_') and isinstance(getattr(module, name), type)
            ]
            raise ValueError(
                f"Class '{class_name}' not found in {script_path}. "
                f"Available classes: {available_classes}"
            )

        flow_class = getattr(module, class_name)

        # Validate it's a Flow class
        try:
            from crewai.flow.flow import Flow
            if not issubclass(flow_class, Flow):
                raise ValueError(
                    f"Class '{class_name}' must inherit from crewai.flow.flow.Flow"
                )
        except ImportError:
            self.logger.warning(
                "Could not import crewai.flow.flow.Flow for validation. "
                "Skipping inheritance check."
            )

        return flow_class

    def _prepare_tools(self) -> Dict[str, Any]:
        """Prepare tools as a dictionary for injection into flow.

        Returns:
            Dictionary mapping tool names to tool functions/objects
        """
        tools_dict = {}

        # Get all registered tools from the tool registry
        if hasattr(self.tool_registry, 'tools'):
            for tool_name, tool_obj in self.tool_registry.tools.items():
                # If it's a CrewAI tool object, we can pass it directly
                # If it's a function, wrap it or pass as-is
                tools_dict[tool_name] = tool_obj

                self.logger.debug(f"Added tool '{tool_name}' to flow tools dict")

        return tools_dict

    def validate_flow_configuration(self) -> Dict[str, Any]:
        """Validate the flow configuration.

        Returns:
            Dictionary with validation results:
            {
                'valid': bool,
                'errors': List[str],
                'warnings': List[str],
                'flow_info': Dict[str, Any]
            }
        """
        errors = []
        warnings = []
        flow_info = {}

        # Check if flow_config exists
        if not self.flow_config:
            errors.append("'flow_config' section is missing from configuration")
            return {
                'valid': False,
                'errors': errors,
                'warnings': warnings,
                'flow_info': flow_info
            }

        # Check required fields
        script_path = self.flow_config.get('script_path')
        flow_class = self.flow_config.get('flow_class')

        if not script_path:
            errors.append("'script_path' is required in flow_config")

        if not flow_class:
            errors.append("'flow_class' is required in flow_config")

        # If we have script_path, check if file exists
        if script_path:
            if self.project_root and script_path and script_path.startswith('.'):
                script_path = Path(str(script_path).replace('.', self.project_root, 1))
            else:
                script_path = Path(script_path).expanduser()

            if not os.path.isabs(script_path):
                # If relative path, make it relative to config root or current directory
                config_root = self.config.get('config_root', '.')
                script_path = os.path.join(config_root, script_path)

            full_path = os.path.abspath(script_path)

            if not os.path.exists(full_path):
                errors.append(f"Flow script not found: {full_path}")
            else:
                flow_info['script_path'] = full_path
                flow_info['script_exists'] = True

                # Try to do basic validation of the class
                if flow_class:
                    try:
                        # Quick check if class exists in file
                        with open(full_path, 'r') as f:
                            content = f.read()
                            if f"class {flow_class}" not in content:
                                warnings.append(
                                    f"Class '{flow_class}' not found in script. "
                                    f"Please verify the class name."
                                )

                            # Check for @start decorator
                            if "@start" not in content:
                                warnings.append(
                                    "No @start() decorator found in flow script. "
                                    "Flows require at least one @start() method."
                                )

                            flow_info['has_start_decorator'] = "@start" in content
                            flow_info['class_definition_found'] = f"class {flow_class}" in content

                    except Exception as e:
                        warnings.append(f"Could not read script file: {e}")

        # Check if tools are configured
        tools_config = self.config.get('tools', {})
        flow_info['tools_configured'] = len(tools_config) > 0
        flow_info['tool_count'] = len(tools_config)

        # Determine validity
        valid = len(errors) == 0

        return {
            'valid': valid,
            'errors': errors,
            'warnings': warnings,
            'flow_info': flow_info
        }

    def get_flow_metadata(self) -> Dict[str, Any]:
        """Get metadata about the flow configuration.

        Returns:
            Dictionary with flow metadata
        """
        return {
            'mode': 'flow',
            'script_path': self.flow_config.get('script_path'),
            'flow_class': self.flow_config.get('flow_class'),
            'validation': self.validate_flow_configuration()
        }