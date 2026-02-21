"""Dynamic class loading utility for importing Python modules and classes."""

from importlib import import_module
from typing import Any


class DynamicClassLoader:
    """Utility class for dynamically loading Python classes from module paths.

    This class provides static methods for runtime import of classes and modules,
    useful for plugin architectures and configuration-driven systems.
    """

    @staticmethod
    def dynamic_import(module_name: str, class_name: str) -> Any:
        """Dynamically import a class from a module.

        Args:
            module_name: Full module path (e.g., 'strands_tools')
            class_name: Name of the class to import

        Returns:
            The imported class

        Raises:
            ImportError: If module or class cannot be found
            AttributeError: If class doesn't exist in the module

        Example:
            >>> Calculator = DynamicClassLoader.dynamic_import('my_tools', 'Calculator')
            >>> calc = Calculator()
        """
        try:
            module = import_module(module_name)
            return getattr(module, class_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_name}': {e}"
            ) from e
        except AttributeError as e:
            raise AttributeError(
                f"Class '{class_name}' not found in module '{module_name}': {e}"
            ) from e

    @staticmethod
    def dynamic_import_module(module_name: str) -> Any:
        """Dynamically import a module.

        Args:
            module_name: Full module path (e.g., 'strands_tools')

        Returns:
            The imported module

        Raises:
            ImportError: If module cannot be found

        Example:
            >>> tools_module = DynamicClassLoader.dynamic_import_module('strands_tools')
        """
        try:
            return import_module(module_name)
        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_name}': {e}"
            ) from e

    @staticmethod
    def dynamic_import_tool(tool_name: str, base_module: str = 'strands_tools') -> Any:
        """Import a tool module from a base package.

        Args:
            tool_name: Name of the tool to import
            base_module: Base module path (default: 'strands_tools')

        Returns:
            The imported tool module

        Raises:
            ImportError: If the tool module cannot be found

        Example:
            >>> calculator = DynamicClassLoader.dynamic_import_tool('calculator')
        """
        try:
            full_module_path = f"{base_module}.{tool_name}"
            return __import__(full_module_path, fromlist=[tool_name])
        except ImportError as e:
            raise ImportError(
                f"Failed to import tool '{tool_name}' from '{base_module}': {e}"
            ) from e
