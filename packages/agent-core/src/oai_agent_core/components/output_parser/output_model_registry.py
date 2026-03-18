import logging
import inspect
import sys
from typing import Optional, Dict, Type
from pydantic import BaseModel
from oai_agent_core.components.output_parser.utils import is_safe_path, resolve_path
from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class OutputModelRegistry:
    """
    A registry for discovering and managing Pydantic output models.

    This class scans a designated directory to find all Python files,
    inspects them for classes that inherit from Pydantic's BaseModel,
    and loads them into a dictionary for easy access.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, project_root: Optional[str] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.project_root = project_root
        self.output_model_registry: Dict[str, Type[BaseModel]] = {}

    def discover_output_models(self, output_model_dir: str) -> None:
        """
        Discovers all Pydantic models in a directory and populates the registry.

        Args:
            output_model_dir: The path to the directory containing the Python
                              files with Pydantic model definitions.
        """
        if not output_model_dir:
            self.logger.info("No output model directory provided. Skipping model discovery.")
            return

        resolved_model_dir = resolve_path(output_model_dir, self.project_root)

        if not resolved_model_dir.exists() or not resolved_model_dir.is_dir():
            self.logger.warning(f"Output model directory not found or not a directory: {resolved_model_dir}")
            return

        self.logger.info(f"Discovering output models in: {resolved_model_dir}")

        # Add the directory to the Python path to allow for dynamic importing
        sys.path.insert(0, str(resolved_model_dir))

        try:
            for python_file in resolved_model_dir.glob("*.py"):
                if not is_safe_path(python_file, resolved_model_dir):
                    self.logger.warning(f"Skipping potentially unsafe file path: {python_file}")
                    continue

                if python_file.name == "__init__.py":
                    continue

                module_name = python_file.stem
                try:
                    module = DynamicClassLoader.dynamic_import_module(module_name)
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        # Ensure the class is a Pydantic model and was defined in the module itself
                        if issubclass(obj, BaseModel) and obj is not BaseModel and obj.__module__ == module_name:
                            self.logger.info(f"Discovered Pydantic model: {name} in {python_file.name}")
                            self.output_model_registry[name] = obj
                except Exception as e:
                    self.logger.error(f"Failed to load models from {python_file.name}: {e}")
        finally:
            # Clean up the system path
            sys.path.pop(0)

        self.logger.info(f"Discovery complete. Found {len(self.output_model_registry)} output models.")

    def get_model(self, name: str) -> Optional[Type[BaseModel]]:
        """
        Retrieves a Pydantic model by its class name.

        Args:
            name: The class name of the model to retrieve.

        Returns:
            The Pydantic model class if found, otherwise None.
        """
        return self.output_model_registry.get(name)

    def get_all_models(self) -> Dict[str, Type[BaseModel]]:
        """
        Retrieves all discovered Pydantic models.

        Returns:
            A dictionary of all Pydantic models in the registry.
        """
        return self.output_model_registry
