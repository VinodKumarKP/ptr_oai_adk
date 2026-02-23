import glob
import json
# Import the dynamic scripts function
import os
import sys
from pathlib import Path
from typing import Dict, Any

from setuptools import setup

from ruamel.yaml import YAML


yaml = YAML()

def load_server_config(config_directory) -> Dict[str, Any]:
    """
    Load server configuration from JSON and YAML files.

    Returns:
        Dict[str, Any]: Dictionary of server configurations
    """
    server_config = {}
    # Support .json, .yaml, and .yml files
    config_patterns = [
        os.path.join(config_directory, "*.json"),
        os.path.join(config_directory, "*.yaml"),
        os.path.join(config_directory, "*.yml"),
    ]

    config_files = []
    for pattern in config_patterns:
        config_files.extend(glob.glob(pattern))

    for config_file in config_files:
        server_name = os.path.basename(config_file).split('.')[0]
        ext = os.path.splitext(config_file)[1].lower()
        if ext == ".json":
            server_config[server_name] = load_individual_json_config(config_file)
        elif ext in [".yaml", ".yml"]:
            server_config[server_name] = load_individual_yaml_config(config_file)

    return server_config


def load_individual_yaml_config(config_file: str):
    """
    Load individual server configuration from YAML file.

    Args:
        config_file (str): Path to configuration file

    Returns:
        Dict[str, Any]: Configuration data
    """
    try:
        with open(config_file, "r", encoding='utf-8') as file:
            return yaml.load(file)
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading server configuration: {str(e)}")
        sys.exit(1)


def load_individual_json_config(config_file):
    """Load server configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in '{config_file}': {e}")
        sys.exit(1)


def discover_scripts():
    """Fallback script discovery during installation"""
    scripts = {}

    # Get the package directory
    package_dir = Path(__file__).parent
    servers_config_directory = os.path.join(package_dir, "mcp_registry_servers", "servers_config")

    config = load_server_config(config_directory=servers_config_directory)

    for server_name in config.keys():
        script_name = f"{server_name.replace('_server', '').replace('_', '-')}-mcp-server"
        entry_point = f"mcp_registry_servers.servers.{server_name}.server:main"
        scripts[script_name] = entry_point
        scripts[server_name] = entry_point

    return scripts


dynamic_scripts = discover_scripts()

# Convert to console_scripts format
console_scripts = [
    f"{script_name} = {entry_point}"
    for script_name, entry_point in dynamic_scripts.items()
]

if __name__ == "__main__":
    setup(
        install_requires=[
            "ruamel.yaml"
            # add other dependencies here
        ],
        # Let pyproject.toml handle most configuration
        entry_points={
            'console_scripts': console_scripts
        }
    )
