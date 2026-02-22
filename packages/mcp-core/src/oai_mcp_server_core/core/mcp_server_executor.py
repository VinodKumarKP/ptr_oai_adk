import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def get_server_config(server_name: str) -> dict:
    """
    Get server configuration from YAML file in servers_config directory.

    Args:
        server_name (str): Name of the server to get config for

    Returns:
        dict: Server configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    # Construct the path to the config file
    config_dir = Path(__file__).parent.parent
    config_file = config_dir / "servers_config" / f"{server_name}.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
            return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML config for {server_name}: {e}")


def get_source_url(server_name: str) -> str:
    """
    Get the source URL from server configuration.

    Args:
        server_name (str): Name of the server

    Returns:
        str: Source URL from the configuration

    Raises:
        KeyError: If 'source' key is not found in config
        FileNotFoundError: If config file doesn't exist
    """
    config = get_server_config(server_name)

    if 'source' not in config:
        raise KeyError(f"'source' key not found in configuration for {server_name}")

    return config['source']


def _run_if_exists(cmd: str, args: list = None) -> int:
    """
    Run command if it exists in PATH.
    
    Args:
        cmd: Command to run
        args: List of arguments
        
    Returns:
        Return code of the command
    """
    if not shutil.which(cmd):
        print(f"Error: {cmd} not found in PATH")
        return 1

    full_cmd = [cmd] + (args or [])   # Skip script name and server name
    return subprocess.run(full_cmd).returncode


def uv(server_name: str, remaining_arg: list) -> int:
    """
    Run uv with server-specific source URL.
    
    Args:
        server_name: Name of the server
        remaining_arg: Remaining arguments to pass to uv
        
    Returns:
        Return code of the command
    """
    try:
        source_url = get_source_url(server_name)
        print(f"Using source URL for {server_name}: {source_url}")

        args = [
            "run",
            "--with",
            source_url,
            server_name
        ]

        args.extend(remaining_arg)
        return _run_if_exists('uv', args)
    except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
        print(f"Error getting configuration for {server_name}: {e}")
        return 1


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(description="MCP Server Executor")
    parser.add_argument("server_name", help="Name of the server to run")

    # Parse only the first argument (server_name), leave the rest for the subprocess
    args, remaining = parser.parse_known_args()

    # Update sys.argv to include remaining arguments for subprocess
    sys.argv = [sys.argv[0]] + remaining

    return uv(args.server_name, remaining)


if __name__ == "__main__":
    sys.exit(main())
