"""Generic launcher for the OAI Agent HTTP Server.

Loads an agent by name from its YAML config, resolves the agent framework
from the config's ``type`` field, wraps it in :class:`AgentHTTPServer`, and
serves it. This is the entry point used by container images (Docker CMD /
AWS Bedrock AgentCore) and the ``oai-agent-server`` console script:

    oai-agent-server <agent_name> [--port 8000] [--config-root .]
    python -m oai_agent_server.cli <agent_name>

All options fall back to environment variables so a container image can be
parameterized without changing its CMD:

    AGENT_NAME     agent to launch (when no positional arg is given)
    CONFIG_ROOT    directory containing agents_config/<agent_name>.yaml
    PORT           listen port (AgentCore contract: HTTP=8080, A2A=9000)
    AGENT_CLASS    explicit "module:Class" override for custom frameworks

The framework module is imported lazily — only the framework the agent
actually uses is loaded, keeping per-session cold start low on serverless
runtimes such as AgentCore.
"""
import argparse
import importlib
import os
import sys
from typing import Optional, Tuple

# Maps the agent config ``type`` field to (module, class). Only the resolved
# entry is imported. "langchain" is a legacy alias for the langgraph runtime.
FRAMEWORK_REGISTRY = {
    "langgraph": ("oai_agent_core.langgraph_core.agents.langgraph_agent", "LangGraphAgent"),
    "langchain": ("oai_agent_core.langgraph_core.agents.langgraph_agent", "LangGraphAgent"),
    "strands": ("oai_agent_core.aws_strands_core.agents.aws_strands_agent", "StrandsAgent"),
    "openai": ("oai_agent_core.openai_core.agents.openai_agent", "OpenAIAgent"),
    "anthropic": ("oai_agent_core.anthropic_core.agents.anthropic_agent", "AnthropicAgent"),
    "crewai": ("oai_agent_core.crewai_core.agents.crewai_agent", "CrewAIAgent"),
}


class AgentLaunchError(Exception):
    """Raised when the agent cannot be resolved or constructed."""


def _resolve_agent_class_path(agent_config: dict) -> Tuple[str, str]:
    """Resolve (module, class) for the agent, in precedence order:

    1. ``AGENT_CLASS`` env var — "package.module:ClassName"
    2. ``agent_class`` key in the agent config — same format
    3. ``type`` key in the agent config via FRAMEWORK_REGISTRY
    """
    explicit = os.environ.get("AGENT_CLASS") or agent_config.get("agent_class")
    if explicit:
        module_path, sep, class_name = explicit.partition(":")
        if not sep or not module_path or not class_name:
            raise AgentLaunchError(
                f"Invalid agent class spec {explicit!r}: expected 'package.module:ClassName'"
            )
        return module_path, class_name

    agent_type = (agent_config.get("type") or "").strip().lower()
    if not agent_type:
        raise AgentLaunchError(
            "Agent config has no 'type' field. Set one of "
            f"{sorted(set(FRAMEWORK_REGISTRY))} or provide AGENT_CLASS."
        )
    if agent_type not in FRAMEWORK_REGISTRY:
        raise AgentLaunchError(
            f"Unsupported agent type {agent_type!r}. Supported: "
            f"{sorted(set(FRAMEWORK_REGISTRY))}; or provide AGENT_CLASS "
            "as 'package.module:ClassName'."
        )
    return FRAMEWORK_REGISTRY[agent_type]


def _load_agent_class(module_path: str, class_name: str):
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise AgentLaunchError(
            f"Could not import {module_path!r} ({e}). Is the framework package "
            "installed in this environment/image?"
        ) from e
    try:
        return getattr(module, class_name)
    except AttributeError as e:
        raise AgentLaunchError(
            f"Module {module_path!r} has no class {class_name!r}."
        ) from e


def build_server(
    agent_name: str,
    config_root: Optional[str] = None,
    allowed_modes: Optional[list] = None,
):
    """Construct an :class:`AgentHTTPServer` for *agent_name* from config.

    Split from :func:`main` so tests and embedders can build without serving.
    """
    # Imported here (not module level) so `--help` stays instant and import
    # errors surface with launch context.
    from oai_agent_core.components.configuration.model_config import ConfigManager
    from oai_agent_server.main import AgentHTTPServer

    config_root = config_root or os.environ.get("CONFIG_ROOT") or os.getcwd()

    config_manager = ConfigManager(config_root=config_root)
    agent_config = config_manager.load_agent_config(
        agent_name=agent_name, abort_if_not_found=False
    )
    if not agent_config:
        raise AgentLaunchError(
            f"No config found for agent {agent_name!r} under "
            f"{config_root}/agents_config/"
        )

    module_path, class_name = _resolve_agent_class_path(agent_config)
    agent_class = _load_agent_class(module_path, class_name)

    agent = agent_class(
        agent_name=agent_name,
        agent_config=agent_config,
        config_root=config_root,
    )

    return AgentHTTPServer(
        agent=agent,
        agent_name=agent_name,
        config_root=config_root,
        allowed_modes=allowed_modes,
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="oai-agent-server",
        description="Serve an OAI ADK agent over HTTP (REST + A2A + AgentCore contracts)",
    )
    parser.add_argument(
        "agent_name", nargs="?", default=None,
        help="Agent to launch (defaults to $AGENT_NAME)",
    )
    parser.add_argument("--port", "-p", type=int, default=None,
                        help="Listen port (defaults to $PORT, then agent config, then 8000)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--config-root", default=None,
                        help="Directory containing agents_config/ (defaults to $CONFIG_ROOT, then CWD)")
    parser.add_argument("--allowed-modes", nargs="+", default=None,
                        help="API modes to enable (always-active modes are added automatically)")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    agent_name = args.agent_name or os.environ.get("AGENT_NAME")
    if not agent_name:
        print(
            "error: no agent specified — pass <agent_name> or set AGENT_NAME",
            file=sys.stderr,
        )
        raise SystemExit(2)

    try:
        server = build_server(
            agent_name=agent_name,
            config_root=args.config_root,
            allowed_modes=args.allowed_modes,
        )
    except AgentLaunchError as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)

    # An explicit --port wins over the PORT env override applied in run().
    if args.port is not None:
        os.environ["PORT"] = str(args.port)

    port = args.port or int(os.environ.get("PORT", 0)) or \
        server.agent.agent_config.get("port", 8000)
    server.agent.agent_config["port"] = port

    server.run(host=args.host, port=port)


if __name__ == "__main__":
    main()
