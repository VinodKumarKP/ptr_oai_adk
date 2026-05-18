"""
Builds an a2a-sdk AgentCard from environment variables and agent metadata.

Used by the server setup in main.py. Override any field by setting the
corresponding environment variable or by constructing AgentCard directly.
"""
import os
from typing import List, Optional, Dict, Any

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill, AgentInterface,
)


def build_agent_card(
    agent_name: str,
    base_url: Optional[str] = None,
    skills: Optional[List[AgentSkill]] = None,
    streaming: bool = True,
    push_notifications: bool = True,
    agent_config: Optional[Dict[str, Any]] = None,
) -> AgentCard:
    """
    Build an AgentCard using a2a-sdk types.

    Priority for each field:
      explicit arg → agent_config → environment variable → sensible default

    Environment variables:
      AGENT_BASE_URL          Public URL of this agent server
      AGENT_VERSION           Semver string (default: 1.0.0)
      AGENT_DESCRIPTION       One-line description
    """
    # Use a placeholder if no URL is provided; it will be updated in main.py
    default_url = os.environ.get('AGENT_BASE_URL')
    if default_url is not None:
        if ':' not in default_url:
            default_url = f"{os.environ.get('AGENT_BASE_URL', 'localhost')}:{os.environ.get('AGENT_BASE_URL_PORT', 8081)}"
    else:
        default_url = "http://placeholder.url"
    resolved_url = base_url or f"{default_url}/{agent_name.lower()}"

    if not resolved_url.startswith('http'):
        resolved_url = f"http://{resolved_url}"

    # --- Description ---
    description = os.environ.get("AGENT_DESCRIPTION", "")

    if isinstance(agent_config, dict):
        # Append system prompt from the main agent config
        if agent_config.get("system_prompt"):
            description += "\n" + str(agent_config["system_prompt"])

        # Append system prompts from the agent list
        agent_list = agent_config.get("agent_list", [])
        if isinstance(agent_list, list):
            if len(agent_list) > 1:
                description += "\nYou have access to the following agents:"
                for agent_data in agent_list:
                    if isinstance(agent_data, dict):
                        for agent_name_key, config in agent_data.items():
                            if isinstance(config, dict) and config.get("system_prompt"):
                                description += f"\n- {agent_name_key}: {config['system_prompt']}"
            elif len(agent_list) == 1:
                for agent_data in agent_list:
                    if isinstance(agent_data, dict):
                        for _, config in agent_data.items():
                            if isinstance(config, dict) and config.get("system_prompt"):
                                description += "\n" + str(config["system_prompt"])

    if not description or not isinstance(description, str):
        # Default description if not provided or if mocked unexpectedly
        description = f"{agent_name} — powered by OAI Agent Server"
    else:
        description = description.strip()

    # --- Skills ---
    resolved_skills = skills
    
    tools = []
    if isinstance(agent_config, dict) and isinstance(agent_config.get("tools"), list):
        tools = agent_config.get("tools")
        
    if not resolved_skills and tools:
        resolved_skills = [
            AgentSkill(
                id=str(tool),
                name=str(tool).replace("_", " ").title(),
                description=f"Use the {tool} tool",
                tags=["tool"],
            )
            for tool in tools
        ]

    if not resolved_skills:
        resolved_skills = [
            AgentSkill(
                id="chat",
                name="Chat",
                description=f"Send a message to {agent_name} and receive a response",
                tags=["chat", "text"],
                examples=["Hello, what can you help me with?"],
            )
        ]

    return AgentCard(
        name=agent_name,
        description=description,
        version=os.environ.get("AGENT_VERSION", "1.0.0"),
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
        ),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=resolved_skills,
        supported_interfaces=[
            # JSON-RPC
            AgentInterface(
                protocol_binding='JSONRPC',
                url=f"{resolved_url}/a2a/",
            ),
            AgentInterface(
                protocol_binding='HTTP+JSON',
                url=f"{resolved_url}/chat/",
            ),
            AgentInterface(
                protocol_binding='HTTP+JSON',
                url=f"{resolved_url}/chat/stream",
            )
        ],
    )
