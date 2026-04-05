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
    AgentSkill,
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
    resolved_url = (
        base_url
        or os.environ.get("AGENT_BASE_URL", "http://localhost:8000")
    ).rstrip("/")

    # --- Description ---
    description = os.environ.get("AGENT_DESCRIPTION", "")
    if agent_config:
        # Append system prompt from the main agent config
        if agent_config.get("system_prompt"):
            description += "\n" + agent_config["system_prompt"]

        # Append system prompts from the agent list
        agent_list = agent_config.get("agent_list", [])
        if len(agent_list) > 1:
            description += "\nYou have access to the following agents:"
            for agent_data in agent_list:
                for agent_name_key, config in agent_data.items():
                    if config.get("system_prompt"):
                        description += f"\n- {agent_name_key}: {config['system_prompt']}"
        elif len(agent_list) == 1:
            for agent_data in agent_list:
                for _, config in agent_data.items():
                    if config.get("system_prompt"):
                        description += "\n" + config["system_prompt"]

    if not description:
        description = f"{agent_name} — powered by OAI Agent Server"

    # --- Skills ---
    resolved_skills = skills
    if not resolved_skills and agent_config and agent_config.get("tools"):
        resolved_skills = [
            AgentSkill(
                id=tool,
                name=tool.replace("_", " ").title(),
                description=f"Use the {tool} tool",
                tags=["tool"],
            )
            for tool in agent_config["tools"]
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
        description=description.strip(),
        url=f"{resolved_url}/",
        version=os.environ.get("AGENT_VERSION", "1.0.0"),
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,
        ),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=resolved_skills,
    )