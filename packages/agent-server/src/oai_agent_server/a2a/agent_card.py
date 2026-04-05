"""
Builds an a2a-sdk AgentCard from environment variables and agent metadata.

Used by the server setup in main.py. Override any field by setting the
corresponding environment variable or by constructing AgentCard directly.
"""
import os
from typing import List, Optional

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
) -> AgentCard:
    """
    Build an AgentCard using a2a-sdk types.

    Priority for each field:
      explicit arg → environment variable → sensible default

    Environment variables:
      AGENT_BASE_URL          Public URL of this agent server
      AGENT_VERSION           Semver string (default: 1.0.0)
      AGENT_DESCRIPTION       One-line description
    """
    resolved_url = (
        base_url
        or os.environ.get("AGENT_BASE_URL", "http://localhost:8000")
    ).rstrip("/")

    resolved_skills = skills or [
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
        description=os.environ.get(
            "AGENT_DESCRIPTION", f"{agent_name} — powered by OAI Agent Server"
        ),
        url=f"{resolved_url}/",
        version=os.environ.get("AGENT_VERSION", "1.0.0"),
        capabilities=AgentCapabilities(
            streaming=streaming,
            push_notifications=push_notifications,

        ),
        default_input_modes=["text"],
        default_output_modes=["text"],
        skills=resolved_skills
    )