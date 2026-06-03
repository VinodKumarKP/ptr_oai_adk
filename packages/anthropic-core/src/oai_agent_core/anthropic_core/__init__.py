"""OAI Anthropic Agent Core — LiteLLM-backed Claude agent for OAI ADK.

Quick start::

    from oai_agent_core.anthropic_core.agents.anthropic_agent import AnthropicAgent

    agent = AnthropicAgent(
        agent_name="my_agent",
        agent_config=yaml.safe_load(open("agent.yaml")),
    )
    await agent.initialize()
    response = await agent.ainvoke("Summarise the latest product docs")
"""

from oai_agent_core.anthropic_core.agents.anthropic_agent import AnthropicAgent

__all__ = ["AnthropicAgent"]
