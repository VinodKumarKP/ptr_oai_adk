"""Builder that converts YAML agent_list entries into claude-agent-sdk AgentDefinition objects.

Each agent in the ``agent_list`` YAML block becomes an ``AgentDefinition``
which the supervisor / agent-as-tool patterns pass to ``ClaudeAgentOptions.agents``.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from oai_agent_core.builders.base_agent_builder import BaseAgentBuilder
from oai_agent_core.components.output_parser.output_model_registry import OutputModelRegistry
from oai_agent_core.components.skills.skill_registry import SkillRegistry

from oai_agent_core.anthropic_core.components.configuration.model_config import AnthropicModelConfigurationManager
from oai_agent_core.anthropic_core.components.knowledge.knowledge_base_factory import KnowledgeBaseFactory
from oai_agent_core.anthropic_core.components.registry.tool_registry import AnthropicToolRegistry


class AgentBuilder(BaseAgentBuilder):
    """Builds ``AgentDefinition`` objects and resolves system prompts from YAML."""

    def __init__(
        self,
        model_manager: AnthropicModelConfigurationManager,
        tool_registry: AnthropicToolRegistry,
        llm: Optional[Dict[str, Any]] = None,
        config_root: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        document_loader=None,
        vector_store=None,
        skill_registry: Optional[SkillRegistry] = None,
        structured_output_model_registry: Optional[OutputModelRegistry] = None,
    ):
        super().__init__(
            model_manager=model_manager,
            tool_registry=tool_registry,
            llm=llm,
            config_root=config_root,
            logger=logger,
            document_loader=document_loader,
            vector_store=vector_store,
            skill_registry=skill_registry,
            structured_output_model_registry=structured_output_model_registry,
        )

    # ── Per-agent config resolution ───────────────────────────────────────────

    async def build_agent_definition(
        self, agent_key: str, agent_data: Dict[str, Any]
    ) -> "AgentDefinition":
        """Return a claude-agent-sdk ``AgentDefinition`` for one YAML agent entry."""
        from claude_agent_sdk import AgentDefinition

        system_prompt = agent_data.get("system_prompt", "You are a helpful assistant.")

        # Inject skills into system prompt
        skill_names = agent_data.get("skills", [])
        if skill_names and self.skill_registry:
            skill_list = self.skill_registry.get_skills(skill_names)
            system_prompt += "\n" + self.skill_registry.generate_skills_prompt(skill_list)

        # Collect allowed_tools.
        # - yaml_tools: bare names like "search_hotels" from the YAML tools list.
        #   These are custom Python tools loaded into the "custom_tools" SDK MCP
        #   server, so they're available as mcp__custom_tools__<name>.
        # - mcp_wildcards: auto-allow all tools on every registered MCP server.
        yaml_tools: List[str] = agent_data.get("tools", [])

        # Convert bare tool names → mcp__custom_tools__<name> when the tool
        # lives in our custom_tools registry.
        resolved_tools: List[str] = []
        for t in yaml_tools:
            if t in self.tool_registry.custom_tools:
                resolved_tools.append(f"mcp__custom_tools__{t}")
            else:
                resolved_tools.append(t)

        mcp_wildcards = self.tool_registry.get_tool_names_for_allowed_list()

        # MCPs at the agent level can be:
        #   list  — names referencing the global mcps: block already registered
        #           e.g.  mcps: [filesystem, database]
        #   dict  — inline per-agent MCP configs (register them now)
        #           e.g.  mcps: {local_mcp: {command: python, args: [...]}}
        agent_mcp = agent_data.get("mcps", {})
        if isinstance(agent_mcp, list):
            # References to globally-registered MCPs — just add wildcards
            mcp_wildcards += [f"mcp__{name}__*" for name in agent_mcp]
        elif isinstance(agent_mcp, dict) and agent_mcp:
            # Inline MCP configs — register and add wildcards
            self.tool_registry.load_mcp_configs(agent_mcp)
            mcp_wildcards += [f"mcp__{name}__*" for name in agent_mcp]

        allowed_tools = list(set(resolved_tools + mcp_wildcards))

        description = agent_data.get(
            "description",
            system_prompt[:120] + "..." if len(system_prompt) > 120 else system_prompt,
        )

        return AgentDefinition(
            description=description,
            prompt=system_prompt,
            tools=allowed_tools or None,
        )

    async def build_all_agent_definitions(
        self, agent_configs: List[Dict[str, Any]]
    ) -> Dict[str, "AgentDefinition"]:
        """Build all AgentDefinitions in parallel."""
        keys, tasks = [], []
        for entry in agent_configs:
            key = list(entry.keys())[0]
            data = entry[key]
            keys.append(key)
            tasks.append(self.build_agent_definition(key, data))

        results = await asyncio.gather(*tasks)
        return dict(zip(keys, results))

    def resolve_entry_system_prompt(
        self,
        agent_configs: List[Dict[str, Any]],
        global_system_prompt: Optional[str],
        entry_agent: Optional[str],
    ) -> str:
        """Return the system prompt for the top-level (entry) agent."""
        if global_system_prompt:
            return global_system_prompt

        # Fall back to the entry agent's system_prompt
        for entry in agent_configs:
            key = list(entry.keys())[0]
            if not entry_agent or key == entry_agent:
                prompt = entry[key].get("system_prompt", "")
                # Inject skills if present
                skill_names = entry[key].get("skills", [])
                if skill_names and self.skill_registry:
                    skill_list = self.skill_registry.get_skills(skill_names)
                    prompt += "\n" + self.skill_registry.generate_skills_prompt(skill_list)
                return prompt

        return "You are a helpful assistant."

    # ── BaseAgentBuilder abstract implementations ─────────────────────────────

    def _create_agent_instance(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        tools: List[Any],
    ) -> Any:
        """Return the agent config dict — claude-agent-sdk uses AgentDefinition,
        not a stateful agent object, so we pass the config through unchanged."""
        return agent_config

    def _create_agent_as_tool(self, agent: Any, name: str, description: str) -> Any:
        """In claude-agent-sdk the Agent tool IS the agent-as-tool mechanism.
        We return a minimal descriptor; OrchestrationBuilder converts it to an
        AgentDefinition registered in ClaudeAgentOptions.agents."""
        return {"name": name, "description": description, "agent": agent}

    def _create_supervisor_agent(
        self,
        crew_config: Dict[str, Any],
        agent_list: List[Any],
        sub_agent_tools: List[Any],
        system_prompt: str,
    ) -> Any:
        """Build a supervisor AgentDefinition that has all sub-agents as tools.

        In claude-agent-sdk the supervisor is the *main* agent — it receives
        the built-in Agent tool and a registered dict of sub-agents.  We return
        a plain dict descriptor; ``OrchestrationBuilder._build_options`` converts
        it into the correct ``ClaudeAgentOptions`` structure.
        """
        return {
            "system_prompt": system_prompt,
            "sub_agents": sub_agent_tools,
            "pattern": crew_config.get("pattern", "supervisor"),
        }

    async def _create_knowledge_base_tool(
        self, agent_name: str, kb_configs: List[Dict[str, Any]]
    ) -> List[Any]:
        """Create KB search + load callables and register them in the tool registry
        as custom tools so they are wrapped into the FastMCP server."""
        kb_factory = await asyncio.to_thread(
            KnowledgeBaseFactory,
            knowledge_base_config=kb_configs,
            logger=self.logger,
            project_root=getattr(self.tool_registry, "project_root", None),
            document_loader=self.document_loader,
            vector_store=self.vector_store,
        )

        tools = []
        for kb_name, kb_data in kb_factory.knowledge_base_tools.items():
            description = kb_data.get("description", f"Search {kb_name}")
            search_tool = kb_factory.create_tool(name=kb_name, description=description)
            load_tool = kb_factory.create_load_tool(
                name=kb_name, description=f"Load documents into {kb_name}"
            )
            # Register as custom tools so they appear in the FastMCP server
            self.tool_registry.custom_tools[search_tool.__name__] = search_tool
            self.tool_registry.custom_tools[load_tool.__name__] = load_tool
            tools.extend([search_tool, load_tool])

        return tools

    def _get_knowledgebase_factory_class(self) -> type:
        return KnowledgeBaseFactory

    def extract_context_map(self, agent_configs: List[Dict]) -> Dict[str, List[str]]:
        context_map = {}
        for entry in agent_configs:
            key = list(entry.keys())[0]
            context = entry[key].get("context", [])
            if context:
                context_map[key] = context
        return context_map
