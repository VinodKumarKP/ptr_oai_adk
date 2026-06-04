#!/usr/bin/env python
"""
Test script to verify agent-level knowledge base tool loading.

Tests that:
1. Agent-level KB configs are processed during agent definition building
2. KB tools are registered in the tool registry
3. KB tools are included in the MCP server
"""

import asyncio
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_agent_kb_loading(config_file: str = "knowledge_base_as_tool_agent.yaml"):
    """Test agent-level KB tool loading.

    Args:
        config_file: Name of config file to test (in examples/agents_config/)
    """
    from oai_agent_core.anthropic_core.agents.anthropic_agent import AnthropicAgent

    # Path to test config
    config_path = Path(__file__).parent / "examples" / "agents_config" / config_file

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False

    print(f"✓ Config file found: {config_path}")

    try:
        # Initialize agent
        agent = AnthropicAgent(config_file=str(config_path))
        await agent.initialize()

        print("\n" + "="*60)
        print("Agent Initialization Summary")
        print("="*60)

        # Check tool registry
        print(f"\n📋 Custom Tools in Registry:")
        if agent.tool_registry.custom_tools:
            for tool_name, tool_func in agent.tool_registry.custom_tools.items():
                print(f"  - {tool_name}")
        else:
            print("  (none)")

        # Check KB tools specifically
        kb_tools = {k: v for k, v in agent.tool_registry.custom_tools.items()
                   if 'search' in k.lower() or 'load' in k.lower() or 'insurance' in k.lower() or 'leave' in k.lower()}

        print(f"\n🔍 KB Tools Found:")
        if kb_tools:
            for tool_name in kb_tools.keys():
                print(f"  ✓ {tool_name}")
        else:
            print("  ❌ No KB tools found!")

        # Check MCP server
        print(f"\n🔌 MCP Server Configs:")
        mcp_servers = agent.tool_registry.get_mcp_server_configs()
        if mcp_servers:
            for server_name in mcp_servers.keys():
                print(f"  - {server_name}")
                if server_name == "custom_tools":
                    print(f"    Tools in server: {len(mcp_servers[server_name].tools or [])}")
        else:
            print("  (none)")

        # Check agent definitions
        if agent.agent_definitions:
            print(f"\n👥 Agent Definitions:")
            for agent_name, agent_def in agent.agent_definitions.items():
                print(f"  - {agent_name}")
                if agent_def.tools:
                    print(f"    Allowed tools: {agent_def.tools[:3]}..." if len(agent_def.tools) > 3 else f"    Allowed tools: {agent_def.tools}")

        print("\n" + "="*60)

        # Verify KB tools are accessible
        expected_kb_tools = ['search_insurance_policies', 'load_insurance_policies',
                            'search_leave_policies', 'load_leave_policies']

        found_tools = []
        missing_tools = []

        for expected in expected_kb_tools:
            if expected in agent.tool_registry.custom_tools:
                found_tools.append(expected)
            else:
                missing_tools.append(expected)

        print(f"\n📊 Verification Results:")
        print(f"  Found KB tools: {len(found_tools)}/{len(expected_kb_tools)}")

        if found_tools:
            print(f"  ✓ Found:")
            for tool in found_tools:
                print(f"    - {tool}")

        if missing_tools:
            print(f"  ❌ Missing:")
            for tool in missing_tools:
                print(f"    - {tool}")
            return False

        print("\n✅ All agent-level KB tools loaded successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during agent initialization:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_both_patterns():
    """Test both single and multi-agent patterns."""
    print("\n" + "="*60)
    print("Testing SINGLE PATTERN with Agent-Level KB")
    print("="*60)
    success_single = await test_agent_kb_loading("knowledge_base_single_pattern.yaml")

    print("\n" + "="*60)
    print("Testing AGENT-AS-TOOL PATTERN with Agent-Level KB")
    print("="*60)
    success_multi = await test_agent_kb_loading("knowledge_base_as_tool_agent.yaml")

    return success_single and success_multi

if __name__ == "__main__":
    success = asyncio.run(test_both_patterns())
    sys.exit(0 if success else 1)
