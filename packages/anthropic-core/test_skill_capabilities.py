#!/usr/bin/env python
"""
Test script to verify skill capability tools loading.

Tests that:
1. Skill capability tools (read_file, write_file, shell_execute) are registered
2. Tools are available in the tool registry
3. Tools work correctly
"""

import asyncio
import logging
import sys
import tempfile
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

async def test_skill_capabilities():
    """Test skill capability tools."""
    from oai_agent_core.anthropic_core.agents.anthropic_agent import AnthropicAgent
    import yaml

    print("\n" + "="*60)
    print("SKILL CAPABILITY TOOLS TEST")
    print("="*60)

    # For this test, we'll test the individual tools directly
    from oai_agent_core.anthropic_core.builders.agent_builder import (
        read_file, write_file, shell_execute
    )

    print("\n📝 Testing read_file tool...")
    # Create a temporary file to read
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name
        f.write("Hello, this is test content!")

    try:
        result = read_file(temp_file)
        print(f"✓ read_file('{temp_file}') returned:")
        print(f"  {result[:100]}...")
        assert "Hello" in result, "File content not read correctly"
    finally:
        Path(temp_file).unlink()

    print("\n📝 Testing write_file tool...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        temp_file = f.name

    try:
        result = write_file(temp_file, "Test content written by agent")
        print(f"✓ write_file result: {result}")
        assert "Successfully wrote" in result, "Write failed"

        # Verify file was written
        with open(temp_file) as f:
            content = f.read()
        assert "Test content" in content, "File content not verified"
        print(f"✓ File content verified")
    finally:
        Path(temp_file).unlink()

    print("\n💻 Testing shell_execute tool...")
    if sys.platform == 'win32':
        result = shell_execute("echo Hello from shell")
        print(f"✓ shell_execute on Windows: {result}")
    else:
        result = shell_execute("echo 'Hello from shell'")
        print(f"✓ shell_execute on Unix: {result}")
        assert "Hello" in result, "Shell command failed"

    print("\n🔧 Testing error cases...")
    # Test reading non-existent file
    result = read_file("/nonexistent/file.txt")
    print(f"✓ read_file non-existent: {result}")
    assert "not found" in result.lower() or "error" in result.lower()

    print("\n" + "="*60)
    print("✅ All skill capability tool tests passed!")
    print("="*60)

    # Now test integration with agent initialization
    print("\n" + "="*60)
    print("SKILL CAPABILITY AGENT INITIALIZATION TEST")
    print("="*60)

    config_path = Path(__file__).parent / "examples" / "agents_config" / "skills_enabled_agent.yaml"

    if config_path.exists():
        print(f"✓ Config file found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        try:
            agent = AnthropicAgent(
                agent_name="test",
                agent_config=config,
                config_root=str(Path(__file__).parent)
            )
            await agent.initialize()

            print("\n📋 Tool Registry Contents:")
            skill_tools = {k: v for k, v in agent.tool_registry.custom_tools.items()
                          if k in ['read_file', 'write_file', 'shell_execute']}

            if skill_tools:
                print(f"✓ Found {len(skill_tools)} skill capability tool(s):")
                for tool_name in skill_tools.keys():
                    print(f"  - {tool_name}")
            else:
                print("❌ No skill capability tools found!")
                return False

            # Check MCP server
            mcp_servers = agent.tool_registry.get_mcp_server_configs()
            if "custom_tools" in mcp_servers:
                print(f"\n✓ Custom tools MCP server registered")
            else:
                print(f"\n❌ Custom tools MCP server not found")
                return False

            print("\n" + "="*60)
            print("✅ Skill capability agent initialization successful!")
            print("="*60)

            return True

        except Exception as e:
            print(f"\n❌ Agent initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("Skipping agent initialization test")
        return True


if __name__ == "__main__":
    success = asyncio.run(test_skill_capabilities())
    sys.exit(0 if success else 1)
