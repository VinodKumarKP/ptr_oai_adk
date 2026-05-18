import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.openai_core.agents.openai_agent import OpenAIAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_mcp_agent():
    """Run an agent with MCP tools."""
    print("\n" + "=" * 60)
    print("Example: Agent with MCP Tools")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'mcp_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = OpenAIAgent(
        agent_name="env_lookup_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: MCP Tool Usage
    print("\n--- Test 1: Environment Lookup (MCP) ---")
    # Note: This query depends on the specific MCP server capabilities
    # query = 'list all the available tools and input schema'
    # query = 'hi'
    query = "What is the value of PATH and GITHUB_TOKEN environment variable?"
    # query = '''
    # Search for flight from Boston to JFK for 2026-03-09 and display the available flights
    # and book McKittrick Hotel hotel in New York. After booking again search for flight from Boston to JFK for 2026-03-09 and display the available flights
    # and book McKittrick Hotel hotel in New York. Also retrieve all the environment variables.
    # '''
    # result = await agent.ainvoke(query)
    # # pprint(result['content']['text'])
    # pprint(result)
    async for chunk in agent.astream(query, {'verbose': True}):
        pprint(chunk)

if __name__ == "__main__":
    asyncio.run(run_mcp_agent())
