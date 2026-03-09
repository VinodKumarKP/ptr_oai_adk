import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.aws_strands_core.agents.aws_strands_agent import StrandsAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_mcp_agent():
    """Run an MCP agent example."""
    print("\n" + "=" * 60)
    print("Example: MCP Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'mcp_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = StrandsAgent(
        agent_name="env_lookup_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: MCP Tool Usage
    print("\n--- Test 1: Environment Lookup (MCP) ---")
    query = "What is the value of PATH environment variable?"
    query = 'get input parameter schema of tools in environment_lookup'
    query = 'Fetch all the environment variables'
    query = 'Fetch all the environment variables and Search for flight from Boston to JFK for 2026-03-09 and display the available flights'

    result = await agent.ainvoke(query)
    pprint(result)
    # pprint(result['content']['text'])

if __name__ == "__main__":
    asyncio.run(run_mcp_agent())