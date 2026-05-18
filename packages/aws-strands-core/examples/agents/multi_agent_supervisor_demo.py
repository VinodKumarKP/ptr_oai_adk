import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.aws_strands_core.agents.aws_strands_agent import StrandsAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_supervisor_agent():
    """Run a multi-agent supervisor example."""
    print("\n" + "=" * 60)
    print("Example: Multi-Agent Supervisor")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'multi_agent_supervisor.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = StrandsAgent(
        agent_name="supervisor",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: Delegated task
    print("\n--- Test 1: Hotel Search (Delegated) ---")
    query = """
    Search for flight from Boston to JFK for 2026-03-30 and book the cheapest one without user confirmation.
    Also, book hotel in New York without user confirmation.
    Provide a summary of the itinerary
    """
    # result = await agent.ainvoke(query)
    # pprint(result)
    # pprint(result['content']['text'])

    async for chunk in agent.astream(query, {'verbose': True, 'include_raw': False}):
        print("\n")
        pprint(chunk)

if __name__ == "__main__":
    asyncio.run(run_supervisor_agent())