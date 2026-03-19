import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.openai_core.agents.openai_agent import OpenAIAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_agent_as_tool_demo():
    """Run a multi-agent system where sub-agents are tools."""
    print("\n" + "=" * 60)
    print("Example: Multi-Agent (Agent as Tool)")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'multi_agent_agent_as_tool.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = OpenAIAgent(
        agent_name="supervisor",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: Complex query requiring sub-agent tool
    print("\n--- Test 1: Flight Booking via Tool ---")
    query = """
    Search for flight from Boston to JFK for 2026-03-19 and book the cheapest one without user confirmation using userid 1
    Provide the summary of the itinerary
    """
    result = await agent.ainvoke(query)
    pprint(result['content']['text'])

if __name__ == "__main__":
    asyncio.run(run_agent_as_tool_demo())
