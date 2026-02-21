import asyncio
import os
from pprint import pprint
import yaml
from oai_openai_agent_core.agents.openai_agent import OpenAIAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_supervisor_demo():
    """Run a multi-agent system with a supervisor."""
    print("\n" + "=" * 60)
    print("Example: Multi-Agent Supervisor")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'multi_agent_supervisor.yaml')
    
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

    # Test 1: Delegated task
    print("\n--- Test 1: Hotel Search (Delegated) ---")
    query = "Find a hotel in downtown Chicago."
    result = await agent.ainvoke(query)
    pprint(result['content'][0]['text'])

if __name__ == "__main__":
    asyncio.run(run_supervisor_demo())
