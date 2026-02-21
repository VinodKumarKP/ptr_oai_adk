import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_simple_agent():
    """Run a simple agent example."""
    print("\n" + "=" * 60)
    print("Example: Simple Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'simple_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="simple_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: General query
    print("\n--- Test 1: General Query ---")
    result = await agent.ainvoke('Hello! What can you do?')
    pprint(result['content'][0]['text'])

if __name__ == "__main__":
    asyncio.run(run_simple_agent())