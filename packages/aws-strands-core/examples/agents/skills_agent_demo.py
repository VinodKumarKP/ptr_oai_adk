import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.aws_strands_core.agents.aws_strands_agent import StrandsAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_simple_agent():

    """Run a simple agent example."""
    print("\n" + "=" * 60)
    print("Example: Simple Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'skills_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = StrandsAgent(
        agent_name="simple_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: General query
    print("\n--- Test 1: General Query ---")
    # result = await agent.ainvoke('Read the csv file /Users/vinodkumarkp/PycharmProjects/strands_demo/data.csv and analyze the data? ')
    # result = await agent.ainvoke(
        # 'Get the transcript for this video https://www.youtube.com/watch?v=0QzopZ78w9M')
    result = await agent.ainvoke('hi, what are the available tools')
    pprint(result['content']['text'])
    pprint(result)

if __name__ == "__main__":
    asyncio.run(run_simple_agent())