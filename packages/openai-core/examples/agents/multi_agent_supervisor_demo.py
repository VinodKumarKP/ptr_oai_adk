import asyncio
import json
import os
from pprint import pprint
import yaml
from oai_agent_core.openai_core.agents.openai_agent import OpenAIAgent

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
    query = """
    Search for flight from Boston to JFK for 2026-03-20 and book the cheapest one without user confirmation using userid 1
    Provide the summary of the itinerary
    """
    result = await agent.ainvoke(query, config= {
        'session_id': 123,
        'user_id': 'test_user'
    })
    pprint(result)
    text = result['content']['text']
    text = text.replace('\n', '').replace('`', '').replace('json', '')
    response = json.loads(text)

    print(response['departure_airport'])

    # query = 'Provide the summary of the itinerary'
    # result = await agent.ainvoke(query, config={
    #     'session_id': 123,
    #     'user_id': 'test_user'
    # })
    # pprint(result)
    # pprint(result['content']['text'])

if __name__ == "__main__":
    asyncio.run(run_supervisor_demo())
