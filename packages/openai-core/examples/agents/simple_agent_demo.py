import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.openai_core.agents.openai_agent import OpenAIAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_simple_agent():
    """Run a simple single agent example."""
    print("\n" + "=" * 60)
    print("Example: Single Agent with Tools")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'simple_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    # We pass the examples directory as config_root so relative paths in yaml are resolved correctly
    agent = OpenAIAgent(
        agent_name="simple_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: General query
    print("\n--- Test 1: General Query ---")
    result = await agent.ainvoke('Hello! What can you do?')
    pprint(result['content'][0]['text'])

    # Test 2: Tool usage (Flight Search)
    print("\n--- Test 2: Tool Usage (Flight Search) ---")
    query = "Find me a flight from BOS to JFK for 2026-01-31 and find hotels in New York"
    result = await agent.ainvoke(query)
    pprint(result['content'][0]['text'])

    # Test 3: Streaming
    print("\n--- Test 3: Streaming Response ---")
    print("User: Find hotels in New York.")
    print("Agent: ", end="", flush=True)
    async for chunk in agent.astream('Find hotels in New York.'):
        pprint(chunk)
        # if chunk.get('type') == 'text':
        #     print(chunk['content'], end="", flush=True)
        # elif chunk.get('type') == 'tool_call_item':
        #      print(f"\n[Tool Call: {chunk['content']}]", end="", flush=True)
    print("\n")

if __name__ == "__main__":
    asyncio.run(run_simple_agent())
