import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_guardrails_agent():
    """Run an MCP agent example."""
    print("\n" + "=" * 60)
    print("Example: MCP Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'guardrails_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="env_lookup_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: MCP Tool Usage
    print("\n--- Test 1: Environment Lookup (MCP) ---")
    query = "I love Apple and Samsung"
    config = {
        "include_raw": False,
        "include_input_message": False,
        "include_original_message": False
    }
    result = await agent.ainvoke(query, config=config)
    pprint(result)
    # pprint(result['content']['text'])


    async for result in agent.astream(query, config=config):
        pprint(result)

if __name__ == "__main__":
    asyncio.run(run_guardrails_agent())