import asyncio
import os
from pprint import pprint
import yaml
from oai_openai_agent_core.agents.openai_agent import OpenAIAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_guardrails_agent():
    """Run an agent with MCP tools."""
    print("\n" + "=" * 60)
    print("Example: Agent with Guard rail")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'guardrails_agent.yaml')
    
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
    query = "I love Apple and Samsung"
    config = {
            'include_input_message': False,
            'include_raw': False,
            'include_original_message': False
        }
    result = await agent.ainvoke(query, config=config)
    # pprint(result['content'][0]['text'])
    pprint(result)

    async for result in agent.astream(query, config=config):
        # pprint(result['content'][0]['text'])
        pprint(result)


if __name__ == "__main__":
    asyncio.run(run_guardrails_agent())
