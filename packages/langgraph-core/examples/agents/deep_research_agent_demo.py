import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')


async def run_deep_research_agent():
    """Run a deep research agent example.

    Demonstrates the deepagents harness driven from YAML:
    - agent_list entries become subagents (market_data_analyst uses MCP
      tools, news_researcher uses registry tools)
    - the root deep agent is configured by the root-level system_prompt and
      the crew_config attributes (pattern: deep, root tools)
    """
    print("\n" + "=" * 60)
    print("Example: Deep Research Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'deep_research_agent.yaml')

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="deep_research_agent",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    query = '''
    'Research AAPL: get the current quote and company fundamentals, '
        'check recent news sentiment, and give me a short outlook.'
    '''

    print("\n--- Test 1: Multi-step research task ---")
    # result = await agent.ainvoke(
    #     'Research AAPL: get the current quote and company fundamentals, '
    #     'check recent news sentiment, and give me a short outlook.'
    # )
    # pprint(result['content']['text'])


    async for chunk in agent.astream(query, {'verbose': True, 'stream_mode':"values", 'subgraphs': True}):
        # response = json.loads(chunk['content']['text'])
        # pprint(response)
        print(chunk)


if __name__ == "__main__":
    asyncio.run(run_deep_research_agent())
