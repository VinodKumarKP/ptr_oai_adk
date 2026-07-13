import asyncio
import json
import os
from pprint import pprint
import yaml
from pip._internal.utils import datetime

from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')


async def run_deep_agent():
    """Run a deep agent example (deepagents harness with subagents)."""
    print("\n" + "=" * 60)
    print("Example: Deep Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'deep_agent.yaml')

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="deep_travel_planner",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    from datetime import datetime, timedelta
    query = f'Search for flight from BOS to JFK New York for {(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")} and Search for hotel in New York for {(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")} and book the cheapest.'

    # async for chunk in agent.agent.astream(
    #         {"messages": [{"role": "user", "content": query}]},
    #         stream_mode="messages",
    #         subgraphs=True,
    #         version="v2",
    #         config={
    #                 "configurable": {
    #                     "thread_id": "1234"
    #                 }}
    # ):
    #     pprint(chunk)

#     stream = agent.agent.stream_events({
#     "messages": [{"role": "user", "content": query}],
# }, version="v3", config={
#         "configurable": {
#             "thread_id": "1234"
#         }
#     })
#
#     for subagent in stream.subagents:
#         print(subagent.name, subagent.path, subagent.status)
#
#         for message in subagent.messages:
#             print(message.text)

    # print("\n--- Test 1: Multi-step planning task ---")
    # query = 'Search for hotel in New York for July 6th 2026 and book the cheapest.'
    # # result = await agent.ainvoke(
    # #     'Search for hotel in New York for July 6th 2026 and book the cheapest.'
    # # )
    # # # result = await agent.ainvoke("Retrieve all the environment variables")
    # # pprint(result['content']['text'])
    # #
    async for chunk in agent.astream(query, {'verbose': True, 'stream_mode':"values", 'subgraphs': False}):
        # response = json.loads(chunk['content']['text'])
        # pprint(response)
        print(chunk)


if __name__ == "__main__":
    asyncio.run(run_deep_agent())
