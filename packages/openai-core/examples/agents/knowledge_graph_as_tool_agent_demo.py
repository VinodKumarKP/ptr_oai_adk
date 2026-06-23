"""Knowledge Graph as a TOOL demo (Neo4j / GraphRAG, read-only).

Same prerequisites as knowledge_graph_agent_demo.py. Difference: the knowledge
graph is registered as a *tool* on the agent (nested under agent_list in the
config), so the agent chooses when to call it.
"""

import asyncio
import os
from pprint import pprint

import yaml
from oai_agent_core.openai_core.agents.openai_agent import OpenAIAgent

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')


async def run_knowledge_graph_tool_agent():
    """Run an agent that uses a Neo4j knowledge graph as a tool."""
    print("\n" + "=" * 60)
    print("Example: Agent with Knowledge Graph as Tool")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'knowledge_graph_as_tool_agent.yaml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = OpenAIAgent(
        agent_name="graph_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    print("\n--- Test 1: Knowledge Graph Tool Query ---")
    query = "Which customers filed fire claims?"
    result = await agent.ainvoke(query)
    pprint(result['content'][0]['text'])


if __name__ == "__main__":
    asyncio.run(run_knowledge_graph_tool_agent())
