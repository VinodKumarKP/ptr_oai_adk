"""Knowledge Graph as a TOOL demo (Neo4j / GraphRAG, read-only).

Same prerequisites as knowledge_graph_agent_demo.py:
  1. pip install 'oai-agent-core[neo4j]'
  2. A running Neo4j (>= 5.x) already loaded with the graph
     (see knowledge_graph_seed.cypher) and the `entityNames` full-text index.

Difference: here the knowledge graph is registered as a *tool* on the agent
(nested under agent_list in the config), so the agent chooses when to call it
rather than every turn being routed through it.
"""

import asyncio
import os
from pprint import pprint

import yaml
from oai_agent_core.aws_strands_core.agents.aws_strands_agent import StrandsAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')


async def run_knowledge_graph_tool_agent():
    """Run an agent that uses a Neo4j knowledge graph as a tool."""
    print("\n" + "=" * 60)
    print("Example: Agent with Knowledge Graph as Tool")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'knowledge_graph_as_tool_agent.yaml')

    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = StrandsAgent(
        agent_name="graph_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: Question that should make the agent call the knowledge-graph tool
    print("\n--- Test 1: Knowledge Graph Tool Query ---")
    # query = "Which customer holds the Home Shield policy, and what coverages does it include?"
    query = "which customers filed fire claims?"
    result = await agent.ainvoke(query)
    pprint(result['content']['text'])


if __name__ == "__main__":
    asyncio.run(run_knowledge_graph_tool_agent())
