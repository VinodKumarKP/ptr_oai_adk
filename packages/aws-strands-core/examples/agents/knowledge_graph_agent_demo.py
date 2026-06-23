"""Knowledge Graph (Neo4j / GraphRAG) agent demo.

Prerequisites
-------------
1. Install the optional Neo4j extra::

       pip install 'oai-agent-core[neo4j]'

2. A running Neo4j (>= 5.x) that is **already loaded** with your graph
   (this store is read-only and does not ingest data).

3. A full-text index over the node properties you want to use as entry points,
   matching ``fulltext_index`` in ``knowledge_graph_agent.yaml``, e.g.::

       CREATE FULLTEXT INDEX entityNames IF NOT EXISTS
       FOR (n:Policy|Coverage|Customer|Claim)
       ON EACH [n.name, n.title, n.description];

4. Export the DB password referenced by the config::

       export NEO4J_PASSWORD=your-password
"""

import asyncio
import os
from pprint import pprint

import yaml
from oai_agent_core.aws_strands_core.agents.aws_strands_agent import StrandsAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')


async def run_knowledge_graph_agent():
    """Run a Neo4j knowledge-graph (GraphRAG) agent example."""
    print("\n" + "=" * 60)
    print("Example: Knowledge Graph Agent (Neo4j / GraphRAG)")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'knowledge_graph_agent.yaml')

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

    # Test 1: Traversal query — entry node via full-text index, then N-hop expansion
    print("\n--- Test 1: Knowledge Graph Query ---")
    query = "Is flood damage covered by the insurance ? What about fire, hail and theft damage ?"
    result = await agent.ainvoke(query)
    pprint(result['content']['text'])


if __name__ == "__main__":
    asyncio.run(run_knowledge_graph_agent())
