"""Knowledge Graph (Neo4j / GraphRAG) agent demo.

Prerequisites
-------------
1. pip install 'oai-agent-core[neo4j]'
2. A running Neo4j (>= 5.x) already loaded with the graph (this store is
   read-only and does not ingest). See knowledge_graph_seed.cypher.
3. A full-text index matching `fulltext_index` in the config, e.g.::

       CREATE FULLTEXT INDEX entityNames IF NOT EXISTS
       FOR (n:Policy|Coverage|Customer|Claim)
       ON EACH [n.name, n.description];
"""

import asyncio
import os
from pprint import pprint

import yaml
from oai_agent_core.anthropic_core.agents.anthropic_agent import AnthropicAgent

EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')


async def run_knowledge_graph_agent():
    """Run a Neo4j knowledge-graph (GraphRAG) agent example."""
    print("\n" + "=" * 60)
    print("Example: Knowledge Graph Agent (Neo4j / GraphRAG)")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'knowledge_graph_agent.yaml')

    with open(config_path) as f:
        config = yaml.safe_load(f)

    agent = AnthropicAgent(
        agent_name="graph_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    print("\n--- Test 1: Knowledge Graph Query ---")
    query = "Is flood damage covered? What about fire and theft?"
    result = await agent.ainvoke(query)
    pprint(result['content']['text'])


if __name__ == "__main__":
    asyncio.run(run_knowledge_graph_agent())
