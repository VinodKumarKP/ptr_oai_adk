import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_kb_agent():
    """Run a knowledge base agent example."""
    print("\n" + "=" * 60)
    print("Example: Knowledge Base Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'knowledge_base_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="rag_assistant",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    await agent.initialize()

    # Test 1: Knowledge Base Query
    print("\n--- Test 1: Knowledge Base Query ---")
    # query = "What is the policy on remote work?"
    # query = "Is architect fees covered ?"
    query = "which file contains flask"
    result = await agent.ainvoke(query)
    pprint(result['content']['text'])

if __name__ == "__main__":
    asyncio.run(run_kb_agent())