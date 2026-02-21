import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_kb_tool_agent():
    """Run an agent with Knowledge Base as a Tool."""
    print("\n" + "=" * 60)
    print("Example: Agent with Knowledge Base as Tool")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'knowledge_base_as_tool_agent.yaml')
    
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

    # Test 1: RAG Query via Tool
    print("\n--- Test 1: Knowledge Base Tool Query ---")
    query = "What does the policy say about vacation days?"
    result = await agent.ainvoke(query)
    pprint(result['content'][0]['text'])

if __name__ == "__main__":
    asyncio.run(run_kb_tool_agent())