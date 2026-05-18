import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent
from oai_agent_server.main import AgentHTTPServer as BaseAgentHTTPServer, main as http_main


# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')

async def run_mcp_agent():
    """Run an MCP agent example."""
    print("\n" + "=" * 60)
    print("Example: MCP Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'mcp_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="env_lookup_agent",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    # await agent.initialize()

    # # Test 1: MCP Tool Usage
    # print("\n--- Test 1: Environment Lookup (MCP) ---")
    query = "What is the value of PATH and GITHUB_TOKEN environment variable?"
    # result = await agent.ainvoke(query)
    # pprint(result['content']['text'])
    async for chunk in agent.astream(query, {'verbose': True}):
        pprint(chunk)
    # file_root = os.path.dirname(os.path.abspath(__file__))
    # agent_name = 'env_lookup_agent'
    # server = BaseAgentHTTPServer(
    #     agent_name=agent_name,
    #     config_root=os.path.dirname(os.path.dirname(file_root)),
    #     agent=agent
    # )
    #
    # # Start the HTTP server
    # http_main(server)
    # return server

if __name__ == "__main__":
    asyncio.run(run_mcp_agent())
    # run_mcp_agent()