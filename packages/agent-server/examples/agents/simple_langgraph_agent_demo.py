import asyncio
import os
from pprint import pprint
import yaml
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent
from oai_agent_server.main import AgentHTTPServer as BaseAgentHTTPServer, main as http_main

# Get the absolute path to the examples directory
EXAMPLES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(EXAMPLES_DIR, 'agents_config')
file_root = os.path.dirname(os.path.abspath(__file__))

os.environ['DB_LOGGING_ENABLED'] = 'true'

def run_simple_agent():
    """Run a simple agent example."""
    print("\n" + "=" * 60)
    print("Example: Simple Agent")
    print("=" * 60)

    config_path = os.path.join(CONFIG_DIR, 'travel_agent.yaml')
    
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Initialize agent
    agent = LangGraphAgent(
        agent_name="travel_agent",
        agent_config=config,
        config_root=EXAMPLES_DIR
    )

    server = BaseAgentHTTPServer(
        agent_name=agent.agent_name,
        config_root=os.path.dirname(os.path.dirname(file_root)),
        agent=agent
    )

    # Start the HTTP server
    http_main(server)
    return server

if __name__ == "__main__":
    run_simple_agent()