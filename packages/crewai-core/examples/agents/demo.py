import asyncio
import os
import yaml
from typing import cast

from oai_agent_core.crewai_core.agents.crewai_agent import CrewAIAgent
from crewai import Crew

# Conditional import for HTTP server
# try:
#     from oai_agent_server.core.agent_http import AgentHTTPServer as BaseAgentHTTPServer, main as http_main
#     HTTP_SERVER_AVAILABLE = True
# except ImportError:
#     BaseAgentHTTPServer = None
#     http_main = None
#     HTTP_SERVER_AVAILABLE = False

from oai_agent_server.main import AgentHTTPServer as BaseAgentHTTPServer, main as http_main

file_root = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(file_root), 'agents_config', 'crewai_rag_agent.yaml')
# config_path = os.path.join(os.path.dirname(file_root), 'agents_config', 'random_generator_agent.yml')
# config_path = os.path.join(os.path.dirname(file_root), 'agents_config', 'random_generator_flow.yml')
# config_path = os.path.join(os.path.dirname(file_root), 'agents_config', 'researcher_crewai_agent_task.yml')
# config_path = os.path.join(os.path.dirname(file_root), 'agents_config', 'search_agent.yaml')

with open(config_path) as f:
    config = yaml.safe_load(f)


def main():
    # if not HTTP_SERVER_AVAILABLE:
    #     print("HTTP server not available. Running agent directly...")
    #     return run_agent_directly()
    
    agent_name = "crewai_agent"
    agent = CrewAIAgent(agent_name="crewai_agent", agent_config=config, config_root=(os.path.dirname(file_root)))

    # crew = cast(Crew, await agent.initialize())


    # Initialize the HTTP server with your custom agent
    server = BaseAgentHTTPServer(
        agent_name=agent_name,
        config_root=os.path.dirname(os.path.dirname(file_root)),
        agent=agent
    )

    # Start the HTTP server
    http_main(server)
    return server

def run_agent_directly():
    """Run the agent directly without HTTP server"""
    crewai_agent = CrewAIAgent(agent_name="crewai_agent", agent_config=config)
    print("CrewAI Agent created successfully")
    print("To test the agent, use: await crewai_agent.ainvoke('your message')")
    return crewai_agent

if __name__ == "__main__":
    main()
