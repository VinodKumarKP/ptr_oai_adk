import os
import sys

file_root = os.path.dirname(os.path.abspath(__file__))
if file_root not in sys.path:
    sys.path.insert(0, file_root)

from oai_agent_server.main import AgentHTTPServer as BaseAgentHTTPServer, main as http_main
from agent  import TravelAgent


def main():
    server = BaseAgentHTTPServer(agent_name=os.path.basename(os.path.dirname(__file__)),
                                 config_root=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                 agent=TravelAgent())
    http_main(server)


if __name__ == "__main__":
    main()
