import asyncio
import os

from pydantic import HttpUrl

from oai_agent_client.agent_client import AgentClient, ClientConfig

async def main():
    """
    Demonstrates the use of AgentClient as an asynchronous context manager
    to ensure that resources are properly managed.
    """
    # Configuration for the agent server
    server_url = os.environ.get("AGENT_URL", "http://localhost:8903")
    api_token = os.environ.get("AGENT_API_TOKEN", "your_default_token_here")

    server_config = {
        "name": "Agent",
        "url": "http://localhost:8903",
        # "port": "8903",
        # "host": "localhost"
    }

    config = ClientConfig(
        url=HttpUrl(server_url),
        timeout=300
    )

    config = ClientConfig(
        command="portfolio-agent-server",
        port=8111
        # args=["--port", "8903"]
        # args=["/Users/vinodkumarkp/PycharmProjects/ptr_portfolio_agents/agentic_registry_agents/agents/market_search_agent/server.py", "--port", "8903"]
    )
    # Use 'async with' to manage the client's lifecycle
    print("--- Using AgentClient with async with ---")
    try:
        async with AgentClient(config=config) as client:
            # --- Synchronous (Invoke) Example ---
            print("--- Running Synchronous (Invoke) Example ---")
            invoke_message = "Hello, agent! What can you do?"
            invoke_config = {
                "session_id": "session_12345",
                "user_id": "user_abc"
            }
            response = await client.invoke(invoke_message, config=invoke_config)
            print("Agent Response (Invoke):")
            print(response)
            print("-" * 20)

            # --- Streaming (Stream) Example ---
            print("\n--- Running Streaming (Stream) Example ---")
            stream_message = "Hello, agent! What can you do?"
            stream_config = {
                "session_id": "session_67890",
                "user_id": "user_xyz",
                "verbose": True
            }
            print("Agent Response (Stream):")
            async for chunk in client.stream(stream_message, config=stream_config):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    asyncio.run(main())
