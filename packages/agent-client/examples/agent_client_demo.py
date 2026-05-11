import asyncio
import os

from pydantic import HttpUrl

from oai_agent_client import AsyncAgentClient, ClientConfig


async def main():
    """
    Demonstrates the use of AgentClient as an asynchronous context manager
    to ensure that resources are properly managed.

    Set ``AGENT_URL`` to point at a running agent server, e.g.::

        AGENT_URL=http://localhost:8903 python agent_client_demo.py
    """
    server_url = os.environ.get("AGENT_URL", "http://localhost:8015")

    config = ClientConfig(
        url=HttpUrl(server_url),
        timeout=300,
    )

    # Alternative: launch a local server process instead of connecting to a
    # running one. Replace the command below with one available on your PATH.
    #
    # config = ClientConfig(
    #     command="portfolio-agent-server",
    #     args=["--port", "8111"],
    # )

    print("--- Using AsyncAgentClient with async with ---")
    try:
        async with AsyncAgentClient(config=config) as client:
            # --- Synchronous (Invoke) Example ---
            print("--- Running Synchronous (Invoke) Example ---")
            invoke_message = "Hello, agent! What can you do?"
            invoke_config = {
                "session_id": "session_12345",
                "user_id": "user_abc",
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
                "verbose": True,
            }
            print("Agent Response (Stream):")
            async for chunk in client.stream(stream_message, config=stream_config):
                print(chunk, end="", flush=True)
            print("\n" + "-" * 20)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    asyncio.run(main())
