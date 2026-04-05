# OAI Agent Registry

The OAI Agent Registry is a FastAPI-based application that acts as a central registry for OAI agents. It provides a way to route requests to multiple agent containers.

## Features

- **Agent Registration**: Agents can be registered with the registry.
- **Request Routing**: The registry can route requests to the appropriate agent based on the request path.
- **Asynchronous**: The registry is built on top of FastAPI and is fully asynchronous.

## Usage

To start the agent registry, run the following command:

```bash
python -m oai_agent_registry.cli
```

This will start the registry on `localhost:8000`. You can then send requests to the registry, which will be forwarded to the appropriate agent.
