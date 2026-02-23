#!/bin/sh
if [ -z "AGENT_NAME" ]; then
    echo "Error: $AGENT_NAME environment variable is required"
    exit 1
fi

exec python -m agentic_registry_agents.agents.${AGENT_NAME}.server "$@"