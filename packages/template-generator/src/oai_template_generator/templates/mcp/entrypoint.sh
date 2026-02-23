#!/bin/sh
if [ -z "$MCP_SERVER_NAME" ]; then
    echo "Error: MCP_SERVER_NAME environment variable is required"
    exit 1
fi
exec python -m mcp_registry_servers.servers.${MCP_SERVER_NAME}.server "$@"