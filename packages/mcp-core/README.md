# MCP Server Core (oai_mcp_server_core)

The `oai_mcp_server_core` library provides the foundational framework for building Model Context Protocol (MCP) servers. It extends `FastMCP` with enterprise capabilities including request isolation, authentication, and standardized configuration.

## Table of Contents

- [MCP Server Core (oai_mcp_server_core)](#mcp-server-core-oai_mcp_server_core)
  - [Core Features](#core-features)
    - [🛡️ Request Isolation](#️-request-isolation)
    - [🔐 Authentication](#-authentication)
    - [⚙️ Configuration & Tooling](#️-configuration--tooling)
- [MCP Server Development Quick Start](#mcp-server-development-quick-start)
  - [Prerequisites](#prerequisites)
  - [Setup](#setup)
    - [1. Create Repository](#1-create-repository)
    - [2. Environment Setup](#2-environment-setup)
  - [Development](#development)
    - [3. Create Your Tools Class](#3-create-your-tools-class)
    - [4. Create Server Config](#4-create-server-config)
    - [5. Implement Server](#5-implement-server)
    - [6. ⚠️ CRITICAL: Update Dependencies](#6-️-critical-update-dependencies)
  - [Testing (5 minutes)](#testing-5-minutes)
    - [Local Testing](#local-testing)
    - [Test with MCP Inspector](#test-with-mcp-inspector)
    - [Docker Testing (Optional)](#docker-testing-optional)
  - [Validation Checklist](#validation-checklist)
  - [Project Structure](#project-structure)
  - [Quick Troubleshooting](#quick-troubleshooting)
  - [Next Steps](#next-steps)

## Core Features

### 🛡️ Request Isolation
Ensures thread-safe handling of concurrent requests by isolating environment variables.
- **Context-Aware Environment**: `os.environ` is patched to be request-aware, ensuring that headers and environment variables from one request do not leak into others.
- **Header Capture**: Middleware automatically captures HTTP headers (e.g., `Authorization`, `X-Custom-Header`) and exposes them as environment variables for the request scope.

### 🔐 Authentication
- **Token Management**: Redis-based token validation with support for permanent and TTL (Time-To-Live) tokens.
- **Middleware**: `AuthenticationMiddleware` enforces security when `AUTH_ENABLED` is set to `true`.
- **Metadata**: Tokens can carry user and role information.

### ⚙️ Configuration & Tooling
- **Unified Config**: Load configurations from YAML/JSON files.
- **Logging**: Standardized rotating file and console logging via `get_logger()`.
- **Health Checks**: Built-in `/health` and `/info` endpoints for monitoring.

---

# MCP Server Development Quick Start

A streamlined guide for developers to quickly build and test MCP servers.

## Prerequisites

- Python 3.11+
- Node.js 18+ (for testing) or
- Postman or similar API client (for testing)
- Redis (Optional, required if `AUTH_ENABLED=true`)
- Git configured with GitHub token for Capgemini-Innersource
- Docker (optional, for containerized testing)

## Setup

### 1. Create Repository
```bash
# Create from template: ptr_mcp_server_template
# Name format: ptr-mcp-{server_name}

# Clone your new repo
git clone {your_repo_url}
cd {your_repo}
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
cd mcp_server_registry
```

## Development

### 3. Create Your Tools Class
Create `mcp_servers_registry/utils/my_server_tools.py`:

```python
class MyServerTools:
    """Your server functionality."""
    
    def process_data(self, input: str) -> dict:
        """
        Process input data.
        
        Args:
            input (str): Data to process
            
        Returns:
            dict: Processing results
        """
        return {"result": f"Processed: {input}"}
    
    def _helper_method(self):
        """Private method - won't be exposed as tool."""
        pass
```

**Key Points:**
- Public methods = MCP tools (automatically exposed)
- Private methods (prefix `_`) = not exposed
- Must include type hints and docstrings

### 4. Create Server Config
Create `mcp_servers_registry/servers_config/{{mcp_server_name}}.yaml`:

```yaml
port: 8001  # Choose unique port > 8000
description: |-
  Brief description of what your server does.
  Explain key functionality and use cases.
tags:
  - relevant_tag1
  - relevant_tag2
environment:
  - KEY1:value1
  - KEY2:value2
```

### 5. Implement Server
```bash
# Create server directory
cd mcp_servers_registry/servers/
mkdir {{mcp_server_name}}
touch {{mcp_server_name}}/__init__.py
touch {{mcp_server_name}}/README.md
```

Create `{{mcp_server_name}}/server.py`:

```python
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

file_root = os.path.dirname(os.path.abspath(__file__))
path_list = [file_root, os.path.dirname(file_root), os.path.dirname(os.path.dirname(file_root))]
for path in path_list:
    if path not in sys.path:
        sys.path.append(path)

import nest_asyncio
nest_asyncio.apply()

from mcp_registry_servers.tools.my_server_tools import MyServerTools
from oai_mcp_server_core.core.base_mcp_server import BaseMCPServer
from oai_mcp_server_core.utils.logger_utils import get_logger

logger = get_logger()

class {{MCPServerClassName}}(BaseMCPServer):
    """My MCP server implementation."""
    
    def __init__(self):
        super().__init__(self.base_directory(__file__), 
                         object_list=[MyServerTools()],
                         source_file=__file__)

def main():
    server = {{MCPServerClassName}}()
    server.main()

if __name__ == "__main__":
    main()
```

### 6. ⚠️ CRITICAL: Update Dependencies

**Update `pyproject.toml`:**
```toml
[project]
name = "mcp_registry_servers_{category}"  # Change {category}
version = "1.0.0"

dependencies = [
    "mcp>=1.0.0",
    "python-dotenv>=1.0.0",
    # Add your new dependencies here
]
```

**Update `requirements.txt`:**
```txt
mcp>=1.0.0
python-dotenv>=1.0.0
# Add your new dependencies here
```

## Testing (5 minutes)

### Local Testing

```bash
# From project root with venv activated
make install

# Start server
python mcp_servers_registry/servers/{{mcp_server_name}}/server.py --transport streamable-http

# Should see: "INFO: Server starting on port {port_number}"
```

### Test with MCP Inspector

```bash
# Install and launch
npx @modelcontextprotocol/inspector
```

**Inspector Configuration:**
- Server URL: `http://localhost:{port_number}/mcp`
- Transport: `Streamable HTTP`

**Test Your Tools:**
1. Select your tool from the list
2. Enter test inputs
3. Verify outputs match expectations
4. Test edge cases and error handling

### Docker Testing (Optional)

```bash
# Set GitHub token
export GITHUB_TOKEN={your_token}

# Build and start
make start-my_server

# Verify
docker ps
```

## Validation Checklist

- [ ] Server starts without errors
- [ ] All tools visible in MCP Inspector
- [ ] Tools execute with valid inputs
- [ ] Error handling works correctly
- [ ] Documentation complete

## Project Structure

```
mcp_servers_registry/
├── servers_config/
│   └── my_server.yaml
├── servers/
│   └── {{mcp_server_name}}/
│       ├── __init__.py
│       ├── README.md
│       └── server.py
├── utils/
│   └── my_server_tools.py
└── tests/
    └── test_my_server_tools.py
```

## Quick Troubleshooting

| Problem | Fix |
|---------|-----|
| Import errors | Check file paths match config |
| Server won't start | Verify dependencies installed |
| Tools not showing | Ensure methods don't start with `_` |
| Port conflict | Change port in YAML config |

## Next Steps

After testing locally:
1. Complete comprehensive testing
2. Submit onboarding request
3. Request PR review from admins
4. Deploy to production

---

**Need Help?** Check the full [Onboarding Instructions](topics/Onboarding_Instructions_MCP.md) for detailed guidance.
