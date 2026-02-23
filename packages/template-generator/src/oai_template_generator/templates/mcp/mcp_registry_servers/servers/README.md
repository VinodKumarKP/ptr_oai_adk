# EMAIL MCP Server

A robust Model Context Protocol (MCP) server for secure, audit-friendly email dispatch and validation. This server empowers AI assistants and automation agents to send, validate, and monitor emails with domain enforcement, encrypted API key handling, and structured payload verification

## Features

### Email Dispatch & Management
- **Send emails** via Mailgun with support for plain text and HTML content
- **Support for multiple recipients** including CC with domain validation
- **Secure API key decryption** using Fernet encryption and environment isolation
- **Persistent HTTPS session** with retry logic and timeout configuration

### Email Validation
- **Strict domain enforcement** for sender and recipient addresses (@mail.orchestrateai.tech, @capgemini.com)
- **Pydantic-based schema validation**  for email payloads with descriptive error handling
- **Support for single or multiple recipients** with automatic formatting for Mailgun compatibility

### Server Utilities
- **Server status check** via structured payload inspection
- **Detailed logging** of email dispatch attempts, failures, and exceptions
- **Modular design** for integration into orchestration flows or multi-agent systems

## Installation

### Option 1: Direct UV Run (Recommended)
Use uv to run the server directly without local installation:

```json
{
  "git-server": {
    "command": "uv",
    "args": [
      "run",
      "--with",
      "git+https://github.com/Capgemini-Innersource/ptr_mcp_servers_registry.git",
      "git-mcp-server"
    ]
  }
}
```

### Option 2: Pip Install + Run
Install the package first, then run the server:

```bash
pip install git+https://github.com/Capgemini-Innersource/ptr_mcp_servers_registry.git
```

Then configure your MCP client:
```json
{
  "git-server": {
    "command": "git-mcp-server",
    "args": []
  }
}
```

### Option 3: Local Clone + UV Run
Clone the repository locally and run with uv:

```bash
git clone https://github.com/Capgemini-Innersource/ptr_mcp_servers_registry.git
```

Then configure your MCP client:
```json
{
  "git-server": {
    "command": "uv",
    "args": [
      "run",
      "--directory",
      "/path/to/ptr_mcp_servers_registry",
      "git-mcp-server"
    ]
  }
}
```

### Option 4: Local Docker Compose Setup

You can run the MCP server registry locally using Docker Compose. This is ideal for development and testing purposes.

#### Prerequisites
- Docker and Docker Compose installed
- Make utility installed
- `jq` command-line JSON processor

#### Setup Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Capgemini-Innersource/ptr_mcp_servers_registry.git
   cd ptr_mcp_servers_registry
   ```

2. **Build and start all services:**
   ```bash
   make start-all
   ```

3. **Or start a specific service:**
   ```bash
   make start-git_server
   ```

4. **List available services:**
   ```bash
   make list-services
   ```

5. **Configure your client to use the local server:**
   In your client configuration (e.g., Claude Desktop), add the server with the local URL:
   ```json
   {
     "git-server": {
       "command":"npx",
    	"args":["mcp-remote@latest","http://localhost:8000/mcp", "--allow-http"]
     }
   }
   ```

#### Available Make Commands

- `make start-all` - Start all services using Docker Compose
- `make stop-all` - Stop all services
- `make start-<service>` - Start a specific service
- `make stop-<service>` - Stop a specific service
- `make restart-<service>` - Restart a specific service
- `make list-services` - List all available services
- `make docker-build` - Build the Docker image
- `make generate-compose` - Generate Docker Compose file

### Option 5: Remote Server Deployment

For production use or when you want to share the MCP server with multiple users, you can deploy it to a remote server.

#### Prerequisites
- Remote server with Docker and Docker Compose installed
- SSH access to the remote server
- Domain name or public IP address

#### Deployment Steps

1. **Deploy to your remote server:**
   ```bash
   # SSH into your remote server
   ssh user@your-remote-server.com
   
   # Clone the repository
   git clone https://github.com/Capgemini-Innersource/ptr_mcp_servers_registry.git
   cd ptr_mcp_servers_registry
   
   # Build and start services
   make start-all
   ```

2. **Configure firewall (if needed):**
   ```bash
   # Allow traffic on port 8000
   # Check the servers_config/git_server.json for the list of ports to be opened
   sudo ufw allow 8000
   ```

3. **Configure your client to use the remote server:**
   In your client configuration, use the remote server URL:
   ```json
   {
     "git-server": {
       "command":"npx",
    	"args":["mcp-remote@latest","http://remote-ip:8000/mcp", "--allow-http"]
     }
   }
   ```

## Configuration

### Claude Desktop
Add to your `claude_desktop_config.json`:

Using uv command

```json
{
  "mcpServers": {
    "git-server": {
      "command": "uv",
      "args": [
        "run",
        "--with",
        "git+https://github.com/Capgemini-Innersource/ptr_mcp_servers_registry.git",
        "git-mcp-server"
      ]
    }
  }
}
```

Using npx command and remote server

```json
{
  "mcpServers": {
    "git-server": {
      "command":"npx",
    	"args":["mcp-remote@latest","http://<<remote-ip>>:8000/mcp", "--allow-http"]
    }
  }
}
```

### Other MCP Clients
The server follows the standard MCP protocol and can be integrated with any MCP-compatible client using the installation methods above.

## Available Tools

### Email Dispatch

#### `send_email`
Send an email using the Mailgun API with support for plain text and HTML content.
- **Parameters:**
  - `to` (string or list): Recipient email address(es)
  - `subject` (string): Subject of the email
  - `text` (string, optional): Plain text content
  - `html` (string, optional): HTML content
  - `from_addr` (string): Sender email address (must be @mail.orchestrateai.tech)
  - `cc` (string or list, optional): CC recipients

- **Returns:** Status message indicating success or failure

### Server Status

#### `get_server_status`
Validate and inspect the email payload without sending it. Useful for dry runs and debugging.
- **Parameters:**
  - `to` (string or list): Recipient email address(es)
  - `subject` (string): Subject of the email
  - `text` (string, optional): Plain text content
  - `html` (string, optional): HTML content
  - `from_addr` (string): Sender email address
  - `cc` (string or list, optional): CC recipients

- **Returns:** Dictionary with validated email field


## Usage Examples

### Email Dispatch
```python
# Send an Email
response = send_email(
    to=["user@capgemini.com"],
    subject="MCP Server Update",
    text="The server is now live.",
    html="<p>The server is now <strong>live</strong>.</p>",
    from_addr="noreply@mail.orchestrateai.tech",
    cc=["manager@capgemini.com"]
)
```

### Server Status
```python
# Check Server Status
status = get_server_status(
    to="user@capgemini.com",
    subject="Dry Run",
    text="Testing payload validation.",
    html="<p>Testing <em>payload</em> validation.</p>",
    from_addr="noreply@mail.orchestrateai.tech"
)
```

## Error Handling
The server includes robust validation and logging for:
- Invalid email formats
- Unauthorized domains
- Missing required fields
- API key decryption failures
- Mailgun response errors
All errors are logged with context and returned as descriptive messages for debugging and audit purposes.


## Requirements

- Python 3.11+
- Recommended: Use a virtual environment for isolation 

## License

This MCP server is part of the Capgemini Innersource MCP Servers Registry. Please refer to the repository for licensing information.

## Contributing

This server is maintained as part of the larger MCP servers registry. For issues, feature requests, or contributions, please visit the [main repository](https://github.com/Capgemini-Innersource/ptr_mcp_servers_registry).
