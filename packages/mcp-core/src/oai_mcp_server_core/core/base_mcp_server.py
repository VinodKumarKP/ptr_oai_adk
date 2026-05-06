import argparse
import os
import asyncio
import httpx
from abc import ABC
from typing import Literal, List
from urllib.parse import urlparse

from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, JSONResponse

import nest_asyncio

nest_asyncio.apply()

from fastmcp import FastMCP
from oai_mcp_server_core.utils.logger_utils import get_logger
from oai_mcp_server_core.core.context import RequestAwareEnviron, request_env
from oai_mcp_server_core.config.settings import get_server_config, MCPServerConfig
from oai_mcp_server_core.middleware.auth import AuthenticationMiddleware
from oai_mcp_server_core.middleware.capture import HeaderCaptureMiddleware
from oai_mcp_server_core.core.registry import MCPRegistry
from oai_mcp_server_core.core.exceptions import TransportError

VALID_TRANSPORTS = {"stdio", "streamable-http", "sse"}


class BaseMCPServer(ABC):
    """
    Base class for MCP servers with request-isolated environment variables.

    This implementation ensures that concurrent requests with different headers
    (tokens, API keys, etc.) don't interfere with each other.
    """

    def __init__(self, server_name: str, object_list: List[object],
                 source_file: str = None, enable_request_isolation: bool = True):
        """
        Initialize the base MCP server.

        Args:
            server_name: Name of the MCP server
            object_list: List of objects whose methods will be registered as tools
            source_file: Path to source file for config resolution
            enable_request_isolation: Enable request-scoped environment isolation
        """
        self.server_name = server_name
        self.logger = get_logger()
        self.server_config: MCPServerConfig = get_server_config(server_name, source_file)
        self.mcp = FastMCP(server_name)
        self.enable_request_isolation = enable_request_isolation
        self.registry = MCPRegistry(self.mcp)

        # Setup request-aware environment if enabled
        if enable_request_isolation:
            self._setup_request_isolation()

        # Register tools after initialization
        self.object_list = object_list
        self.registry.register_tools(object_list)
        self._register_routes()

    def _setup_request_isolation(self):
        """
        Setup request-aware environment wrapper.
        This makes os.environ automatically use request-scoped values.
        """
        # Only wrap once
        if not isinstance(os.environ, RequestAwareEnviron):
            original_environ = os.environ
            os.environ = RequestAwareEnviron(original_environ)
            self.logger.info("Request isolation enabled: os.environ is now request-aware")
        else:
            self.logger.debug("Request isolation already enabled")

    def base_directory(self, file_name):
        """Get the base directory of the MCP server."""
        return os.path.basename(os.path.dirname(os.path.abspath(file_name)))

    def _register_routes(self):
        """Register system routes."""

        @self.mcp.custom_route("/health", methods=["GET"])
        async def health_check(request: Request) -> PlainTextResponse:
            """
            Health check endpoint that returns "OK" when the server is healthy.
            """
            return PlainTextResponse("OK")

        @self.mcp.custom_route("/info", methods=["GET"])
        async def server_info(request: Request) -> JSONResponse:
            """Get server information including configuration and status."""
            # Get AUTH_ENABLED from original environ
            original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ
            auth_enabled = original_environ.get('AUTH_ENABLED', '').lower() == 'true'

            info = {
                "server_name": self.server_name,
                "status": "running",
                "server_config": self.server_config.model_dump(),
                "auth_enabled": auth_enabled,
                "request_isolation": self.enable_request_isolation
            }
            return JSONResponse(info)

        @self.mcp.custom_route("/", methods=["GET"])
        async def root(request: Request):
            """Root endpoint with API documentation."""
            # Get AUTH_ENABLED from original environ
            original_environ = os.environ._original if isinstance(os.environ, RequestAwareEnviron) else os.environ
            auth_enabled = original_environ.get('AUTH_ENABLED', '').lower() == 'true'

            info = {
                "message": f"MCP Server: {self.server_name}",
                "endpoints": {
                    "POST /mcp": "MCP server tools via streamable http",
                    "GET /health": "Health check endpoint",
                    "GET /info": "Get MCP Server information",
                    "GET /debug/env": "Debug request environment (if enabled)"
                },
                "auth_enabled": auth_enabled,
                "features": {
                    "request_isolation": self.enable_request_isolation,
                    "concurrent_requests": "supported" if self.enable_request_isolation else "not isolated"
                }
            }
            return JSONResponse(info)

        @self.mcp.custom_route("/debug/env", methods=["GET"])
        async def debug_env(request: Request) -> JSONResponse:
            """
            Debug endpoint to show current request environment.
            Useful for testing request isolation.
            """
            req_env = request_env.get()

            # Sanitize sensitive values
            sanitized = {}
            sensitive_keys = {'AUTHORIZATION', 'API_KEY', 'API_TOKEN', 'TOKEN', 'SECRET', 'PASSWORD'}

            for key, value in req_env.items():
                key_upper = key.upper()
                if any(sensitive in key_upper for sensitive in sensitive_keys):
                    if len(value) > 8:
                        sanitized[key] = f"{value[:4]}...{value[-4:]}"
                    else:
                        sanitized[key] = "***"
                else:
                    sanitized[key] = value

            return JSONResponse({
                "request_env_count": len(req_env),
                "request_env": sanitized,
                "isolation_enabled": self.enable_request_isolation
            })

    def _get_local_registry_url(self):
        """
        Get the local registry URL by parsing MCP_BASE_URL and forcing localhost.
        Returns the local URL or None if the environment variable is not set.
        """
        if os.environ.get('MCP_REGISTRY_URL') :
            mcp_base_url = os.environ.get('MCP_REGISTRY_URL')
        elif os.environ.get('MCP_BASE_URL'):
            mcp_base_url = f"{os.environ.get('MCP_BASE_URL')}:{os.environ.get('MCP_BASE_URL_PORT', "8082")}"
        else:
            return None

        if not mcp_base_url:
            return None

        try:
            if not mcp_base_url.startswith('http'):
                mcp_base_url = f"http://{mcp_base_url}"

            parsed_url = urlparse(mcp_base_url)
            port = parsed_url.port
            if not port:
                self.logger.warning(f"Could not extract port from '{mcp_base_url}'. Using original URL.")
                return mcp_base_url.rstrip('/')
            
            local_url = os.environ.get('MCP_REGISTRY_URL') or f"http://localhost:{port}"
            self.logger.info(f"MCP_BASE_URL is set. Forcing registry connection to {local_url}")
            return local_url
        except Exception as e:
            self.logger.error(f"Failed to parse '{mcp_base_url}': {e}")
            return None

    async def _register_with_registry(self, port: int):
        """Register the server with the MCP registry if MCP_BASE_URL is set."""
        registry_base_url = self._get_local_registry_url()
        if not registry_base_url:
            self.logger.info("MCP_BASE_URL not set, skipping registration.")
            return

        registry_url = f"{registry_base_url}/register"
        server_info = {
            "name": self.server_name,
            "description": getattr(self.server_config, "description", "No description available"),
            "endpoint": f"http://localhost:{port}",
            "port": port,
            "registered_via": "dynamic",
            "framework": "mcp"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(registry_url, json=server_info)
                if response.status_code == 200:
                    self.logger.info(f"Successfully registered server '{self.server_name}' with registry at {registry_base_url}")
                else:
                    self.logger.error(f"Failed to register server using {registry_url}. Status: {response.status_code}, Response: {response.text}")
        except httpx.RequestError as e:
            self.logger.error(f"Error connecting to MCP registry at {registry_url}: {e}")

    async def _deregister_from_registry(self):
        """Deregister the server from the MCP registry."""
        registry_base_url = self._get_local_registry_url()
        if not registry_base_url:
            return

        registry_url = f"{registry_base_url}/deregister"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(registry_url, json={"name": self.server_name})
                if response.status_code == 200:
                    self.logger.info(f"Successfully deregistered server '{self.server_name}' from registry.")
                else:
                    self.logger.error(f"Failed to deregister server. Status: {response.status_code}, Response: {response.text}")
        except httpx.RequestError as e:
            self.logger.error(f"Error connecting to MCP registry at {registry_url}: {e}")

    def run(self, transport: Literal["stdio", "streamable-http", "sse"] = "stdio", port: int = None):
        """
        Run the MCP server.

        Args:
            transport: Transport method to use (stdio, streamable-http, sse)
            port: Port to run the server on (for http/sse transports)
        """
        if transport not in VALID_TRANSPORTS:
            raise TransportError(f"Invalid transport: {transport}. Must be one of {VALID_TRANSPORTS}")

        # # Add CORS middleware to allow cross-origin requests for web interfaces
        # self.mcp.add_middleware(
        #     CORSMiddleware,
        #     allow_origins=["*"],
        #     allow_credentials=True,
        #     allow_methods=["*"],
        #     allow_headers=["*"]
        # )
        #
        # # Add authentication middleware first (runs first in the chain)
        # self.mcp.add_middleware(AuthenticationMiddleware(self.server_name))

        # Add header capture middleware for request isolation
        if self.enable_request_isolation:
            self.mcp.add_middleware(HeaderCaptureMiddleware())
            self.logger.info("HeaderCaptureMiddleware added - request isolation active")

        port = port if port is not None else self.server_config.port

        if transport in ["streamable-http", "sse"]:
            port = port if port else 8000
            
            # Register with registry
            try:
                asyncio.run(self._register_with_registry(port))
            except Exception as e:
                self.logger.error(f"Error during registration: {e}")

            self.logger.info(f"Starting MCP server '{self.server_name}' on {transport}://0.0.0.0:{port}")
            try:
                self.mcp.run(transport=transport, port=port, host="0.0.0.0")
            finally:
                # Deregister from registry
                try:
                    asyncio.run(self._deregister_from_registry())
                except Exception as e:
                    self.logger.error(f"Error during deregistration: {e}")
        else:
            self.logger.info(f"Starting MCP server '{self.server_name}' on {transport}")
            self.mcp.run(transport=transport)

    def main(self):
        """
        Main entry point for running the server from command line.
        Parses arguments and calls run().
        """
        parser = argparse.ArgumentParser(description=f"Run MCP server - {self.server_name}")
        parser.add_argument("--transport", default="stdio", help="Transport method (default: stdio)")
        parser.add_argument("--port", help="Port for streamable http transport method", required=False)
        args = parser.parse_args()
        self.run(transport=args.transport, port=int(args.port) if args.port else None)
