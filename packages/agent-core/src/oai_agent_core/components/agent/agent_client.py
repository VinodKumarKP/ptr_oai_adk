"""
AgentClient - Production HTTP Client with integrated server management and streaming support

This module provides a robust HTTP client that can manage its own server process
and handle both regular and streaming responses.
"""

import asyncio
import json
import os
import signal
import subprocess
import threading
from typing import Optional, Dict, Any, AsyncGenerator, Union, Generator, List

import aiohttp
import requests
import sys
from pydantic import BaseModel, Field, model_validator

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_core.utils.logger import create_agent_logger, Logger


class AgentClientError(Exception):
    """Base exception for AgentClient errors"""
    pass


class ServerStartupError(AgentClientError):
    """Raised when server fails to start"""
    pass


class ConnectionError(AgentClientError):
    """Raised when client cannot connect to server"""
    pass


class StreamingError(AgentClientError):
    """Raised when streaming operations fail"""
    pass


class ServerConfig(BaseModel):
    name: str = Field(..., min_length=1, description="The name of the server, must not be empty.")
    command: Optional[str] = Field(None, min_length=1,
                                   description="The command to execute, if using command-line execution.")
    args: Optional[List[str]] = Field(None, description="A list of arguments for the command.")
    url: Optional[str] = Field(None, min_length=1, description="The URL for the server, if using URL-based connection.")

    # Custom validator to enforce the 'Command/args OR url' rule
    @model_validator(mode='after')
    def check_command_or_url(cls, values):
        command = values.command
        args = values.args
        url = values.url

        has_command_and_args = command is not None and args is not None
        has_url = url is not None

        if has_command_and_args and has_url:
            raise ValueError("Cannot specify both 'Command'/'args' and 'url'. Choose one.")

        if not (has_command_and_args or has_url):
            raise ValueError(
                "Either 'Command' and 'args' must both be present (and not empty), or 'url' must be present (and not empty).")

        # Further checks for command and args if they are present
        if has_command_and_args:
            if not command:  # Pydantic's min_length would catch this for 'Command', but explicit check for clarity.
                raise ValueError("'Command' cannot be empty if present.")
            if not isinstance(args, list):  # Pydantic's List[str] would catch this, but explicit for clarity.
                raise ValueError("'args' must be a list if present.")
            if not args:  # Check if args list is empty
                raise ValueError("'args' list cannot be empty if present.")

        # Pydantic's min_length for 'url' would handle empty url if present.

        return values


class AgentClient(BaseAgent):
    """
    HTTP Client with integrated server management and streaming support

    Features:
    - Automatic server process management
    - Support for both regular and streaming HTTP requests
    - Server-Sent Events (SSE) parsing
    - Graceful shutdown and error handling
    - Connection health monitoring
    """

    def __init__(self,
                 server_config: Dict[str, Any],
                 timeout: int = 300,
                 logger: Optional[Logger] = None,
                 log_config: Optional[Dict[str, Any]] = None
                 ):
        """
        Initialize AgentClient

        Args:
            server_config: Configuration for server management
                - For managed server (stdio): {'command': str, 'args': list, 'port': str, ...}
                - For external server (http): {'url': str, 'health_endpoint': str, ...}
            timeout: Request timeout in seconds
            logger: Optional custom logger instance
            log_config: Optional configuration for creating a new logger
        """
        ServerConfig(**server_config)
        self.server_config = server_config
        self.agent_name = server_config.get('name')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self.connected = False

        # Initialize logger
        if logger:
            self.logger = logger
        else:
            log_config = log_config or {}
            self.logger = create_agent_logger(
                name=log_config.get('name', self.agent_name),
                log_dir=log_config.get('log_dir', 'logs'),
                log_level=log_config.get('log_level', 'INFO'),
                enable_json=log_config.get('enable_json', False),
                enable_performance=log_config.get('enable_performance', True)
            )

        # Server management
        self.server_process: Optional[subprocess.Popen] = None
        self.server_port = server_config.get('port', 8000)
        self.server_host = server_config.get('host', 'localhost')

        # Determine client type
        self.type = self._determine_client_type()
        self.server_url = self._build_server_url()
        self.headers = server_config.get('headers', {})
        self.headers.update(server_config.get('env', {}))
        self.agent_config = {}

        # Shutdown handling
        self._shutdown_event = asyncio.Event()
        self._setup_signal_handlers()

    def _determine_client_type(self) -> str:
        """Determine client type based on server config"""
        if 'command' in self.server_config:
            return 'stdio'
        elif 'url' in self.server_config:
            return 'http'
        else:
            raise ValueError("server_config must contain either 'command' or 'url'")

    def _build_server_url(self) -> str:
        """Build server URL based on configuration"""
        if self.type == 'http':
            return self.server_config['url']
        else:
            return f"http://{self.server_host}:{self.server_port}"

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        if sys.platform != 'win32' and threading.current_thread() is threading.main_thread():
            def signal_handler(signum, frame):
                self.logger.info(f"Received signal {signum}, initiating shutdown...")
                self._shutdown_event.set()

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

    async def __aenter__(self):
        """Async context manager entry"""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.stop()

    async def start(self) -> bool:
        """
        Start server and establish client connection

        Returns:
            bool: True if successfully started, False otherwise

        Raises:
            ServerStartupError: If server fails to start
            ConnectionError: If client cannot connect to server
        """
        self.logger.info("Starting AgentClient...")

        try:
            # Start the server if needed
            if self.type == 'stdio':
                if not await self._start_server():
                    raise ServerStartupError("Failed to start server process")

            # Connect client to server
            if not await self._connect_to_server():
                await self._stop_server()
                raise ConnectionError("Failed to connect to server")

            self.logger.info("AgentClient started successfully!")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start AgentClient: {e}")
            await self.stop()
            raise

    async def initialize(self):
        """Initialize the client"""
        await self.start()

    async def log_stream(self, stream, log_func):
        """Log output from a stream line by line.

        Args:
            stream: The stream to read from (e.g., stdout/stderr).
            log_func: The logging function to use (e.g., logger.info).
        """
        while True:
            line = await asyncio.get_event_loop().run_in_executor(None, stream.readline)
            if not line:
                break
            log_func(line.rstrip())

    async def _start_server(self) -> bool:
        """Start the HTTP server as subprocess"""
        try:
            if not await self._wait_for_server_ready(health_check_attempts=2):
                cmd = self._build_server_command()
                self.logger.info(f"Starting server: {' '.join(cmd)}")

                env = os.environ.copy()
                env.update(self.server_config.get('env', {}))

                # Start server process
                self.server_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    shell=False,
                    env=env
                )

                asyncio.create_task(self.log_stream(self.server_process.stdout, self.logger.info))
                asyncio.create_task(self.log_stream(self.server_process.stderr, self.logger.error))

                # Wait for server to start
                startup_delay = self.server_config.get('startup_delay', 10)
                await asyncio.sleep(startup_delay)

                # Check if process is still running
                if self.server_process.poll() is not None:
                    stdout, stderr = self.server_process.communicate()
                    self.logger.error(f"Server failed to start. stdout: {stdout}, stderr: {stderr}")
                    return False

                # Wait for server to be ready
                if await self._wait_for_server_ready():
                    self.logger.info(f"Server ready at {self.server_url}")
                    return True
                else:
                    self.logger.error("Server started but not responding to health checks")
                    await self._stop_server()
                    return False
            else:
                return True

        except Exception as e:
            self.logger.error(f"Failed to start server: {e}")
            return False

    def _build_server_command(self) -> list:
        """Build server command from configuration"""
        cmd = [self.server_config['command']]

        # Add server arguments
        if 'args' in self.server_config:
            cmd.extend(self.server_config['args'])
        #
        # # Add port if not already in args
        if 'port' in self.server_config and '--port' not in cmd:
            cmd.extend(['--port', str(self.server_config['port'])])

        if '--port' in cmd:
            port_index = cmd.index('--port') + 1
            self.server_port = int(cmd[port_index])
            cmd.extend(['--port', str(self.server_port)])

        self.server_url = self._build_server_url()
        # # Add host if specified and not already in args
        if self.server_host != 'localhost' and '--host' not in cmd:
            cmd.extend(['--host', self.server_host])

        return cmd

    async def _wait_for_server_ready(self, health_check_attempts=10) -> bool:
        """Wait for server to become available"""
        max_attempts = self.server_config.get('health_check_attempts', health_check_attempts)
        health_endpoint = self.server_config.get('health_endpoint', '/health')
        self.logger.error(f"Checking for server health at {self.server_url}{health_endpoint}")

        for attempt in range(max_attempts):
            if self._shutdown_event.is_set():
                return False

            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            f"{self.server_url}{health_endpoint}",
                            timeout=aiohttp.ClientTimeout(total=60)
                    ) as response:
                        if response.status == 200:
                            return True
            except Exception:
                pass

            await asyncio.sleep(1)
            self.logger.error(f"Waiting for server... (attempt {attempt + 1}/{max_attempts})")

        return False

    async def _connect_to_server(self) -> bool:
        """Establish HTTP client connection"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=self.timeout,
                connector=aiohttp.TCPConnector(limit=100, limit_per_host=30)
            )

            # Test connection
            health_endpoint = self.server_config.get('health_endpoint', '/health')
            async with self.session.get(f"{self.server_url}{health_endpoint}") as response:
                if response.status == 200:
                    self.connected = True
                    self.logger.info(f"Client connected to server at {self.server_url}")
                    agent_config = await self.agent_info()
                    self.agent_config = agent_config.get('agent_config', {})
                    return True
                else:
                    self.logger.error(f"Server health check failed with status {response.status}")
                    return False

        except Exception as e:
            self.logger.error(f"Failed to connect client to server: {e}")
            return False

    async def stop(self):
        """Stop client and server gracefully"""
        self.logger.info("Stopping AgentClient...")

        # Set shutdown event first
        self._shutdown_event.set()

        # Give any running streams a moment to notice the shutdown
        await asyncio.sleep(0.1)

        # Disconnect client
        if self.session and not self.session.closed:
            try:
                await self.session.close()
                self.connected = False
                self.logger.info("Client disconnected")
            except Exception as e:
                self.logger.debug(f"Error closing session: {e}")

        # Stop server
        await self._stop_server()
        self.logger.info("AgentClient stopped")

    async def _stop_server(self):
        """Stop the server subprocess gracefully"""
        if not self.server_process:
            return

        self.logger.info("Stopping server...")

        try:
            # Try graceful shutdown first
            self.server_process.terminate()

            try:
                await asyncio.wait_for(
                    asyncio.create_task(self._wait_for_process()),
                    timeout=5.0
                )
                self.logger.info("Server stopped gracefully")
            except asyncio.TimeoutError:
                self.logger.warning("Server didn't stop gracefully, forcing shutdown...")
                self.server_process.kill()
                await self._wait_for_process()
                self.logger.info("Server force-stopped")

        except Exception as e:
            self.logger.error(f"Error stopping server: {e}")
        finally:
            self.server_process = None

    async def _wait_for_process(self):
        """Wait for process to terminate"""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.server_process.wait)

    def is_server_running(self) -> bool:
        """Check if server process is running"""
        return (self.server_process is not None and
                self.server_process.poll() is None)

    def is_connected(self) -> bool:
        """Check if client is connected to server"""
        return self.connected and self.session and not self.session.closed

    async def agent_info(self) -> Dict[str, Any]:
        """Get agent info from server"""
        try:
            agent_info_endpoint = self.server_config.get('agent_info_endpoint', '/agent/info')
            result = await self.get(agent_info_endpoint)
            return result
        except Exception as e:
            return {
                'error': str(e)
            }

    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on server

        Returns:
            Dict containing health status
        """
        try:
            health_endpoint = self.server_config.get('health_endpoint', '/health')
            result = await self.get(health_endpoint)
            return {
                'status': 'healthy' if not result.get('error') else 'unhealthy',
                'server_running': self.is_server_running(),
                'client_connected': self.is_connected(),
                'response': result
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'server_running': self.is_server_running(),
                'client_connected': self.is_connected(),
                'error': str(e)
            }

    async def restart_server(self) -> bool:
        """
        Restart the server

        Returns:
            bool: True if restart successful, False otherwise
        """
        self.logger.info("Restarting server...")

        # Disconnect client
        if self.session and not self.session.closed:
            await self.session.close()
            self.connected = False

        # Stop server
        await self._stop_server()

        # Start server again
        if self.type == 'stdio':
            if await self._start_server():
                return await self._connect_to_server()

        return False

    def _parse_sse_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a Server-Sent Events line"""
        line = line.strip()
        if not line:
            return None

        # Handle different SSE formats
        if line.startswith('data: '):
            data_part = line[6:]  # Remove 'data: ' prefix
            if data_part == '[DONE]':
                return {'type': 'done'}
            try:
                # Try to parse as JSON
                parsed_data = json.loads(data_part)
                return {'type': 'data', 'content': parsed_data}
            except json.JSONDecodeError:
                # If not JSON, return as plain text
                return {'type': 'data', 'content': data_part}
        elif line.startswith('event: '):
            return {'type': 'event', 'event': line[7:]}
        elif line.startswith('id: '):
            return {'type': 'id', 'id': line[4:]}
        elif line.startswith('retry: '):
            try:
                return {'type': 'retry', 'retry': int(line[7:])}
            except ValueError:
                return {'type': 'retry', 'retry': line[7:]}
        elif line.startswith('{') or line.startswith('['):
            # Direct JSON without SSE prefix
            try:
                parsed_data = json.loads(line)
                return {'type': 'json', 'content': parsed_data}
            except json.JSONDecodeError:
                return {'type': 'raw', 'content': line}
        else:
            # Plain text line
            return {'type': 'text', 'content': line}

    async def send_request(self,
                           method: str,
                           endpoint: str,
                           data: Optional[Dict[Any, Any]] = None,
                           stream: bool = False,
                           headers: Optional[Dict[str, str]] = None) -> Union[
        Optional[Dict[Any, Any]], AsyncGenerator[Dict[str, Any], None]]:
        """
        Send HTTP request to server

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint
            data: Request data (for POST/PUT requests)
            stream: If True, return an async generator for streaming responses
            headers: Additional headers to send

        Returns:
            For non-streaming: Dict with response data or error
            For streaming: AsyncGenerator yielding parsed SSE events

        Raises:
            ConnectionError: If client is not connected
        """
        if not self.is_connected():
            error_msg = "Client not connected to server"
            self.logger.error(error_msg)
            if stream:
                async def error_generator():
                    yield {"error": error_msg}

                return error_generator()
            raise ConnectionError(error_msg)

        url = f"{self.server_url}/{endpoint.lstrip('/')}"

        headers = self.headers.copy()
        if headers:
            headers.update(headers)

        if stream:
            return self._stream_request(method, url, data, headers)
        else:
            return await self._regular_request(method, url, data, headers)

    async def _regular_request(self,
                               method: str,
                               url: str,
                               data: Optional[Dict[Any, Any]] = None,
                               headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Handle regular (non-streaming) request"""
        try:
            request_headers = headers or {}

            async with self.session.request(
                    method.upper(),
                    url,
                    json=data if data else None,
                    headers=request_headers
            ) as response:

                if response.status in [200, 201]:
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        try:
                            return await response.json()
                        except json.JSONDecodeError:
                            text = await response.text()
                            return {"response": text}
                    else:
                        text = await response.text()
                        return {"response": text}
                else:
                    error_text = await response.text()
                    return {"error": f"HTTP {response.status}: {error_text}"}

        except Exception as e:
            self.logger.error(f"Request failed: {e}")
            return {"error": str(e)}

    async def _stream_request(self,
                              method: str,
                              url: str,
                              data: Optional[Dict[Any, Any]] = None,
                              headers: Optional[Dict[str, str]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming request"""
        response = None
        try:
            # Prepare headers for streaming
            stream_headers = {
                'Accept': 'text/event-stream, text/plain, application/json',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive'
            }
            if headers:
                stream_headers.update(headers)

            # Start the request
            response = await self.session.request(
                method.upper(),
                url,
                json=data if data else None,
                headers=stream_headers
            )

            if response.status not in [200, 201]:
                error_text = await response.text()
                yield {"error": f"HTTP {response.status}: {error_text}"}
                return

            # Handle the streaming response
            async for chunk in self._handle_stream_response(response):
                if self._shutdown_event.is_set():
                    yield {"type": "cancelled", "message": "Client shutting down"}
                    break
                yield chunk

        except (asyncio.CancelledError, GeneratorExit):
            self.logger.info("Stream request cancelled or closed")
            yield {"type": "cancelled", "message": "Stream request cancelled"}
        except Exception as e:
            self.logger.error(f"Stream request failed: {e}")
            yield {"error": f"Stream request failed: {str(e)}"}
        finally:
            # Clean up response
            if response and not response.closed:
                try:
                    response.close()
                except Exception as e:
                    self.logger.debug(f"Error closing response: {e}")

    async def _handle_stream_response(self, response: aiohttp.ClientResponse) -> AsyncGenerator[Dict[str, Any], None]:
        """Handle streaming Server-Sent Events response"""
        buffer = ""

        try:
            async for chunk_bytes in response.content.iter_chunked(8192):
                if self._shutdown_event.is_set():
                    break

                try:
                    chunk_str = chunk_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    chunk_str = chunk_bytes.decode('utf-8', errors='replace')

                buffer += chunk_str

                # Process complete lines
                lines = buffer.split('\n')
                buffer = lines[-1]  # Keep the last incomplete line

                # Process all complete lines
                for line in lines[:-1]:
                    line = line.strip()
                    if line:  # Skip empty lines
                        parsed = self._parse_sse_line(line)
                        if parsed:
                            yield parsed
                            # Stop if we receive a done signal
                            if parsed.get('type') == 'done':
                                return

            # Process any remaining buffer content
            if buffer.strip():
                parsed = self._parse_sse_line(buffer.strip())
                if parsed:
                    yield parsed

        except (asyncio.CancelledError, GeneratorExit):
            self.logger.info("Stream cancelled or closed")
            yield {"type": "cancelled", "message": "Stream cancelled"}
        except Exception as e:
            self.logger.error(f"Stream processing failed: {e}")
            yield {"error": f"Stream processing failed: {str(e)}"}

    # HTTP Method Convenience Methods
    async def get(self, endpoint: str, stream: bool = False, headers: Optional[Dict[str, str]] = None) -> Union[
        Optional[Dict[Any, Any]], AsyncGenerator[Dict[str, Any], None]]:
        """Send GET request"""
        return await self.send_request("GET", endpoint, stream=stream, headers=headers)

    async def post(self, endpoint: str, data: Optional[Dict[Any, Any]] = None, stream: bool = False,
                   headers: Optional[Dict[str, str]] = None) -> Union[
        Optional[Dict[Any, Any]], AsyncGenerator[Dict[str, Any], None]]:
        """Send POST request"""
        return await self.send_request("POST", endpoint, data, stream=stream, headers=headers)

    async def put(self, endpoint: str, data: Optional[Dict[Any, Any]] = None, stream: bool = False,
                  headers: Optional[Dict[str, str]] = None) -> Union[
        Optional[Dict[Any, Any]], AsyncGenerator[Dict[str, Any], None]]:
        """Send PUT request"""
        return await self.send_request("PUT", endpoint, data, stream=stream, headers=headers)

    async def delete(self, endpoint: str, stream: bool = False, headers: Optional[Dict[str, str]] = None) -> Union[
        Optional[Dict[Any, Any]], AsyncGenerator[Dict[str, Any], None]]:
        """Send DELETE request"""
        return await self.send_request("DELETE", endpoint, stream=stream, headers=headers)

    # Application-specific Methods
    async def health(self) -> Dict[str, Any]:
        """Get server health status"""
        return await self.health_check()

    async def chat(self, message: str, stream: bool = False, **kwargs) -> Union[
        Dict[str, Any], AsyncGenerator[Dict[str, Any], None]]:
        """
        Send chat message

        Args:
            message: Chat message to send
            stream: Whether to stream the response
            **kwargs: Additional data to send

        Returns:
            Response data or async generator for streaming
        """
        data = {"message": message}
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            data.update(kwargs.pop('config'))
        else:
            data.update(kwargs)

        if stream:
            return await self.post('/chat/stream', data=data, stream=True)
        else:
            return await self.post('/chat', data=data, stream=False)

    async def chat_stream(self, message: str, **kwargs) -> AsyncGenerator[Dict[str, Any], None]:
        """Send streaming chat request"""
        data = {"message": message, **kwargs}
        return await self.post('/chat/stream', data=data, stream=True)

    def chat_sync(self, message: str, stream: bool = False, **kwargs) -> Union[
        requests.Response, Dict[str, Any], Generator]:
        """
        Send chat message synchronously

        Args:
            message: Chat message to send
            stream: Whether to stream the response
            **kwargs: Additional data to send with the message

        Returns:
            For stream=False: Dict with parsed JSON response or error
            For stream=True: Generator yielding streaming chunks
        """

        if not self.is_connected():
            error_msg = "Client not connected to server"
            self.logger.error(error_msg)
            return {"error": error_msg}

        if stream:
            url = f"{self.server_url}/chat/stream"
        else:
            url = f"{self.server_url}/chat"

        payload = {"message": message}
        if 'config' in kwargs and isinstance(kwargs['config'], dict):
            payload.update(kwargs.pop('config'))
        else:
            payload.update(kwargs)

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                stream=stream,
                timeout=self.timeout.total if hasattr(self.timeout, 'total') else 30
            )

            if not stream:
                # Handle non-streaming response
                if response.status_code in [200, 201]:
                    try:
                        return response.json()
                    except json.JSONDecodeError:
                        return {"response": response.text}
                else:
                    return {"error": f"HTTP {response.status_code}: {response.text}"}
            else:
                # Return generator for streaming response
                return self._process_streaming_response(response)

        except requests.exceptions.RequestException as e:
            if stream:
                def error_generator():
                    yield {"error": f"Request failed: {str(e)}"}

                return error_generator()
            else:
                return {"error": f"Request failed: {str(e)}"}

    def _process_streaming_response(self, response: requests.Response) -> Generator[Dict[str, Any], None, None]:
        """
        Process the streaming response from the agent

        Args:
            response: requests.Response object with stream=True

        Yields:
            Dict containing parsed chunk data or raw content
        """
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix
                    if data == "[DONE]":
                        yield {"type": "done", "message": "Stream ended"}
                        break
                    try:
                        chunk = json.loads(data)
                        yield {"type": "data", "content": chunk}
                    except json.JSONDecodeError:
                        yield {"type": "text", "content": data}
                elif line.strip():  # Handle other non-empty lines
                    try:
                        # Try parsing as direct JSON
                        chunk = json.loads(line)
                        yield {"type": "json", "content": chunk}
                    except json.JSONDecodeError:
                        yield {"type": "raw", "content": line}
        except Exception as e:
            yield {"error": f"Stream processing failed: {str(e)}"}
        finally:
            response.close()

    def chat_stream_sync(self, message: str) -> Generator[Dict[str, Any], None, None]:
        """
        Send streaming chat request synchronously (convenience method)

        Args:
            message: Chat message to send
            **kwargs: Additional data to send

        Returns:
            Generator yielding streaming chunks
        """
        return self.chat_sync(message, stream=True)

    async def astream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Returns a streaming response from the agent"""
        # await self._ensure_initialized()
        return await self.chat(message=user_message, stream=True, **config)

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Returns a streaming response from the agent"""
        await self._ensure_initialized()
        return await self.chat(message=user_message, stream=True, **config)

    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        await self._ensure_initialized()
        return await self.chat(message=user_message, stream=False, **config)

    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        return self.chat_sync(
            message=user_message, stream=False,
            **config
        )
