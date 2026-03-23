import asyncio
import json
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from urllib.parse import urljoin

import aiohttp
from .config import ClientConfig
from .exceptions import ConnectionError, APIError, ServerStartupError

class AgentClient:
    """
    An asynchronous HTTP client for interacting with an agent server.
    Can either connect to a running server or manage a local server process.
    """

    def __init__(self, config: ClientConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None
        self._server_process: Optional[asyncio.subprocess.Process] = None
        
        host, port = self._extract_host_port_from_args()
        self._host = host
        self._port = port

        base_url_str = str(config.url) if config.url else f"http://{self._host}:{self._port}"
        self._base_url = base_url_str if base_url_str.endswith('/') else base_url_str + '/'
        
        # Setup logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.config.log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _extract_host_port_from_args(self):
        host = self.config.host
        port = self.config.port
        
        if self.config.args:
            try:
                port_index = self.config.args.index("--port") + 1
                if port_index < len(self.config.args):
                    port = int(self.config.args[port_index])
            except (ValueError, IndexError):
                pass

            try:
                host_index = self.config.args.index("--host") + 1
                if host_index < len(self.config.args):
                    host = self.config.args[host_index]
            except (ValueError, IndexError):
                pass
                
        return host, port

    async def __aenter__(self):
        if self.config.command:
            await self._start_server()

        try:
            self._session = aiohttp.ClientSession(
                headers=self.config.headers,
                timeout=aiohttp.ClientTimeout(total=self.config.timeout)
            )
            await self._wait_for_server()
        except Exception as e:
            await self.__aexit__(None, None, None)
            raise ConnectionError(f"Failed to connect to the server: {e}") from e
        
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session and not self._session.closed:
            await self._session.close()
        if self._server_process:
            await self._stop_server()

    async def _start_server(self):
        try:
            cmd = [self.config.command] + (self.config.args or [])
            self.logger.info(f"Starting server with command: {' '.join(cmd)}")
            self._server_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            # Start background tasks to log stdout and stderr
            asyncio.create_task(self._log_stream(self._server_process.stdout, self.logger.info))
            asyncio.create_task(self._log_stream(self._server_process.stderr, self.logger.error))

        except Exception as e:
            raise ServerStartupError(f"Failed to start server process: {e}") from e

    async def _log_stream(self, stream: asyncio.StreamReader, log_func):
        """Reads from a stream and logs each line."""
        while not stream.at_eof():
            line = await stream.readline()
            if line:
                log_func(line.decode().strip())

    async def _stop_server(self):
        if self._server_process:
            self.logger.info("Stopping server process...")
            self._server_process.terminate()
            try:
                await asyncio.wait_for(self._server_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("Server did not terminate gracefully, killing it.")
                self._server_process.kill()
            self._server_process = None

    async def _wait_for_server(self):
        health_url = urljoin(self._base_url, self.config.health_endpoint.lstrip('/'))
        self.logger.info(f"Waiting for server to become healthy at {health_url}...")
        
        async def check_health():
            while True:
                try:
                    async with self._session.get(health_url) as response:
                        if response.status == 200:
                            self.logger.info("Server is healthy.")
                            return
                except aiohttp.ClientError:
                    pass
                await asyncio.sleep(1)

        try:
            await asyncio.wait_for(check_health(), timeout=self.config.startup_timeout)
        except asyncio.TimeoutError:
            raise ConnectionError(f"Server at {self._base_url} did not become healthy in {self.config.startup_timeout} seconds.")

    async def _request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if not self._session:
            raise ConnectionError("Client session is not started. Use 'async with AgentClient(...)'.")

        url = urljoin(self._base_url, endpoint.lstrip('/'))
        try:
            async with self._session.request(method, url, json=data) as response:
                if response.status != 200:
                    raise APIError(response.status, await response.text())
                return await response.json()
        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to the server: {e}") from e

    async def _stream_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        if not self._session:
            raise ConnectionError("Client session is not started. Use 'async with AgentClient(...)'.")

        url = urljoin(self._base_url, endpoint.lstrip('/'))
        try:
            async with self._session.request(method, url, json=data) as response:
                if response.status != 200:
                    raise APIError(response.status, await response.text())
                
                async for line in response.content:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data:"):
                        content = line[5:].strip()
                        if content == "[DONE]":
                            break
                        try:
                            yield json.loads(content)
                        except json.JSONDecodeError:
                            yield {"content": content}

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Failed to connect to the server: {e}") from e
        except asyncio.CancelledError:
            pass

    async def invoke(self, message: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload = {"message": message}
        if config:
            payload.update(config)
        return await self._request("POST", "chat", data=payload)

    async def stream(self, message: str, config: Optional[Dict[str, Any]] = None) -> AsyncGenerator[Dict[str, Any], None]:
        payload = {"message": message}
        if config:
            payload.update(config)
        async for chunk in self._stream_request("POST", "chat/stream", data=payload):
            yield chunk
