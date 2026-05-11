"""Backward-compat shim.

Historically this module defined :class:`AgentClient` (async, aiohttp-based).
The implementation has moved to :mod:`async_client`. ``AgentClient`` is now
an alias for :class:`AsyncAgentClient`, kept here so existing
``from oai_agent_client.agent_client import AgentClient`` and
``from oai_agent_client.agent_client import _redact_headers`` imports keep
working.
"""
from .async_client import AsyncAgentClient
from ._base import _redact_headers

# Backward-compat alias.
AgentClient = AsyncAgentClient

__all__ = ["AgentClient", "AsyncAgentClient", "_redact_headers"]
