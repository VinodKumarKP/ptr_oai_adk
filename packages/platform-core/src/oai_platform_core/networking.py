"""
oai_platform_core.networking — host network address utilities.

Provides best-effort helpers for discovering the machine's public and local
IP addresses.  All functions fall back to ``127.0.0.1`` on failure and never
raise exceptions to the caller.

Public API
----------
get_public_ip()   — external IP as seen by the internet (via ipify.org)
get_local_ip()    — LAN / default-route IP address
get_private_ip()  — alias for get_local_ip() (backward-compat name)
"""

import logging
import socket
from typing import Optional

import requests

logger = logging.getLogger(__name__)

__all__ = ["get_public_ip", "get_local_ip", "get_private_ip"]

_FALLBACK = "127.0.0.1"


def get_public_ip(timeout: int = 5) -> str:
    """Return the machine's public (internet-facing) IP address.

    Uses ``https://api.ipify.org`` for the lookup.  Falls back to
    ``127.0.0.1`` if the request fails or times out.

    Args:
        timeout: HTTP request timeout in seconds (default: 5).

    Returns:
        Public IP string, e.g. ``"203.0.113.42"``.
    """
    try:
        response = requests.get("https://api.ipify.org", timeout=timeout)
        if response.ok:
            return response.text.strip()
    except requests.RequestException as exc:
        logger.warning(
            "Could not determine public IP: %s. Falling back to %s.",
            exc,
            _FALLBACK,
        )
    return _FALLBACK


def get_local_ip() -> str:
    """Return the machine's local (LAN / default-route) IP address.

    Opens a non-blocking UDP socket towards ``8.8.8.8:1`` to discover which
    network interface the OS would use for outbound traffic.  No packets are
    actually sent.  Falls back to ``127.0.0.1`` on any error.

    Returns:
        Local IP string, e.g. ``"192.168.1.10"``.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 1))
        return sock.getsockname()[0]
    except Exception as exc:
        logger.warning(
            "Could not determine local IP: %s. Falling back to %s.",
            exc,
            _FALLBACK,
        )
        return _FALLBACK
    finally:
        sock.close()


def get_private_ip() -> str:
    """Alias for :func:`get_local_ip` (backward-compatible name).

    .. deprecated::
        Use :func:`get_local_ip` in new code.  This alias will be kept
        indefinitely for existing callers.
    """
    return get_local_ip()
