"""
oai_platform_core.security.trusted_peers — Docker-aware trusted-peer detection.

:func:`is_trusted_peer` decides whether an incoming request's source IP should
bypass token authentication.  It covers:

* Exact loopback addresses and well-known names (``127.0.0.1``, ``::1``,
  ``localhost``, ``host.docker.internal``, …).
* The standard Docker bridge and overlay subnets — containers calling the host
  via ``host.docker.internal`` appear with a bridge IP (e.g. ``172.17.0.2``)
  as their *source*, never the literal string ``host.docker.internal``.
* Any extra CIDR blocks supplied at runtime via the ``TRUSTED_SUBNETS``
  environment variable (comma-separated, e.g. ``"10.8.0.0/24,10.9.0.0/24"``).

Typical usage inside a FastAPI security dependency::

    from oai_platform_core.security.trusted_peers import is_trusted_peer

    peer = request.client.host if request.client else ""
    if is_trusted_peer(peer) and not force_auth:
        return  # bypass authentication
"""

from __future__ import annotations

import functools
import ipaddress
import logging
import os

__all__ = ["is_trusted_peer", "trusted_subnets", "TRUSTED_PEER_NAMES"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exact-match trusted names / IPs
# ---------------------------------------------------------------------------

TRUSTED_PEER_NAMES: frozenset = frozenset({
    "127.0.0.1",
    "::1",
    "localhost",
    "0.0.0.0",
    "host.docker.internal",
})

# ---------------------------------------------------------------------------
# Subnet-based trust (Docker bridge / overlay networks)
# ---------------------------------------------------------------------------

#: Default CIDR blocks that are trusted without a token.
#:
#: ``172.16.0.0/12``  — Docker's default bridge range (172.17–31.x.x)
#: ``192.168.65.0/24``— Docker Desktop host-gateway on macOS and Windows
DEFAULT_TRUSTED_CIDRS: tuple[str, ...] = (
    "172.16.0.0/12",
    "192.168.65.0/24",
)


@functools.lru_cache(maxsize=1)
def trusted_subnets() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    """Return the cached tuple of trusted :class:`ipaddress.ip_network` objects.

    Built once from :data:`DEFAULT_TRUSTED_CIDRS` plus any extra entries in the
    ``TRUSTED_SUBNETS`` environment variable.  The cache is invalidated
    automatically whenever the environment changes (process restart).

    Example ``TRUSTED_SUBNETS`` value::

        TRUSTED_SUBNETS=10.8.0.0/24,10.9.0.0/24
    """
    cidrs: list[str] = list(DEFAULT_TRUSTED_CIDRS)
    extra = os.environ.get("TRUSTED_SUBNETS", "").strip()
    if extra:
        cidrs.extend(c.strip() for c in extra.split(",") if c.strip())

    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for cidr in cidrs:
        try:
            nets.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid TRUSTED_SUBNETS entry: %r", cidr)

    logger.debug("Trusted subnets: %s", [str(n) for n in nets])
    return tuple(nets)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_trusted_peer(peer: str) -> bool:
    """Return ``True`` if *peer* should bypass token authentication.

    Checks in order:

    1. Exact membership in :data:`TRUSTED_PEER_NAMES` (loopback, Docker host).
    2. IP address contained in any network returned by :func:`trusted_subnets`.

    Args:
        peer: The client's source address as reported by the ASGI server
              (``request.client.host``).  May be an IPv4 address, an IPv6
              address, a hostname, or an empty string.

    Returns:
        ``True`` when the peer is trusted; ``False`` otherwise.
    """
    if not peer:
        return False

    if peer in TRUSTED_PEER_NAMES:
        logger.debug("Peer %r matched trusted name", peer)
        return True

    try:
        addr = ipaddress.ip_address(peer)
        matched = next((n for n in trusted_subnets() if addr in n), None)
        if matched:
            logger.debug("Peer %s matched trusted subnet %s", peer, matched)
            return True
    except ValueError:
        # peer is a hostname we don't recognise — fall through
        pass

    return False
