"""
TokenManager — Redis-compatible token store with automatic fallback.

This is the **single canonical implementation** shared by all OAI platform
services (agent-registry, mcp-registry, agent-server, mcp-core, etc.).
Each service passes a ``db_path_name`` to keep its local SQLite file separate
when running without Redis.

Connection priority
-------------------
1. **Redis / Valkey** — connects to ``REDIS_HOST``/``REDIS_PORT`` (or the
   constructor ``host``/``port`` args).  A ``ping()`` with a 2-second timeout
   confirms reachability before committing.
2. **redislite** — SQLite-backed, purely local Redis-compatible store used
   automatically when no Redis/Valkey instance is reachable.  Requires the
   ``redislite`` extra (``pip install oai-platform-core[redislite]`` or
   ``pip install redislite``).

The fallback path keeps all token data in a local SQLite file so that any
service works out-of-the-box for development / demo without a Redis server.

Environment variables
---------------------
REDIS_HOST            Valkey/Redis host            (default: localhost)
REDIS_PORT            Valkey/Redis port             (default: 6379)
REDISLITE_DB_PATH     Full path to the redislite DB (overrides db_path_name)

Token format
------------
``<43-char random>.<base64url metadata>``

The 43-character random prefix comes from ``secrets.token_urlsafe(32)``, which
always produces exactly 43 URL-safe characters.  Metadata (server, user_id,
role_id) is JSON-encoded and base64url-encoded without padding.  This lets
``parse_token`` extract metadata without a DB round-trip.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import secrets
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TokenManager:
    """
    API Token Management System.

    Supports multiple tokens per server with optional per-user scoping and TTL.

    Token format: ``<43-char random>.<base64url metadata>``

    Uses Redis/Valkey when available; falls back to a redislite
    (SQLite-backed) store automatically.

    Parameters
    ----------
    host:
        Redis/Valkey hostname.  Overridden by the ``REDIS_HOST`` env var.
    port:
        Redis/Valkey port.  Overridden by the ``REDIS_PORT`` env var.
    db:
        Redis logical DB index (ignored for redislite).
    redislite_path:
        Explicit full path for the redislite SQLite file.  Takes precedence
        over ``REDISLITE_DB_PATH`` env var and ``db_path_name``.
    db_path_name:
        Filename (not full path) used for the redislite SQLite file when
        neither ``redislite_path`` nor ``REDISLITE_DB_PATH`` is set.
        Each service should pass its own unique name to avoid file collisions,
        e.g. ``"agent_registry_tokens.db"``.  The file is stored under
        ``~/.oai/<db_path_name>``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        redislite_path: Optional[str] = None,
        db_path_name: str = "tokens.db",
    ) -> None:
        redis_host = os.environ.get("REDIS_HOST", host)
        redis_port = int(os.environ.get("REDIS_PORT", port))

        self.r, self.backend = self._connect(
            redis_host, redis_port, db, redislite_path, db_path_name
        )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(
        host: str,
        port: int,
        db: int,
        redislite_path: Optional[str],
        db_path_name: str,
    ):
        """Return ``(client, backend_name)`` for the best available store.

        Tries Redis/Valkey first; on any failure falls back to redislite.
        """
        # ── 1. Try Redis / Valkey ──────────────────────────────────────
        try:
            import redis as _redis_lib

            client = _redis_lib.Redis(
                host=host,
                port=port,
                db=db,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            client.ping()
            logger.info(
                "TokenManager: connected to Redis/Valkey at %s:%s", host, port
            )
            return client, "redis"
        except Exception as exc:
            logger.warning(
                "TokenManager: Redis/Valkey not available at %s:%s (%s) "
                "— falling back to redislite",
                host,
                port,
                exc,
            )

        # ── 2. Fall back to redislite ──────────────────────────────────
        try:
            import redislite as _redislite
        except ImportError:
            raise RuntimeError(
                "Redis/Valkey is unreachable and 'redislite' is not installed.\n"
                "Install it with:  pip install 'oai-platform-core[redislite]'\n"
                "Or start a Redis/Valkey server and set REDIS_HOST/REDIS_PORT."
            ) from None

        resolved_path = (
            redislite_path
            or os.environ.get("REDISLITE_DB_PATH")
            or os.path.join(os.path.expanduser("~"), ".oai", db_path_name)
        )
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)

        client = _redislite.Redis(resolved_path, decode_responses=True)
        logger.info("TokenManager: using redislite backend at %s", resolved_path)
        return client, "redislite"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hset_mapping(self, key: str, mapping: Dict[str, Any]) -> None:
        """Write a hash mapping — compatible with both redis-py and redislite.

        redis-py ≥ 3.x supports ``hset(key, mapping=…)``.  Older redislite
        builds only expose ``hmset``.  This shim tries the modern form first.
        """
        try:
            self.r.hset(key, mapping=mapping)
        except TypeError:
            self.r.hmset(key, mapping)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    def generate_token(
        self,
        server_key: str,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Generate a new API token and persist it in the store.

        Args:
            server_key:   Logical name of the server this token is scoped to.
            user_id:      Optional user identifier; e-mail domain is stripped
                          automatically.  Defaults to ``"anonymous"``.
            role_id:      Optional role identifier.  Defaults to ``"default"``.
            ttl_seconds:  Lifetime in seconds.  ``None`` creates a permanent
                          token.

        Returns:
            Token string: ``<43-char random>.<base64url metadata>``
        """
        random_part = secrets.token_urlsafe(32)  # always 43 chars

        # Normalise user_id
        if user_id is not None:
            if "@" in user_id:
                user_id = user_id.split("@")[0]
        else:
            user_id = "anonymous"

        if role_id is None:
            role_id = "default"

        # Embed metadata in the token itself (no DB round-trip needed to parse)
        metadata = {
            "server": server_key,
            "user_id": user_id,
            "role_id": role_id,
        }
        metadata_json = json.dumps(metadata, separators=(",", ":"))
        metadata_b64 = (
            base64.urlsafe_b64encode(metadata_json.encode()).decode().rstrip("=")
        )
        token = f"{random_part}.{metadata_b64}"

        # ── Membership indices (for listing / bulk revocation) ─────────
        if ttl_seconds is not None:
            expiry_ts = int(time.time()) + ttl_seconds
            self.r.zadd(f"{server_key}:{user_id}:ttl", {token: expiry_ts})
            self.r.zadd(f"{server_key}:ttl", {token: expiry_ts})
        else:
            self.r.sadd(f"{server_key}:{user_id}:permanent", token)
            self.r.sadd(f"{server_key}:permanent", token)

        # ── Token metadata hash ────────────────────────────────────────
        now = datetime.utcnow().isoformat()
        token_data: Dict[str, str] = {
            "server_name": server_key,
            "user_id": user_id,
            "role_id": role_id,
            "created_at": now,
            "last_accessed": now,
            "access_count": "0",
            "ttl": str(ttl_seconds) if ttl_seconds is not None else "none",
        }
        self._hset_mapping(f"tokens:{token}", token_data)

        if ttl_seconds is not None:
            self.r.expire(f"tokens:{token}", ttl_seconds)

        return token

    # ------------------------------------------------------------------
    # Token parsing
    # ------------------------------------------------------------------

    def parse_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Extract embedded metadata from a token string without a DB lookup.

        Args:
            token: Token string.

        Returns:
            Dict with keys ``random_part``, ``server``, ``user_id``,
            ``role_id`` — or ``None`` if the token is malformed.
        """
        try:
            # The random prefix is always exactly 43 characters
            if len(token) < 44 or token[43] != ".":
                return None

            random_part = token[:43]
            metadata_b64 = token[44:]

            # Restore stripped base64 padding
            padding = 4 - (len(metadata_b64) % 4)
            if padding != 4:
                metadata_b64 += "=" * padding

            metadata = json.loads(
                base64.urlsafe_b64decode(metadata_b64).decode()
            )

            result: Dict[str, Any] = {
                "random_part": random_part,
                "server": metadata.get("server"),
            }
            if "user_id" in metadata:
                result["user_id"] = metadata["user_id"]
            if "role_id" in metadata:
                result["role_id"] = metadata["role_id"]

            return result
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Token validation
    # ------------------------------------------------------------------

    def validate_token(
        self, server_key: str, token: str
    ) -> Optional[Dict[str, str]]:
        """Validate a token and return user context on success.

        Handles both current (hash-backed) and legacy (set-only) tokens.
        Expired TTL tokens are cleaned up on discovery.

        Args:
            server_key: Logical server name the token must be scoped to.
            token:      Token string to validate.

        Returns:
            ``{"user_id": …, "role_id": …}`` when valid, otherwise ``None``.
        """
        parsed = self.parse_token(token)
        if not parsed:
            return None

        if parsed.get("server") != server_key:
            return None

        # ── Modern token: metadata hash present ───────────────────────
        if self.r.exists(f"tokens:{token}"):
            pipe = self.r.pipeline()
            pipe.hincrby(f"tokens:{token}", "access_count", 1)
            pipe.hset(f"tokens:{token}", "last_accessed", datetime.utcnow().isoformat())
            pipe.execute()
            return {
                "user_id": parsed.get("user_id", "anonymous"),
                "role_id": parsed.get("role_id", "default"),
            }

        # ── Legacy permanent token (no metadata hash) ──────────────────
        user_id = parsed.get("user_id", "anonymous")
        if self.r.sismember(f"{server_key}:{user_id}:permanent", token):
            return {
                "user_id": user_id,
                "role_id": parsed.get("role_id", "default"),
            }

        # ── Legacy TTL token ───────────────────────────────────────────
        ttl_key = f"{server_key}:{user_id}:ttl"
        score = self.r.zscore(ttl_key, token)
        if score is not None:
            if int(score) < int(time.time()):
                # Expired — clean up eagerly
                self.r.zrem(ttl_key, token)
                self.r.zrem(f"{server_key}:ttl", token)
                return None
            return {
                "user_id": user_id,
                "role_id": parsed.get("role_id", "default"),
            }

        return None

    # ------------------------------------------------------------------
    # Token revocation
    # ------------------------------------------------------------------

    def revoke_token(self, token: str) -> bool:
        """Revoke a single token, removing all associated index entries.

        Args:
            token: Token string to revoke.

        Returns:
            ``True`` if the token was found and removed; ``False`` if it was
            not present in the store (already expired / never existed).
        """
        parsed = self.parse_token(token)
        if not parsed:
            return False

        server_key = parsed["server"]
        user_id = parsed.get("user_id", "anonymous")

        removed = self.r.srem(f"{server_key}:{user_id}:permanent", token)
        removed += self.r.zrem(f"{server_key}:{user_id}:ttl", token)

        # Also remove from server-level indices
        self.r.srem(f"{server_key}:permanent", token)
        self.r.zrem(f"{server_key}:ttl", token)

        # Drop the metadata hash
        self.r.delete(f"tokens:{token}")

        return removed > 0

    def revoke_tokens_by_user(self, server_key: str, user_id: str) -> int:
        """Revoke all tokens belonging to *user_id* on *server_key*.

        Returns:
            Number of tokens revoked.
        """
        count = 0
        for info in self.get_all_tokens(
            f"{server_key}:{user_id}", include_expired=True
        ):
            if self.revoke_token(info["token"]):
                count += 1
        return count

    def revoke_tokens_by_server(self, server_key: str) -> int:
        """Revoke all tokens for *server_key*.

        Returns:
            Number of tokens revoked.
        """
        count = 0
        for info in self.get_all_tokens(server_key, include_expired=True):
            if self.revoke_token(info["token"]):
                count += 1
        return count

    def revoke_all_tokens(self, server_key: str) -> int:
        """Alias for :py:meth:`revoke_tokens_by_server`."""
        return self.revoke_tokens_by_server(server_key)

    # ------------------------------------------------------------------
    # Token maintenance
    # ------------------------------------------------------------------

    def cleanup_expired_tokens(self, server_key: str) -> int:
        """Remove all expired TTL tokens for *server_key* from the sorted set.

        Args:
            server_key: Logical server name.

        Returns:
            Number of tokens removed.
        """
        return self.r.zremrangebyscore(
            f"{server_key}:ttl", 0, int(time.time())
        )

    # ------------------------------------------------------------------
    # Token introspection
    # ------------------------------------------------------------------

    def get_token_info(self, token: str) -> Optional[Dict[str, Any]]:
        """Return the stored metadata hash for *token*, or ``None`` if absent.

        Args:
            token: Token string.

        Returns:
            Dict of stored fields, or ``None`` if the token has no hash entry.
        """
        data = self.r.hgetall(f"tokens:{token}")
        return data if data else None

    def get_all_tokens(
        self, token_key: str, include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Return all tokens under *token_key* with their metadata.

        Args:
            token_key:       Key prefix, e.g. ``"my_server"`` or
                             ``"my_server:alice"``.
            include_expired: When ``True``, expired TTL tokens are included.

        Returns:
            List of token-info dicts, permanent tokens first.
        """
        # ── Permanent tokens ───────────────────────────────────────────
        permanent: List[Dict[str, Any]] = []
        for token in self.r.smembers(f"{token_key}:permanent"):
            parsed = self.parse_token(token)
            entry: Dict[str, Any] = {"token": token, "type": "permanent"}
            if parsed and "user_id" in parsed:
                entry["user_id"] = parsed["user_id"]
            permanent.append(entry)

        # ── TTL tokens ─────────────────────────────────────────────────
        current_time = int(time.time())
        ttl_tokens: List[Dict[str, Any]] = []
        for token, expiry in self.r.zrange(
            f"{token_key}:ttl", 0, -1, withscores=True
        ):
            is_expired = expiry <= current_time
            if not include_expired and is_expired:
                continue
            parsed = self.parse_token(token)
            raw = self.get_token_info(token) or {}
            entry = {
                "token": token,
                "type": "ttl",
                "expires_at": int(expiry),
                "expires_in": max(0, int(expiry - current_time)),
                "is_expired": is_expired,
                "access_count": raw.get("access_count", "0"),
                "last_accessed": raw.get("last_accessed"),
                "created_at": raw.get("created_at"),
            }
            if parsed and "user_id" in parsed:
                entry["user_id"] = parsed["user_id"]
            ttl_tokens.append(entry)

        return permanent + ttl_tokens

    def get_tokens_by_user(
        self, server_key: str, user_id: str, include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Return all tokens for *user_id* on *server_key*."""
        return self.get_all_tokens(f"{server_key}:{user_id}", include_expired)

    def get_tokens_by_server(
        self, server_key: str, include_expired: bool = False
    ) -> List[Dict[str, Any]]:
        """Return all tokens for *server_key*."""
        return self.get_all_tokens(server_key, include_expired)
