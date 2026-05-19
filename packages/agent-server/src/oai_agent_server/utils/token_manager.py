"""
TokenManager — Redis-compatible token store with automatic fallback.

Connection priority
-------------------
1. **Redis / Valkey** — connects to ``REDIS_HOST``/``REDIS_PORT`` (or constructor
   args).  A ``ping()`` with a 2-second timeout confirms reachability.
2. **redislite** — SQLite-backed, purely local Redis-compatible store used
   automatically when no Redis/Valkey instance is reachable.  Requires the
   ``redislite`` package (``pip install redislite``).

The fallback path keeps all token data in a local SQLite file so that the
server works out-of-the-box for development/demo without a Redis server.

Environment variables
---------------------
REDIS_HOST            Valkey/Redis host           (default: localhost)
REDIS_PORT            Valkey/Redis port            (default: 6379)
REDISLITE_DB_PATH     Path to the redislite DB     (default: ~/.oai/agent_server_tokens.db)
"""

import base64
import json
import logging
import os
import secrets
import time
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class TokenManager:
    """
    API Token Management System.

    Supports multiple tokens per server with optional user ID and TTL.
    Token format: ``<random_part>.<base64_metadata>``

    Uses Redis/Valkey when available; falls back to a redislite
    (SQLite-backed) store automatically.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        redislite_path: Optional[str] = None,
    ) -> None:
        """Initialise the token store.

        Args:
            host:            Redis/Valkey host (overridden by ``REDIS_HOST`` env var).
            port:            Redis/Valkey port (overridden by ``REDIS_PORT`` env var).
            db:              Redis logical DB index (ignored for redislite).
            redislite_path:  Path for the redislite SQLite file.  Falls back to
                             the ``REDISLITE_DB_PATH`` env var or
                             ``~/.oai/agent_server_tokens.db``.
        """
        redis_host = os.environ.get("REDIS_HOST", host)
        redis_port = int(os.environ.get("REDIS_PORT", port))

        self.r, self.backend = self._connect(
            redis_host, redis_port, db, redislite_path
        )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(host: str, port: int, db: int, redislite_path: Optional[str]):
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
                "Install it with:  pip install redislite\n"
                "Or start a Redis/Valkey server and set REDIS_HOST/REDIS_PORT."
            ) from None

        db_path = (
            redislite_path
            or os.environ.get("REDISLITE_DB_PATH")
            or os.path.join(
                os.path.expanduser("~"), ".oai", "agent_server_tokens.db"
            )
        )
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        client = _redislite.Redis(db_path, decode_responses=True)
        logger.info("TokenManager: using redislite backend at %s", db_path)
        return client, "redislite"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hset_mapping(self, key: str, mapping: Dict[str, Any]) -> None:
        """Set multiple hash fields — compatible with both redis-py and redislite.

        redis-py ≥ 3.x supports ``hset(key, mapping=…)``.  Older redislite
        builds (which may carry an older redis-py) only have ``hmset``.
        """
        try:
            self.r.hset(key, mapping=mapping)
        except TypeError:
            # Older redis-py / redislite that don't accept mapping= kwarg
            self.r.hmset(key, mapping)  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_token(
        self,
        server_key: str,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        """Generate a token with embedded metadata.

        Args:
            server_key:   Name of the server this token is valid for.
            user_id:      Optional user identifier (email domain stripped).
            role_id:      Optional role identifier.
            ttl_seconds:  Time-to-live in seconds; ``None`` for permanent tokens.

        Returns:
            Token string: ``<43-char random>.<base64 metadata>``
        """
        random_part = secrets.token_urlsafe(32)

        if user_id is not None:
            if "@" in user_id:
                user_id = user_id.split("@")[0]
        else:
            user_id = "anonymous"

        if role_id is None:
            role_id = "default"

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

        # Store membership indices
        if ttl_seconds is not None:
            expiry_ts = int(time.time()) + ttl_seconds
            self.r.zadd(f"{server_key}:{user_id}:ttl", {token: expiry_ts})
            self.r.zadd(f"{server_key}:ttl", {token: expiry_ts})
        else:
            self.r.sadd(f"{server_key}:{user_id}:permanent", token)
            self.r.sadd(f"{server_key}:permanent", token)

        # Store token metadata hash
        token_data = {
            "server_name": server_key,
            "user_id": user_id,
            "role_id": role_id,
            "created_at": datetime.utcnow().isoformat(),
            "last_accessed": datetime.utcnow().isoformat(),
            "access_count": "0",
            "ttl": str(ttl_seconds) if ttl_seconds is not None else "none",
        }
        self._hset_mapping(f"tokens:{token}", token_data)

        if ttl_seconds is not None:
            self.r.expire(f"tokens:{token}", ttl_seconds)

        return token

    def parse_token(self, token: str) -> Optional[Dict]:
        """Parse a token and return its embedded metadata.

        Args:
            token: Token string.

        Returns:
            ``{"random_part", "server", "user_id", "role_id"}`` or ``None`` if
            malformed.
        """
        try:
            if len(token) < 44 or token[43] != ".":
                return None

            random_part = token[:43]
            metadata_b64 = token[44:]

            padding = 4 - (len(metadata_b64) % 4)
            if padding != 4:
                metadata_b64 += "=" * padding

            metadata = json.loads(base64.urlsafe_b64decode(metadata_b64).decode())

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

    def validate_token(self, server_key: str, token: str) -> Optional[Dict[str, str]]:
        """Validate a token for the given server key and return user info.

        Handles both current (hash-backed) and legacy (set-only) tokens.
        Expired TTL tokens are removed on discovery.

        Args:
            server_key: Server the token must be valid for.
            token:      Token string.

        Returns:
            ``{"user_id": …, "role_id": …}`` when valid, otherwise ``None``.
        """
        parsed = self.parse_token(token)
        if not parsed:
            return None

        if parsed.get("server") != server_key:
            return None

        # Modern token: metadata hash present
        if self.r.exists(f"tokens:{token}"):
            pipe = self.r.pipeline()
            pipe.hincrby(f"tokens:{token}", "access_count", 1)
            pipe.hset(f"tokens:{token}", "last_accessed", datetime.utcnow().isoformat())
            pipe.execute()
            return {
                "user_id": parsed.get("user_id", "anonymous"),
                "role_id": parsed.get("role_id", "default"),
            }

        # Legacy permanent token
        user_id = parsed.get("user_id", "anonymous")
        if self.r.sismember(f"{server_key}:{user_id}:permanent", token):
            return {
                "user_id": user_id,
                "role_id": parsed.get("role_id", "default"),
            }

        # Legacy TTL token
        ttl_key = f"{server_key}:{user_id}:ttl"
        score = self.r.zscore(ttl_key, token)
        if score is not None:
            if int(score) < int(time.time()):
                self.r.zrem(ttl_key, token)
                self.r.zrem(f"{server_key}:ttl", token)
                return None
            return {
                "user_id": user_id,
                "role_id": parsed.get("role_id", "default"),
            }

        return None

    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token.

        Args:
            token: Token to revoke.

        Returns:
            ``True`` if the token was found and removed.
        """
        parsed = self.parse_token(token)
        if not parsed:
            return False

        server_key = parsed["server"]
        user_id = parsed.get("user_id", "anonymous")

        removed = self.r.srem(f"{server_key}:{user_id}:permanent", token)
        removed += self.r.zrem(f"{server_key}:{user_id}:ttl", token)

        self.r.srem(f"{server_key}:permanent", token)
        self.r.zrem(f"{server_key}:ttl", token)
        self.r.delete(f"tokens:{token}")

        return removed > 0

    def cleanup_expired_tokens(self, server_key: str) -> int:
        """Remove all expired TTL tokens for a server.

        Args:
            server_key: Server key.

        Returns:
            Number of tokens removed.
        """
        return self.r.zremrangebyscore(
            f"{server_key}:tokens:ttl", 0, int(time.time())
        )

    def get_token_info(self, token: str) -> Optional[Dict]:
        """Return the metadata hash for *token*, or ``None`` if absent."""
        return self.r.hgetall(f"tokens:{token}") or None

    def get_all_tokens(
        self, token_key: str, include_expired: bool = False
    ) -> list:
        """Return all tokens under *token_key* with their metadata.

        Args:
            token_key:       Key prefix (e.g. ``"server"`` or ``"server:user"``).
            include_expired: When ``True``, expired TTL tokens are included.

        Returns:
            List of token-info dicts.
        """
        permanent_tokens = []
        for token in self.r.smembers(f"{token_key}:permanent"):
            parsed = self.parse_token(token)
            info: Dict[str, Any] = {"token": token, "type": "permanent"}
            if parsed and "user_id" in parsed:
                info["user_id"] = parsed["user_id"]
            permanent_tokens.append(info)

        current_time = int(time.time())
        ttl_tokens = []
        for token, expiry in self.r.zrange(
            f"{token_key}:ttl", 0, -1, withscores=True
        ):
            is_expired = expiry <= current_time
            if include_expired or not is_expired:
                parsed = self.parse_token(token)
                raw = self.get_token_info(token) or {}
                info = {
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
                    info["user_id"] = parsed["user_id"]
                ttl_tokens.append(info)

        return permanent_tokens + ttl_tokens

    def get_tokens_by_user(
        self, server_key: str, user_id: str, include_expired: bool = False
    ) -> list:
        """Return all tokens for *user_id* on *server_key*."""
        return self.get_all_tokens(f"{server_key}:{user_id}", include_expired)

    def get_tokens_by_server(
        self, server_key: str, include_expired: bool = False
    ) -> list:
        """Return all tokens for *server_key*."""
        return self.get_all_tokens(server_key, include_expired)

    def revoke_tokens_by_user(self, server_key: str, user_id: str) -> int:
        """Revoke all tokens for *user_id* on *server_key*.

        Returns:
            Number of tokens revoked.
        """
        count = 0
        for info in self.get_all_tokens(f"{server_key}:{user_id}", include_expired=True):
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
