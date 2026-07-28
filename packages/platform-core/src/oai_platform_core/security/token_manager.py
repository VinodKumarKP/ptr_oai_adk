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
2. **DiskCache** — SQLite-backed, purely local Redis-compatible store used
   automatically when no Redis/Valkey instance is reachable.  Works on all
   platforms including Windows. Requires the ``diskcache`` extra
   (``pip install oai-platform-core[diskcache]`` or ``pip install diskcache``).

The fallback path keeps all token data in a local SQLite file so that any
service works out-of-the-box for development / demo without a Redis server.

Note that probing an *absent* Redis costs the full connect timeout at startup
(several seconds on hosts that neither accept nor reject quickly). Services
that know they have no Redis should set ``TOKEN_CACHE_BACKEND=diskcache`` to
skip the probe entirely.

Environment variables
---------------------
TOKEN_CACHE_BACKEND   ``auto`` (default, probe Redis then fall back),
                      ``redis`` (require Redis; raise if unreachable),
                      ``diskcache`` (skip the Redis probe entirely)
REDIS_HOST            Valkey/Redis host            (default: localhost)
REDIS_PORT            Valkey/Redis port             (default: 6379)
REDIS_CONNECT_TIMEOUT Seconds to wait when probing Redis (default: 2)
CACHE_DB_PATH         Full path to the cache DB (overrides db_path_name)

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


class _DiskCacheRedisAdapter:
    """Adapter to make DiskCache API compatible with redis-py."""

    def __init__(self, cache):
        self.cache = cache
        self.decode_responses = True

    def ping(self):
        """Check cache connectivity."""
        try:
            self.cache['__ping__'] = True
            del self.cache['__ping__']
            return True
        except Exception:
            return False

    def sadd(self, key, *values):
        """Add members to a set."""
        set_val = self.cache.get(key, set())
        if not isinstance(set_val, set):
            set_val = set()
        set_val.update(values)
        self.cache[key] = set_val
        return len(values)

    def srem(self, key, *values):
        """Remove members from a set."""
        if key not in self.cache:
            return 0
        set_val = self.cache.get(key, set())
        if not isinstance(set_val, set):
            return 0
        initial_len = len(set_val)
        set_val.difference_update(values)
        self.cache[key] = set_val
        return initial_len - len(set_val)

    def sismember(self, key, value):
        """Check if value is a member of set."""
        set_val = self.cache.get(key, set())
        return value in set_val if isinstance(set_val, set) else False

    def smembers(self, key):
        """Get all members of a set."""
        return self.cache.get(key, set()) or set()

    def zadd(self, key, mapping):
        """Add members to a sorted set with scores."""
        zset = self.cache.get(key, {})
        if not isinstance(zset, dict):
            zset = {}
        zset.update(mapping)
        self.cache[key] = zset
        return len(mapping)

    def zrem(self, key, *members):
        """Remove members from a sorted set."""
        if key not in self.cache:
            return 0
        zset = self.cache.get(key, {})
        if not isinstance(zset, dict):
            return 0
        count = 0
        for member in members:
            if member in zset:
                del zset[member]
                count += 1
        self.cache[key] = zset
        return count

    def zscore(self, key, member):
        """Get score of member in a sorted set."""
        zset = self.cache.get(key, {})
        return zset.get(member) if isinstance(zset, dict) else None

    def zrange(self, key, start, stop, withscores=False):
        """Get range of members from sorted set."""
        zset = self.cache.get(key, {})
        if not isinstance(zset, dict):
            return []
        items = sorted(zset.items(), key=lambda x: x[1])
        if stop == -1:
            items = items[start:]
        else:
            items = items[start:stop + 1]
        if withscores:
            return items
        return [item[0] for item in items]

    def zremrangebyscore(self, key, min_score, max_score):
        """Remove members with scores in range."""
        if key not in self.cache:
            return 0
        zset = self.cache.get(key, {})
        if not isinstance(zset, dict):
            return 0
        to_remove = [k for k, v in zset.items() if min_score <= v <= max_score]
        count = 0
        for member in to_remove:
            del zset[member]
            count += 1
        self.cache[key] = zset
        return count

    def hset(self, key, mapping=None, **kwargs):
        """Set hash fields."""
        if mapping is None:
            mapping = kwargs
        hash_val = self.cache.get(key, {})
        if not isinstance(hash_val, dict):
            hash_val = {}
        hash_val.update(mapping)
        self.cache[key] = hash_val
        return len(mapping)

    def hmset(self, key, mapping):
        """Legacy: set multiple hash fields."""
        return self.hset(key, mapping=mapping)

    def hgetall(self, key):
        """Get all hash fields."""
        return self.cache.get(key, {}) or {}

    def hincrby(self, key, field, increment=1):
        """Increment hash field by integer."""
        hash_val = self.cache.get(key, {})
        if not isinstance(hash_val, dict):
            hash_val = {}
        current = int(hash_val.get(field, 0))
        hash_val[field] = str(current + increment)
        self.cache[key] = hash_val
        return current + increment

    def exists(self, key):
        """Check if key exists."""
        return 1 if key in self.cache else 0

    def expire(self, key, seconds):
        """Set key expiration (best effort)."""
        # DiskCache doesn't have built-in TTL, but we can track it manually
        # For now, we'll just return 1 to indicate success
        return 1

    def delete(self, key):
        """Delete a key."""
        if key in self.cache:
            del self.cache[key]
            return 1
        return 0

    def pipeline(self):
        """Return a pipeline (for compatibility - we'll use direct calls)."""
        return _DiskCachePipeline(self)


class _DiskCachePipeline:
    """Pipeline adapter for DiskCache."""

    def __init__(self, adapter):
        self.adapter = adapter
        self.commands = []

    def hincrby(self, key, field, increment):
        """Queue hincrby command."""
        self.commands.append(('hincrby', key, field, increment))
        return self

    def hset(self, key, field, value):
        """Queue hset command."""
        self.commands.append(('hset', key, {field: value}))
        return self

    def execute(self):
        """Execute all queued commands."""
        results = []
        for cmd in self.commands:
            if cmd[0] == 'hincrby':
                results.append(self.adapter.hincrby(cmd[1], cmd[2], cmd[3]))
            elif cmd[0] == 'hset':
                results.append(self.adapter.hset(cmd[1], mapping=cmd[2]))
        return results



class TokenManager:
    """
    API Token Management System.

    Supports multiple tokens per server with optional per-user scoping and TTL.

    Token format: ``<43-char random>.<base64url metadata>``

    Uses Redis/Valkey when available; falls back to a DiskCache
    (SQLite-backed) store automatically. DiskCache works on all platforms
    including Windows.

    Parameters
    ----------
    host:
        Redis/Valkey hostname.  Overridden by the ``REDIS_HOST`` env var.
    port:
        Redis/Valkey port.  Overridden by the ``REDIS_PORT`` env var.
    db:
        Redis logical DB index (ignored for DiskCache).
    cache_path:
        Explicit full path for the DiskCache directory.  Takes precedence
        over ``CACHE_DB_PATH`` env var and ``db_path_name``.
    db_path_name:
        Filename (not full path) used for the DiskCache directory when
        neither ``cache_path`` nor ``CACHE_DB_PATH`` is set.
        Each service should pass its own unique name to avoid file collisions,
        e.g. ``"agent_registry_tokens"``.  The directory is stored under
        ``~/.oai/<db_path_name>``.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        cache_path: Optional[str] = None,
        db_path_name: str = "tokens",
    ) -> None:
        redis_host = os.environ.get("REDIS_HOST", host)
        redis_port = int(os.environ.get("REDIS_PORT", port))

        self.r, self.backend = self._connect(
            redis_host, redis_port, db, cache_path, db_path_name
        )

    # ------------------------------------------------------------------
    # Backend selection
    # ------------------------------------------------------------------

    @staticmethod
    def _connect(
        host: str,
        port: int,
        db: int,
        cache_path: Optional[str],
        db_path_name: str,
    ):
        """Return ``(client, backend_name)`` for the best available store.

        Tries Redis/Valkey first; on any failure falls back to DiskCache.
        ``TOKEN_CACHE_BACKEND`` overrides that: ``diskcache`` skips the Redis
        probe (and the connect-timeout wait it costs), ``redis`` makes Redis
        mandatory instead of best-effort.
        """
        backend_pref = os.environ.get("TOKEN_CACHE_BACKEND", "auto").strip().lower()

        # ── 1. Try Redis / Valkey ──────────────────────────────────────
        if backend_pref != "diskcache":
            try:
                import redis as _redis_lib

                timeout = float(os.environ.get("REDIS_CONNECT_TIMEOUT", "2"))
                client = _redis_lib.Redis(
                    host=host,
                    port=port,
                    db=db,
                    decode_responses=True,
                    socket_connect_timeout=timeout,
                    socket_timeout=timeout,
                )
                client.ping()
                logger.info(
                    "TokenManager: connected to Redis/Valkey at %s:%s", host, port
                )
                return client, "redis"
            except Exception as exc:
                if backend_pref == "redis":
                    raise RuntimeError(
                        f"TOKEN_CACHE_BACKEND=redis but Redis/Valkey at "
                        f"{host}:{port} is unreachable: {exc}"
                    ) from exc
                logger.warning(
                    "TokenManager: Redis/Valkey not available at %s:%s (%s) "
                    "— falling back to DiskCache "
                    "(set TOKEN_CACHE_BACKEND=diskcache to skip this probe)",
                    host,
                    port,
                    exc,
                )

        # ── 2. Fall back to DiskCache ──────────────────────────────────
        try:
            import diskcache as _diskcache
        except ImportError:
            raise RuntimeError(
                "Redis/Valkey is unreachable and 'diskcache' is not installed.\n"
                "Install it with:  pip install 'oai-platform-core[diskcache]'\n"
                "Or start a Redis/Valkey server and set REDIS_HOST/REDIS_PORT."
            ) from None

        resolved_path = (
            cache_path
            or os.environ.get("CACHE_DB_PATH")
            or os.path.join(os.path.expanduser("~"), ".oai", db_path_name)
        )
        os.makedirs(resolved_path, exist_ok=True)

        disk_cache = _diskcache.Cache(resolved_path)
        client = _DiskCacheRedisAdapter(disk_cache)
        logger.info("TokenManager: using DiskCache backend at %s", resolved_path)
        return client, "diskcache"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _hset_mapping(self, key: str, mapping: Dict[str, Any]) -> None:
        """Write a hash mapping — compatible with both redis-py and DiskCache.

        Works with both Redis and DiskCache backends.
        """
        self.r.hset(key, mapping=mapping)

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def count_active_tokens(self, server_key: str) -> int:
        """Return the number of currently active (non-expired) tokens for *server_key*.

        Args:
            server_key: Logical server name.

        Returns:
            Count of active tokens.
        """
        return len(self.get_all_tokens(server_key, include_expired=False))

    # ------------------------------------------------------------------
    # Token generation
    # ------------------------------------------------------------------

    def generate_token(
        self,
        server_key: str,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        ttl_seconds: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate a new API token and persist it in the store.

        Args:
            server_key:   Logical name of the server this token is scoped to.
            user_id:      Optional user identifier; e-mail domain is stripped
                          automatically.  Defaults to ``"anonymous"``.
            role_id:      Optional role identifier.  Defaults to ``"default"``.
            ttl_seconds:  Lifetime in seconds.  ``None`` creates a permanent
                          token.
            max_tokens:   Optional hard cap on active tokens per server.  When
                          set and the current count is already at the cap, a
                          ``ValueError`` is raised instead of generating a new
                          token.  Pass ``None`` to disable the check.

        Returns:
            Token string: ``<43-char random>.<base64url metadata>``

        Raises:
            ValueError: if *max_tokens* is set and the limit has been reached.
        """
        if max_tokens is not None:
            current_count = self.count_active_tokens(server_key)
            if current_count >= max_tokens:
                raise ValueError(
                    f"Token limit reached: {current_count}/{max_tokens} active tokens already "
                    f"exist for '{server_key}'. Revoke existing tokens before generating new ones."
                )

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
