import base64
import json
import os
import secrets
import time
from datetime import datetime
from typing import Optional, Dict, Any

import redis


class TokenManager:
    """
    API Token Management System using Redis.
    Supports multiple tokens per server with optional user ID and TTL.
    Token format: <random_token>.<metadata>
    """

    def __init__(self,
                 host='localhost',
                 port=6379, db=0):
        """Initialize Redis connection."""
        redis_host = os.environ.get('REDIS_HOST', host)
        redis_port = int(os.environ.get('REDIS_PORT', port))
        self.r = redis.Redis(host=redis_host, port=redis_port, db=db, decode_responses=True)

    def generate_token(self, server_key: str,
                       user_id: Optional[str] = None,
                       role_id: Optional[str] = None,
                       ttl_seconds: Optional[int] = None) -> str:
        """
        Generate a token with embedded metadata.

        Args:
            server_key: Name of the MCP server
            user_id: Optional user identifier
            role_id: Optional role identifier
            ttl_seconds: Time-to-live in seconds (None for permanent tokens)

        Returns:
            Token in format: <random_part>.<base64_metadata>
        """
        # Generate random token part (43 chars for 32 bytes)
        random_part = secrets.token_urlsafe(32)

        # Create metadata
        metadata = {
            "server": server_key,
        }
        if user_id is not None:
            #Remove email domain if present
            if "@" in user_id:
                user_id = user_id.split("@")[0]
        else:
            user_id = "anonymous"
        
        if role_id is None:
            role_id = "default"
        
        metadata["user_id"] = user_id
        metadata["role_id"] = role_id

        # Encode metadata as base64 (URL-safe, no padding)
        metadata_json = json.dumps(metadata, separators=(',', ':'))
        metadata_b64 = base64.urlsafe_b64encode(metadata_json.encode()).decode().rstrip('=')

        # Combine: random_part (always 43 chars).metadata
        # Using dot as separator - we know the random part is exactly 43 characters
        token = f"{random_part}.{metadata_b64}"

        # Store in Redis
        if ttl_seconds is not None:
            expiry_timestamp = int(time.time()) + ttl_seconds
            key = f"{server_key}:{user_id}:ttl"
            self.r.zadd(key, {token: expiry_timestamp})
            self.r.zadd(f"{server_key}:ttl", {token: expiry_timestamp})
        else:
            key = f"{server_key}:{user_id}:permanent"
            self.r.sadd(key, token)
            self.r.sadd(f"{server_key}:permanent", token)

        token_data = {
            'server_name': server_key,
            'user_id': user_id,
            'role_id': role_id,
            'created_at': datetime.utcnow().isoformat(),
            'last_accessed': datetime.utcnow().isoformat(),
            'access_count': '0',
            'ttl': str(ttl_seconds) if ttl_seconds is not None else 'none'
        }


        self.r.hset(f"tokens:{token}", mapping=token_data)

        if ttl_seconds is not None:
            self.r.expire(f"tokens:{token}", ttl_seconds)

        return token

    def parse_token(self, token: str) -> Optional[Dict]:
        """
        Parse token to extract metadata.

        Args:
            token: Token string

        Returns:
            Dictionary with server_key, user_id (if present), and random_part
        """
        try:
            # token_urlsafe(32) always produces 43 characters
            # Format: <43_char_random>.<metadata>
            if len(token) < 44 or token[43] != '.':
                return None

            random_part = token[:43]
            metadata_b64 = token[44:]

            # Add padding if needed
            padding = 4 - (len(metadata_b64) % 4)
            if padding != 4:
                metadata_b64 += '=' * padding

            # Decode metadata
            metadata_json = base64.urlsafe_b64decode(metadata_b64).decode()
            metadata = json.loads(metadata_json)

            result = {
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
        """
        Check if a token is valid for the given server key and return user info.
        This function handles both modern (with hash) and legacy (without hash) tokens.
        It automatically removes expired tokens upon discovery.

        Args:
            server_key: Name of the MCP server.
            token: Token to validate.

        Returns:
            A dictionary with 'user_id' and 'role_id' if the token is valid, otherwise None.
        """
        # Step 1: Parse the token to verify its format and extract claims.
        parsed_data = self.parse_token(token)
        if not parsed_data:
            return None  # Token is malformed.

        # Step 2: Verify the server key from the token's claims.
        if parsed_data.get("server") != server_key:
            return None  # Token is for a different server.

        # Step 3: Check for the modern token hash first for efficient validation.
        if self.r.exists(f"tokens:{token}"):
            # Token hash exists, it's a valid modern token.
            # Update access metadata.
            pipe = self.r.pipeline()
            pipe.hincrby(f"tokens:{token}", 'access_count', 1)
            pipe.hset(f"tokens:{token}", 'last_accessed', datetime.utcnow().isoformat())
            pipe.execute()
            # Return user info from parsed token
            return {
                "user_id": parsed_data.get("user_id", "anonymous"),
                "role_id": parsed_data.get("role_id", "default")
            }

        # Step 4: If hash doesn't exist, check legacy sets (permanent and TTL).
        user_id = parsed_data.get('user_id', 'anonymous')

        # Check permanent token set.
        permanent_key = f"{server_key}:{user_id}:permanent"
        if self.r.sismember(permanent_key, token):
            # It's a valid legacy permanent token.
            return {
                "user_id": parsed_data.get("user_id", "anonymous"),
                "role_id": parsed_data.get("role_id", "default")
            }

        # Check TTL token sorted set.
        ttl_key = f"{server_key}:{user_id}:ttl"
        score = self.r.zscore(ttl_key, token)

        if score is not None:
            # Token found in TTL set, check if it's expired.
            if int(score) < int(time.time()):
                # Token is expired, remove it.
                self.r.zrem(ttl_key, token)
                self.r.zrem(f"{server_key}:ttl", token)  # Also remove from server-wide set
                return None
            else:
                # Token is not expired.
                return {
                    "user_id": parsed_data.get("user_id", "anonymous"),
                    "role_id": parsed_data.get("role_id", "default")
                }

        # Step 5: If token is not found anywhere, it's invalid.
        return None

    def revoke_token(self, token: str) -> bool:
        """
        Revoke a specific token.
        Automatically determines the server from token metadata.

        Args:
            token: Token to revoke

        Returns:
            True if token was found and removed, False otherwise
        """
        parsed = self.parse_token(token)
        if not parsed:
            return False

        server_key = parsed["server"]
        user_id = parsed.get("user_id")
        permanent_key = f"{server_key}:{user_id}:permanent"
        ttl_key = f"{server_key}:{user_id}:ttl"

        removed = self.r.srem(permanent_key, token)
        removed += self.r.zrem(ttl_key, token)

        self.r.srem(f"{server_key}:permanent", token)
        self.r.zrem(f"{server_key}:ttl", token)

        self.r.delete(f"tokens:{token}")

        return removed > 0

    def cleanup_expired_tokens(self, server_key: str) -> int:
        """
        Remove all expired tokens for a server key.

        Args:
            server_key: Name of the MCP server

        Returns:
            Number of tokens removed
        """
        ttl_key = f"{server_key}:tokens:ttl"
        current_time = int(time.time())
        return self.r.zremrangebyscore(ttl_key, 0, current_time)

    def get_all_tokens(self, token_key: str, include_expired: bool = False) -> list[Any]:
        """
        Get all tokens for a server key with their metadata.

        Args:
            server_key: Name of the MCP server
            include_expired: Whether to include expired tokens

        Returns:
            Dictionary with 'permanent' and 'ttl' token lists
        """
        permanent_key = f"{token_key}:permanent"
        ttl_key = f"{token_key}:ttl"

        # Get permanent tokens
        permanent_tokens = []
        for token in self.r.smembers(permanent_key):
            parsed = self.parse_token(token)
            token_info = {
                "token": token,
                "type": "permanent",
            }
            if parsed and "user_id" in parsed:
                token_info["user_id"] = parsed["user_id"]
            permanent_tokens.append(token_info)

        # Get TTL tokens
        current_time = int(time.time())
        ttl_tokens_raw = self.r.zrange(ttl_key, 0, -1, withscores=True)

        ttl_tokens = []
        for token, expiry in ttl_tokens_raw:
            is_expired = expiry <= current_time
            if include_expired or not is_expired:
                parsed = self.parse_token(token)
                token_info = self.get_token_info(token)
                token_info = {
                    "token": token,
                    "type": "ttl",
                    "expires_at": int(expiry),
                    "expires_in": max(0, int(expiry - current_time)),
                    "is_expired": is_expired,
                    "access_count": token_info.get('access_count', '0'),
                    "last_accessed": token_info.get('last_accessed', None),
                    "created_at": token_info.get('created_at', None)
                }
                if parsed and "user_id" in parsed:
                    token_info["user_id"] = parsed["user_id"]
                ttl_tokens.append(token_info)

        token_list = []
        token_list.extend(permanent_tokens)
        token_list.extend(ttl_tokens)

        return token_list

    def revoke_all_tokens(self, server_key: str) -> int:
        """
        Revoke all tokens for a server key.

        Args:
            server_key: Name of the MCP server

        Returns:
            Number of tokens removed
        """
        permanent_key = f"{server_key}:tokens:permanent"
        ttl_key = f"{server_key}:tokens:ttl"

        count = self.r.delete(permanent_key, ttl_key)
        return count

    def get_tokens_by_user(self, server_key: str, user_id: str,
                           include_expired: bool = False) -> list[Any]:
        """
        Get all tokens for a specific user on a server.

        Args:
            server_key: Name of the MCP server
            user_id: User identifier
            include_expired: Whether to include expired tokens

        Returns:
            Dictionary with 'permanent' and 'ttl' token lists for the user
        """

        token_key = f"{server_key}:{user_id}"
        return self.get_all_tokens(token_key, include_expired)

    def get_tokens_by_server(self, server_key: str,
                             include_expired: bool = False) -> list[Any]:
        """
        Get all tokens for a specific user on a server.

        Args:
            server_key: Name of the MCP server
            user_id: User identifier
            include_expired: Whether to include expired tokens

        Returns:
            Dictionary with 'permanent' and 'ttl' token lists for the user
        """

        token_key = f"{server_key}"
        return self.get_all_tokens(token_key, include_expired)

    def revoke_tokens_by_user(self, server_key: str, user_id: str) -> int:
        """
        Revoke all tokens for a specific user on a server.

        Args:
            server_key: Name of the MCP server
            user_id: User identifier

        Returns:
            Number of tokens removed
        """
        token_key = f"{server_key}:{user_id}"
        all_tokens = self.get_all_tokens(token_key, include_expired=True)

        count = 0
        for token_info in all_tokens:
            if self.revoke_token(token_info["token"]):
                count += 1

        return count

    def revoke_tokens_by_server(self, server_key: str) -> int:
        """
        Revoke all tokens for a specific server.

        Args:
            server_key: Name of the MCP server

        Returns:
            Number of tokens removed
        """
        all_tokens = self.get_all_tokens(server_key, include_expired=True)

        count = 0
        for token_info in all_tokens:
            if self.revoke_token(token_info["token"]):
                count += 1

        return count

    def get_token_info(self, token: str) -> Optional[Dict]:
        """
        Get information about a specific token.

        Args:
            token: Token to query

        Returns:
            Token info dict or None if not found
        """
        data = self.r.hgetall(f"tokens:{token}")
        return data
