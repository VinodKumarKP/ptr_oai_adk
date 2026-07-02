"""Additional coverage for TokenManager.

TokenManager stores tokens via self.r (a Redis-compatible client).
We mock self.r after construction to avoid needing real Redis/DiskCache.
"""
from unittest.mock import MagicMock, patch

import pytest

from oai_platform_core.security.token_manager import TokenManager


@pytest.fixture
def token_manager():
    """Create a TokenManager with a fully mocked Redis client."""
    # Patch _connect so no real Redis/DiskCache is attempted
    with patch.object(
        TokenManager,
        "_connect",
        return_value=(MagicMock(), "redis"),
    ):
        manager = TokenManager()
    # Replace the client with a clean MagicMock for each test
    manager.r = MagicMock()
    return manager


# ---------------------------------------------------------------------------
# generate_token
# ---------------------------------------------------------------------------

def test_generate_token_basic(token_manager):
    """generate_token should produce a non-empty token string."""
    # r.exists returns 0 so it goes through the code path that stores
    token_manager.r.exists.return_value = 0
    token_manager.r.hset.return_value = 1
    token_manager.r.sadd.return_value = 1
    token_manager.r.zadd.return_value = 1

    token = token_manager.generate_token("my_server", user_id="user1")
    assert isinstance(token, str)
    assert len(token) > 0
    assert "." in token  # format: <random>.<b64_metadata>


def test_generate_token_with_ttl(token_manager):
    """generate_token with TTL should produce a valid token."""
    token_manager.r.zcard.return_value = 0
    token = token_manager.generate_token("srv", user_id="user@example.com", ttl_seconds=3600)
    assert token is not None
    # email domain should be stripped — check by parsing
    parsed = token_manager.parse_token(token)
    assert parsed["user_id"] == "user"


def test_generate_token_max_tokens_exceeded(token_manager):
    """generate_token should raise ValueError when token limit is reached."""
    # count_active_tokens calls get_all_tokens → smembers + zrange
    # Return 5 permanent tokens so count_active_tokens() == 5
    token_manager.r.smembers.return_value = {"tok1", "tok2", "tok3", "tok4", "tok5"}
    token_manager.r.zrange.return_value = []  # no TTL tokens
    # All parse_token calls on fake tokens will return None → entries still counted
    with pytest.raises(ValueError, match="Token limit reached"):
        token_manager.generate_token("srv", max_tokens=3)


# ---------------------------------------------------------------------------
# validate_token
# ---------------------------------------------------------------------------

def test_validate_token_valid(token_manager):
    """validate_token with a valid current-format token should return user context."""
    # Generate a real token so parse_token works correctly
    token_manager.r.hset.return_value = 1
    token_manager.r.sadd.return_value = 1
    token = token_manager.generate_token("srv", user_id="alice", role_id="admin")

    # Token metadata hash exists → modern token path
    token_manager.r.exists.return_value = 1
    pipeline_mock = MagicMock()
    pipeline_mock.execute.return_value = [1, 1]
    token_manager.r.pipeline.return_value = pipeline_mock

    result = token_manager.validate_token("srv", token)
    assert result is not None
    assert result["user_id"] == "alice"
    assert result["role_id"] == "admin"


def test_validate_token_wrong_server(token_manager):
    """validate_token with wrong server_key should return None."""
    token_manager.r.hset.return_value = 1
    token = token_manager.generate_token("server_a", user_id="bob")

    result = token_manager.validate_token("server_b", token)
    assert result is None


def test_validate_token_invalid_token(token_manager):
    """validate_token with a garbage token should return None."""
    result = token_manager.validate_token("srv", "not-a-valid-token")
    assert result is None


def test_validate_token_missing_from_store(token_manager):
    """validate_token when token not in any store should return None."""
    token = token_manager.generate_token("srv", user_id="ghost")
    # Token not in hash store, not in permanent set, not in TTL zset
    token_manager.r.exists.return_value = 0
    token_manager.r.sismember.return_value = 0
    token_manager.r.zscore.return_value = None

    result = token_manager.validate_token("srv", token)
    assert result is None


# ---------------------------------------------------------------------------
# revoke_token
# ---------------------------------------------------------------------------

def test_revoke_token_success(token_manager):
    """revoke_token should return True when token was found and removed."""
    token = token_manager.generate_token("srv", user_id="eve")
    token_manager.r.srem.return_value = 1
    token_manager.r.zrem.return_value = 0
    token_manager.r.delete.return_value = 1

    result = token_manager.revoke_token(token)
    assert result is True


def test_revoke_token_not_found(token_manager):
    """revoke_token should return False for garbage/missing tokens."""
    result = token_manager.revoke_token("invalid-token-garbage")
    assert result is False


# ---------------------------------------------------------------------------
# get_token_info
# ---------------------------------------------------------------------------

def test_get_token_info_existing(token_manager):
    """get_token_info should return metadata dict when token metadata exists."""
    token = token_manager.generate_token("srv", user_id="carol")
    token_manager.r.hgetall.return_value = {
        "user_id": "carol",
        "server": "srv",
        "created_at": "2024-01-01T00:00:00",
    }
    result = token_manager.get_token_info(token)
    assert result is not None
    assert "user_id" in result or "server" in result


def test_get_token_info_missing(token_manager):
    """get_token_info should return None when token metadata hash is empty."""
    token = token_manager.generate_token("srv", user_id="dave")
    token_manager.r.hgetall.return_value = {}
    result = token_manager.get_token_info(token)
    assert result is None


# ---------------------------------------------------------------------------
# count_active_tokens
# ---------------------------------------------------------------------------

def test_count_active_tokens(token_manager):
    """count_active_tokens returns the count of permanent + non-expired TTL tokens."""
    import time
    future_expiry = int(time.time()) + 3600

    # Generate 3 real tokens so smembers returns parseable token strings
    token_manager.r.hset.return_value = 1
    token_manager.r.sadd.return_value = 1
    t1 = token_manager.generate_token("srv", user_id="u1")
    t2 = token_manager.generate_token("srv", user_id="u2")
    t3 = token_manager.generate_token("srv", user_id="u3")

    # Fake the redis calls get_all_tokens makes
    token_manager.r.smembers.return_value = {t1, t2, t3}  # 3 permanent
    token_manager.r.zrange.return_value = []  # no TTL tokens

    count = token_manager.count_active_tokens("srv")
    assert count == 3


# ---------------------------------------------------------------------------
# parse_token
# ---------------------------------------------------------------------------

def test_parse_token_valid(token_manager):
    """parse_token should decode metadata from a generated token."""
    token = token_manager.generate_token("srv", user_id="frank", role_id="viewer")
    parsed = token_manager.parse_token(token)
    assert parsed is not None
    assert parsed["server"] == "srv"
    assert parsed["user_id"] == "frank"
    assert parsed["role_id"] == "viewer"


def test_parse_token_invalid(token_manager):
    """parse_token should return None for malformed input."""
    assert token_manager.parse_token("notvalid") is None
    assert token_manager.parse_token("") is None
    assert token_manager.parse_token("a.b.c.d.e") is None
