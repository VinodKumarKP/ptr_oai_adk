
import pytest
from unittest.mock import MagicMock, patch
from oai_platform_core.security.token_manager import TokenManager
import time

@pytest.fixture
def token_manager():
    with patch('redis.Redis') as mock_redis:
        tm = TokenManager()
        tm.r = mock_redis.return_value
        return tm

def test_generate_token(token_manager):
    token_manager.r.pipeline.return_value.execute.return_value = []
    token = token_manager.generate_token("server1", "user1")
    assert "." in token
    token_manager.r.hset.assert_called()

def test_generate_token_with_ttl(token_manager):
    token = token_manager.generate_token("server1", "user1", ttl_seconds=60)
    token_manager.r.expire.assert_called()

def test_validate_token_valid(token_manager):
    token = "random.metadata"
    user_info = {"user_id": "anonymous", "role_id": "default"}
    token_manager.parse_token = MagicMock(return_value={"server": "server1", **user_info})
    token_manager.r.exists.return_value = True
    
    result = token_manager.validate_token("server1", token)
    assert result == user_info

def test_validate_token_invalid_server(token_manager):
    token = "random.metadata"
    token_manager.parse_token = MagicMock(return_value={"server": "server2"})
    
    assert token_manager.validate_token("server1", token) is None

def test_revoke_token(token_manager):
    token = "random.metadata"
    token_manager.parse_token = MagicMock(return_value={"server": "s1", "user_id": "u1"})
    token_manager.r.srem.return_value = 1
    token_manager.r.zrem.return_value = 0
    
    assert token_manager.revoke_token(token) is True
    token_manager.r.delete.assert_called()

def test_get_all_tokens(token_manager):
    token_manager.r.smembers.return_value = {"t1"}
    token_manager.r.zrange.return_value = [("t2", time.time() + 100)]
    token_manager.parse_token = MagicMock(side_effect=[{"user_id": "u1"}, {"user_id": "u1"}])
    token_manager.get_token_info = MagicMock(return_value={})
    
    tokens = token_manager.get_all_tokens("s1")
    assert len(tokens) == 2
    assert tokens[0]['type'] == 'permanent'
    assert tokens[1]['type'] == 'ttl'

def test_revoke_all_tokens(token_manager):
    # revoke_all_tokens delegates to revoke_tokens_by_server which iterates
    # get_all_tokens and calls revoke_token for each entry.
    token_manager.get_all_tokens = MagicMock(return_value=[
        {"token": "t1"}, {"token": "t2"}
    ])
    token_manager.revoke_token = MagicMock(return_value=True)
    assert token_manager.revoke_all_tokens("s1") == 2

def test_get_tokens_by_user(token_manager):
    token_manager.get_all_tokens = MagicMock(return_value=[])
    token_manager.get_tokens_by_user("s1", "u1")
    token_manager.get_all_tokens.assert_called_with("s1:u1", False)

def test_revoke_tokens_by_user(token_manager):
    token_manager.get_all_tokens = MagicMock(return_value=[{"token": "t1"}])
    token_manager.revoke_token = MagicMock(return_value=True)
    assert token_manager.revoke_tokens_by_user("s1", "u1") == 1

def test_parse_token_invalid(token_manager):
    assert token_manager.parse_token("invalid") is None


# --- Email-domain stripping in user_id ---
def test_generate_token_strips_email_domain(token_manager):
    token = token_manager.generate_token("server1", "alice@example.com", ttl_seconds=60)
    parsed = token_manager.parse_token(token)
    assert parsed["user_id"] == "alice"


def test_generate_token_anonymous_when_no_user(token_manager):
    token = token_manager.generate_token("server1", None)
    parsed = token_manager.parse_token(token)
    assert parsed["user_id"] == "anonymous"
    assert parsed["role_id"] == "default"


# --- parse_token round-trip ---
def test_parse_token_round_trip(token_manager):
    token = token_manager.generate_token("srv-name", "u1", role_id="admin")
    parsed = token_manager.parse_token(token)
    assert parsed["server"] == "srv-name"
    assert parsed["user_id"] == "u1"
    assert parsed["role_id"] == "admin"
    assert len(parsed["random_part"]) == 43


def test_parse_token_too_short_returns_none(token_manager):
    assert token_manager.parse_token("short") is None


def test_parse_token_no_dot_at_position_43_returns_none(token_manager):
    # 44 chars but no '.' at index 43
    bad = "x" * 44
    assert token_manager.parse_token(bad) is None


def test_parse_token_bad_base64_returns_none(token_manager):
    bad = ("a" * 43) + "." + "!!!not-base64!!!"
    assert token_manager.parse_token(bad) is None


# --- validate_token legacy paths ---
def test_validate_token_legacy_permanent(token_manager):
    token = "x" * 43 + ".meta"
    token_manager.parse_token = MagicMock(return_value={
        "server": "s1", "user_id": "u1", "role_id": "default"
    })
    token_manager.r.exists.return_value = False
    token_manager.r.sismember.return_value = True

    result = token_manager.validate_token("s1", token)
    assert result == {"user_id": "u1", "role_id": "default"}


def test_validate_token_legacy_ttl_active(token_manager):
    token = "x" * 43 + ".meta"
    token_manager.parse_token = MagicMock(return_value={
        "server": "s1", "user_id": "u1", "role_id": "r1"
    })
    token_manager.r.exists.return_value = False
    token_manager.r.sismember.return_value = False
    token_manager.r.zscore.return_value = float(time.time() + 1000)

    result = token_manager.validate_token("s1", token)
    assert result == {"user_id": "u1", "role_id": "r1"}


def test_validate_token_legacy_ttl_expired_removed(token_manager):
    token = "x" * 43 + ".meta"
    token_manager.parse_token = MagicMock(return_value={
        "server": "s1", "user_id": "u1"
    })
    token_manager.r.exists.return_value = False
    token_manager.r.sismember.return_value = False
    token_manager.r.zscore.return_value = float(time.time() - 1000)

    result = token_manager.validate_token("s1", token)
    assert result is None
    token_manager.r.zrem.assert_any_call("s1:u1:ttl", token)
    token_manager.r.zrem.assert_any_call("s1:ttl", token)


def test_validate_token_not_found(token_manager):
    token = "x" * 43 + ".meta"
    token_manager.parse_token = MagicMock(return_value={"server": "s1", "user_id": "u1"})
    token_manager.r.exists.return_value = False
    token_manager.r.sismember.return_value = False
    token_manager.r.zscore.return_value = None

    assert token_manager.validate_token("s1", token) is None


def test_validate_token_malformed_returns_none(token_manager):
    token_manager.parse_token = MagicMock(return_value=None)
    assert token_manager.validate_token("s1", "garbage") is None


# --- revoke_token guard ---
def test_revoke_token_unparseable_returns_false(token_manager):
    token_manager.parse_token = MagicMock(return_value=None)
    assert token_manager.revoke_token("garbage") is False


# --- cleanup_expired_tokens ---
def test_cleanup_expired_tokens(token_manager):
    token_manager.r.zremrangebyscore.return_value = 3
    removed = token_manager.cleanup_expired_tokens("s1")
    assert removed == 3
    token_manager.r.zremrangebyscore.assert_called_once()


# --- get_tokens_by_server delegates to get_all_tokens ---
def test_get_tokens_by_server(token_manager):
    token_manager.get_all_tokens = MagicMock(return_value=[])
    token_manager.get_tokens_by_server("s1")
    token_manager.get_all_tokens.assert_called_with("s1", False)


# --- revoke_tokens_by_server ---
def test_revoke_tokens_by_server(token_manager):
    token_manager.get_all_tokens = MagicMock(return_value=[
        {"token": "t1"}, {"token": "t2"}
    ])
    token_manager.revoke_token = MagicMock(side_effect=[True, False])
    assert token_manager.revoke_tokens_by_server("s1") == 1


# --- get_token_info ---
def test_get_token_info_returns_hash_data(token_manager):
    # TokenManager connects with decode_responses=True, so Redis returns
    # plain str keys/values — the mock should match that behaviour.
    token_manager.r.hgetall.return_value = {"server_name": "s1", "user_id": "u1"}
    result = token_manager.get_token_info("some.token")
    assert result == {"server_name": "s1", "user_id": "u1"}
    token_manager.r.hgetall.assert_called_with("tokens:some.token")


def test_get_token_info_returns_none_when_empty(token_manager):
    token_manager.r.hgetall.return_value = {}
    assert token_manager.get_token_info("missing.token") is None


# --- _connect fallback paths ---

def test_connect_falls_back_to_redislite_when_redis_unavailable(tmp_path):
    """When Redis ping fails, _connect falls back to redislite."""
    import sys

    fake_rl_client = MagicMock()
    fake_rl_module = MagicMock()
    fake_rl_module.Redis.return_value = fake_rl_client

    mock_redis_client = MagicMock()
    mock_redis_client.ping.side_effect = ConnectionError("refused")
    mock_redis_module = MagicMock()
    mock_redis_module.Redis.return_value = mock_redis_client

    db_path = str(tmp_path / "tokens.db")
    with patch.dict(sys.modules, {"redis": mock_redis_module, "redislite": fake_rl_module}):
        client, backend = TokenManager._connect("localhost", 9999, 0, db_path, "t.db")

    assert backend == "redislite"
    fake_rl_module.Redis.assert_called_once_with(db_path, decode_responses=True)


def test_connect_raises_when_redis_unavailable_and_no_redislite():
    """When both Redis and redislite are unavailable a RuntimeError is raised."""
    import sys

    mock_redis_client = MagicMock()
    mock_redis_client.ping.side_effect = ConnectionError("refused")
    mock_redis_module = MagicMock()
    mock_redis_module.Redis.return_value = mock_redis_client

    with patch.dict(sys.modules, {"redis": mock_redis_module, "redislite": None}):
        with pytest.raises((RuntimeError, ImportError)):
            TokenManager._connect("localhost", 9999, 0, None, "t.db")


def test_connect_uses_redislite_db_path_env(tmp_path, monkeypatch):
    """REDISLITE_DB_PATH env var is used when no explicit path is given."""
    import sys

    db_env = str(tmp_path / "env.db")
    monkeypatch.setenv("REDISLITE_DB_PATH", db_env)

    fake_rl_module = MagicMock()
    mock_redis_module = MagicMock()
    mock_redis_module.Redis.return_value.ping.side_effect = ConnectionError("x")

    with patch.dict(sys.modules, {"redis": mock_redis_module, "redislite": fake_rl_module}):
        client, backend = TokenManager._connect("localhost", 9999, 0, None, "t.db")

    fake_rl_module.Redis.assert_called_once_with(db_env, decode_responses=True)


# --- _hset_mapping fallback ---

def test_hset_mapping_falls_back_to_hmset(token_manager):
    """If hset(mapping=…) raises TypeError, hmset is used instead."""
    token_manager.r.hset.side_effect = TypeError("no mapping kwarg")
    token_manager.r.hmset = MagicMock()
    token_manager._hset_mapping("mykey", {"a": "1"})
    token_manager.r.hmset.assert_called_once_with("mykey", {"a": "1"})


def test_hset_mapping_uses_hset_when_available(token_manager):
    token_manager.r.hset.side_effect = None
    token_manager._hset_mapping("mykey", {"a": "1"})
    token_manager.r.hset.assert_called_once_with("mykey", mapping={"a": "1"})
