import pytest
from unittest.mock import MagicMock, patch
from oai_agent_server.utils.token_manager import TokenManager
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
    token_manager.parse_token = MagicMock(return_value={"server": "server1"})
    token_manager.r.hgetall.return_value = {"server_name": "server1"}
    token_manager.r.ttl.return_value = 100
    
    assert token_manager.validate_token("server1", token) is True

def test_validate_token_invalid_server(token_manager):
    token = "random.metadata"
    token_manager.parse_token = MagicMock(return_value={"server": "server2"})
    token_manager.r.hgetall.return_value = {"server_name": "server2"}
    
    assert token_manager.validate_token("server1", token) is False


def test_revoke_token(token_manager):
    token = "random.metadata"
    token_manager.parse_token = MagicMock(return_value={"server": "s1", "user_id": "u1"})
    # srem returns int (number of removed items)
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
    token_manager.r.delete.return_value = 2
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
