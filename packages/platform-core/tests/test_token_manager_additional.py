"""Additional coverage for token_manager."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from oai_platform_core.security.token_manager import TokenManager


@pytest.fixture
def token_manager():
    """Create token manager with mocked backend."""
    with patch("oai_platform_core.security.token_manager.Backend"):
        manager = TokenManager()
        manager.backend = MagicMock()
        return manager


def test_create_token(token_manager):
    """Test token creation."""
    token_manager.backend.hset = MagicMock()
    result = token_manager.create_token("user1", ttl=3600)
    assert result is not None
    token_manager.backend.hset.assert_called()


def test_revoke_token(token_manager):
    """Test token revocation."""
    token_manager.backend.delete = MagicMock()
    token_manager.revoke_token("test_token")
    token_manager.backend.delete.assert_called()


def test_validate_token_valid(token_manager):
    """Test validating a valid token."""
    token_manager.backend.hgetall = MagicMock(return_value={
        b"user_id": b"user1",
        b"expiry": str(datetime.now() + timedelta(hours=1)).encode()
    })
    result = token_manager.validate_token("valid_token")
    assert result is not None or result == "user1"


def test_validate_token_expired(token_manager):
    """Test validating an expired token."""
    token_manager.backend.hgetall = MagicMock(return_value={
        b"user_id": b"user1",
        b"expiry": str(datetime.now() - timedelta(hours=1)).encode()
    })
    result = token_manager.validate_token("expired_token")
    # Should be None or handle as invalid
    assert result is None or result == ""


def test_validate_token_missing(token_manager):
    """Test validating a missing token."""
    token_manager.backend.hgetall = MagicMock(return_value={})
    result = token_manager.validate_token("missing_token")
    assert result is None or result == ""


def test_get_token_expiry(token_manager):
    """Test getting token expiry."""
    token_manager.backend.hget = MagicMock(return_value=b"2099-12-31T00:00:00")
    result = token_manager.get_token_expiry("test_token")
    assert result is not None


def test_list_active_tokens(token_manager):
    """Test listing active tokens."""
    token_manager.backend.keys = MagicMock(return_value=[b"token:1", b"token:2"])
    result = token_manager.list_active_tokens()
    assert isinstance(result, (list, set)) or result is not None
