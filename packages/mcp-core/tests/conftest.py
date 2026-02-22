import os
import sys
import pytest
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add src to sys.path to ensure local packages are discoverable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture
def mock_env():
    """Mock environment variables."""
    with pytest.MonkeyPatch.context() as m:
        yield m

@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    with pytest.MonkeyPatch.context() as m:
        mock_redis_client = MagicMock()
        m.setattr("redis.Redis", MagicMock(return_value=mock_redis_client))
        yield mock_redis_client
