import os
import sys
from unittest.mock import MagicMock, AsyncMock

import pytest

# Add src to sys.path to ensure local packages are discoverable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture
def mock_agent_class():
    """Creates a mock Agent class that behaves like OpenAIAgent."""
    mock_cls = MagicMock()

    # Mock instance
    mock_instance = AsyncMock()
    mock_instance.initialize = AsyncMock()

    # Mock ainvoke response
    mock_instance.ainvoke.return_value = {
        'content': [{'text': 'Mock response'}],
        'model': {'model_id': 'mock-model'}
    }

    mock_cls.return_value = mock_instance
    return mock_cls


@pytest.fixture
def mock_judge_response():
    """Standard mock response for the judge agent."""
    return {
        'content': [{
            'text': '{"correctness": {"score": 10, "explanation": "Perfect match"}, "relevance": {"score": 10, "explanation": "Relevant"}}'
        }]
    }
