import os

import sys
from unittest.mock import MagicMock

# Add src to sys.path to ensure local packages are discoverable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Mock external dependencies that might not be installed in the test environment
sys.modules['agents'] = MagicMock()
sys.modules['agents.mcp'] = MagicMock()
sys.modules['agents.extensions.models.litellm_model'] = MagicMock()
