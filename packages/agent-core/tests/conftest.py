import os
import sys
from typing import Any

import pytest
import logging
from unittest.mock import MagicMock

# ── Global module mocks ──────────────────────────────────────────────────────
# These must be in place before ANY test module imports packages that depend on
# optional third-party libraries.  Putting them here (conftest.py) ensures the
# mocks are applied regardless of test collection order.

# guardrails / guardrails-hub
if 'guardrails' not in sys.modules:
    sys.modules['guardrails'] = MagicMock()
if 'guardrails.hub' not in sys.modules:
    sys.modules['guardrails.hub'] = MagicMock()

# langfuse (optional observability dependency)
if 'langfuse' not in sys.modules:
    sys.modules['langfuse'] = MagicMock()
if 'langfuse.callback' not in sys.modules:
    sys.modules['langfuse.callback'] = MagicMock()

# Add src to sys.path to ensure local packages are discoverable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

@pytest.fixture
def mock_logger():
    return MagicMock(spec=logging.Logger)

# Define Mock Classes
class MockBaseToolRegistry:
    def __init__(self, *args, **kwargs):
        self.logger = MagicMock()
        self.tools = {}
        self.mcp_configs = {}
        self.mcp_clients = []
        self.project_root = "/tmp"
    
    def _get_mcp_name_list_from_mcp_config(self, config):
        return list(config.keys()) if config else []
    
    def _get_mcp_config(self, config, name):
        return config.get(name, {})
    
    def get_tools_for_agent(self, tool_names):
        return []
        
    def load_tools_from_config(self, config):
        pass
        
    def has_tool(self, tool_name):
        return True
        
    def get_mcp_configs(self):
        return {}
        
    def load_mcp_config(self, mcp_config):
        pass

    def get_input_parameter_schema(self, tool_list: str) -> str:
        pass

    async def execute_tool(self, tool_name: str, arguments: Any) -> Any:
        pass

class MockBaseKnowledgeBaseFactory:
    def __init__(self, knowledge_base_config=None, logger=None, **kwargs):
        self.knowledge_base_config = knowledge_base_config
        self.logger = logger or MagicMock()
    
    def search_custom_knowledge_base(self, query):
        return "Mock Search Result"
        
    def create_tool(self):
        return MagicMock(name="kb_tool")

class MockBaseAgent:
    def __init__(self, llm=None, agent_name="", agent_config=None, session_id="default", config_root=None, agent_type="crewai", user_id="default", document_loader=None, vector_store=None, **kwargs):
        self.llm = llm
        self.agent_name = agent_name
        self.agent_config = agent_config or {}
        self.session_id = session_id
        self.config_root = config_root
        self.agent_type = agent_type
        self.user_id = user_id
        self.document_loader = document_loader
        self.vector_store = vector_store
        self.logger = MagicMock()
        self.langfuse_manager = MagicMock()
        self.langfuse_manager.callback_handler = None
        self.langfuse_manager.is_enabled = False
        self._initialized = False
        
    async def _ensure_initialized(self):
        pass

class MockBaseModelConfigurationManager:
    def __init__(self):
        self.logger = MagicMock()

    def create_model(self, config):
        return MagicMock()
        
    def _merge_with_defaults(self, config):
        return config
        
    def _validate_config(self, config):
        pass
        
    @property
    def default_config(self):
        return {}

class MockBaseVectorStore:
    def __init__(self, collection_name=None, embedding_function=None, persist_directory=None, **kwargs):
        self.collection_name = collection_name
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.logger = MagicMock()

class MockBaseDocumentLoader:
    def __init__(self, db_name=None, vector_store=None, embedding=None, persist_directory=None, **kwargs):
        self.db_name = db_name
        self.vector_store = vector_store
        self.embedding = embedding
        self.persist_directory = persist_directory
        self.logger = MagicMock()
        self.reinitialize = False

class MockConstants:
    CREWAI = "crewai"
    REMOTE = "remote"
    LANGCHAIN = "langchain"
    BEDROCK = "bedrock"

# Create mock modules
base_tool_registry_module = MagicMock()
base_tool_registry_module.BaseToolRegistry = MockBaseToolRegistry

base_kb_factory_module = MagicMock()
base_kb_factory_module.BaseKnowledgeBaseFactory = MockBaseKnowledgeBaseFactory

base_agent_module = MagicMock()
base_agent_module.BaseAgent = MockBaseAgent

base_model_config_manager_module = MagicMock()
base_model_config_manager_module.BaseModelConfigurationManager = MockBaseModelConfigurationManager

base_vector_store_module = MagicMock()
base_vector_store_module.BaseVectorStore = MockBaseVectorStore

base_document_loader_module = MagicMock()
base_document_loader_module.BaseDocumentLoader = MockBaseDocumentLoader
base_document_loader_module.LoaderError = Exception

constants_module = MagicMock()
constants_module.Constants = MockConstants

# Patch sys.modules
# Note: We should be careful not to break imports for modules we are testing if they import these base classes.
# If we patch sys.modules, any import of these modules will get the mock.
# This is what we want for tests that depend on them but don't test them directly.
# But for tests that test the base classes themselves (like test_core_base_agent.py), we might have issues if they import from the module we mocked.
# However, test_core_base_agent.py imports from oai_agent_core.core.base_agent.
# If we mock oai_agent_core.core.base_agent, then test_core_base_agent.py will test the mock, not the real class.

# So we should NOT mock the modules that contain the classes we are testing in unit tests.
# But we DO want to mock them for integration tests or when they are dependencies.

# Since we are running all tests together, global patching in conftest might be problematic for unit tests of the patched modules.

# A better approach is to only patch external dependencies or use patching in specific test files.
# But the user asked to fix the tests, and the previous strategy relied on conftest patching.

# The issue with test_components_document_loader.py is that it imports DocumentLoader which imports BaseDocumentLoader.
# If we don't patch BaseDocumentLoader, it uses the real one.
# The real BaseDocumentLoader works fine, but maybe it has side effects or dependencies we want to avoid.

# If I don't patch sys.modules here, then DocumentLoader uses real BaseDocumentLoader.
# Real BaseDocumentLoader sets self.embedding.
# So why did it fail?

# Maybe because I had a partial patch or something.

# Let's NOT patch sys.modules globally for core modules in conftest.py for agent_core project, 
# because we are testing those core modules.
# We should only patch external dependencies or things that are hard to test.

# So I will revert conftest.py to just have the fixture, and maybe some external mocks if needed.
# But wait, test_components_document_loader.py failed with AttributeError.
# This implies that DocumentLoader was instantiated but didn't have embedding attribute.
# This happens if BaseDocumentLoader.__init__ was NOT called or didn't set it.

# If I use the real BaseDocumentLoader, it sets it.
# So the previous failure must have been due to some patching that replaced BaseDocumentLoader with something that didn't set it.

# I will keep conftest.py minimal and rely on specific patches in test files if needed.
# But I need to make sure test_components_document_loader.py works.

# Let's look at test_components_document_loader.py again.
# It does NOT patch BaseDocumentLoader.
# So it uses the real one.
# Real BaseDocumentLoader.__init__:
# self.embedding = embedding
# ...
# self.reinitialize = self.reinitialize_database()

# DocumentLoader.__init__:
# super().__init__(...)

# If it failed, it means something went wrong.

# Wait, I see `oai_agent_core/core/base_document_loader.py` imports:
# from langchain_community.document_loaders import DirectoryLoader
# ...

# If these imports fail, the module might not load correctly? No, it would raise ImportError.

# Let's try to run the test without global mocks.
# I will write a minimal conftest.py.
