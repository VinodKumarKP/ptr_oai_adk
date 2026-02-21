import os
import sys
import types
from unittest.mock import MagicMock

# Add src to sys.path to ensure local packages are discoverable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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

class MockBaseKnowledgeBaseFactory:
    def __init__(self, knowledge_base_config=None, logger=None, **kwargs):
        self.knowledge_base_config = knowledge_base_config
        self.logger = logger or MagicMock()
    
    def search_custom_knowledge_base(self, query):
        return "Mock Search Result"
        
    def create_tool(self):
        return MagicMock(name="kb_tool")

class MockBaseAgent:
    def __init__(self, llm=None, agent_name="", agent_config=None, session_id="default", config_root=None, agent_type="crewai", user_id="default", document_loader=None, vector_store=None, model_manager=None, **kwargs):
        self.llm = llm
        self.agent_name = agent_name
        self.agent_config = agent_config or {}
        self.session_id = session_id
        self.config_root = config_root
        self.agent_type = agent_type
        self.user_id = user_id
        self.document_loader = document_loader
        self.vector_store = vector_store
        self.model_manager = model_manager
        self.logger = MagicMock()
        self.langfuse_manager = MagicMock()
        self.langfuse_manager.callback_handler = None
        self.langfuse_manager.is_enabled = False
        self._initialized = False
        self.memory_store = MagicMock()
        
    async def _ensure_initialized(self):
        pass

    def _augment_message(self, message: str, original_query: str = None):
        return f"KB Context {original_query}\n\n{message}"

    def _get_conversation_context(self, current_message: str) -> str:
        pass

    async def _load_tools_and_kb_and_memory(self, kb_factory_class):
        return 'Success'

    def _guardrail_input_message(self, message):
        return message

    def _guardrail_output_message(self, message):
        return message

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

class MockConstants:
    CREWAI = "crewai"
    REMOTE = "remote"
    LANGCHAIN = "langchain"
    BEDROCK = "bedrock"

class MockBaseResultExtractor:
    def __init__(self, logger=None):
        self.logger = logger or MagicMock()
    
    def format_response(self, result, session_id, model_id, model_provider, include_raw=False, input_message=None, original_message=None, final=True):
        return {
            "content": {
                "text": str(result),
                "type": "text",
                "final": final,
                "session_id": session_id
            },
            "model": {
                "model_id": model_id,
                "model_provider": model_provider
            }
        }

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

base_result_extractor_module = MagicMock()
base_result_extractor_module.BaseResultExtractor = MockBaseResultExtractor

# Patch sys.modules
# Ensure oai_agent_core package exists
if 'oai_agent_core' not in sys.modules:
    oai_agent_core = types.ModuleType('oai_agent_core')
    oai_agent_core.__path__ = [os.path.join(src_path, 'oai_agent_core')]
    sys.modules['oai_agent_core'] = oai_agent_core

# Ensure oai_agent_core.core package exists
if 'oai_agent_core.core' not in sys.modules:
    core_pkg = types.ModuleType('oai_agent_core.core')
    core_pkg.__path__ = []
    sys.modules['oai_agent_core.core'] = core_pkg
    sys.modules['oai_agent_core'].core = core_pkg

# Patch specific modules
sys.modules['oai_agent_core.core.base_tool_registry'] = base_tool_registry_module
sys.modules['oai_agent_core.core.base_knowledge_base_factory'] = base_kb_factory_module
sys.modules['oai_agent_core.core.base_agent'] = base_agent_module
sys.modules['oai_agent_core.core.base_model_configuration_manager'] = base_model_config_manager_module
sys.modules['oai_agent_core.core.base_vector_store'] = base_vector_store_module
sys.modules['oai_agent_core.core.base_document_loader'] = base_document_loader_module
sys.modules['oai_agent_core.core.constants'] = constants_module

# Also patch attributes on the core package itself to support "from oai_agent_core.core import X"
sys.modules['oai_agent_core.core'].BaseToolRegistry = MockBaseToolRegistry
sys.modules['oai_agent_core.core'].BaseKnowledgeBaseFactory = MockBaseKnowledgeBaseFactory
sys.modules['oai_agent_core.core'].BaseAgent = MockBaseAgent
sys.modules['oai_agent_core.core'].BaseModelConfigurationManager = MockBaseModelConfigurationManager
sys.modules['oai_agent_core.core'].BaseVectorStore = MockBaseVectorStore
sys.modules['oai_agent_core.core'].BaseDocumentLoader = MockBaseDocumentLoader
sys.modules['oai_agent_core.core'].Constants = MockConstants

# Patch other modules
sys.modules['oai_agent_core.utils'] = MagicMock()
sys.modules['oai_agent_core.utils.dynamic_class_loader'] = MagicMock()
sys.modules['oai_agent_core.processing'] = MagicMock()
sys.modules['oai_agent_core.processing.output_serializer'] = MagicMock()
sys.modules['oai_agent_core.processing.message_formatter'] = MagicMock()
sys.modules['oai_agent_core.processing.base_result_extractor'] = base_result_extractor_module
