import sys
import types
import os
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
        self.enable_lazy_loading = True
    
    def _get_mcp_name_list_from_mcp_config(self, config):
        return list(config.get('tools', {}).keys())
    
    def _get_mcp_config(self, config, name):
        return config.get('tools', {}).get(name, {})
    
    def get_tools_for_agent(self, tool_names):
        return []
        
    def load_tools_from_config(self, config):
        pass
        
    def has_tool(self, tool_name):
        return True
        
    async def load_mcp_tools_from_config(self, config):
        return []

class MockBaseKnowledgeBaseFactory:
    def __init__(self, knowledge_base_config=None, logger=None, **kwargs):
        self.knowledge_base_config = knowledge_base_config
        self.logger = logger or MagicMock()
    
    def search_custom_knowledge_base(self, query):
        return "Mock Search Result"
        
    def create_tool(self):
        return MagicMock(name="kb_tool")

class MockBaseAgent:
    def __init__(self, llm=None, agent_name="", agent_config=None, session_id="default", config_root=None, agent_type="langchain", user_id="default", document_loader=None, vector_store=None, model_manager=None, **kwargs):
        self.llm = llm or MagicMock()
        self.agent_name = agent_name
        self.agent_config = agent_config or {}
        self.session_id = session_id
        self.config_root = config_root
        self.agent_type = agent_type
        self.user_id = user_id
        self.document_loader = document_loader
        self.vector_store = vector_store
        self.memory_store = MagicMock()
        self.model_manager = model_manager
        self.logger = MagicMock()
        self.langfuse_manager = MagicMock()
        self.langfuse_manager.callback_handler = None
        self._initialized = False
        self.global_kb_factory = None

    async def _ensure_initialized(self):
        pass

    def _augment_message(self, message: str, original_query: str = None):
        return f"KB Context {original_query}\n\n{message}"

    def _guardrail_input_message(self, message):
        return message

    async def _load_tools_and_kb_and_memory(self, kb_factory_class):
        kb_config = self.agent_config.get('knowledge_base')
        if kb_config and kb_factory_class:
            try:
                self.global_kb_factory = kb_factory_class(
                    knowledge_base_config=kb_config,
                    logger=self.logger,
                    project_root=self.config_root,
                    llm=self.llm,
                    document_loader=self.document_loader,
                    vector_store=self.vector_store
                )
            except Exception as e:
                if isinstance(e, ImportError):
                    raise e
                self.logger.error(f"Failed to init KB: {e}")

class MockConfigManager:
    def __init__(self, config_root=None):
        self.config_root = config_root
        
    def ruamel_to_native(self, config):
        return config
        
    def load_agent_config(self, agent_name):
        return {'system_prompt': 'mock prompt'}

class MockConstants:
    LANGGRAPH = "langgraph"
    REMOTE = "remote"
    PATTERN_SUPERVISOR = 'supervisor'
    PATTERN_SWARM = 'swarm'
    PATTERN_AGENT_AS_TOOL = 'agent_as_tool'

class MockBaseModelConfigurationManager:
    def __init__(self, *args, **kwargs):
        self.logger = MagicMock()
        self.default_config = {}
    
    def create_model(self, model_config=None):
        return MagicMock()
        
    def _merge_with_defaults(self, config):
        return config or {}
        
    def _validate_config(self, config):
        pass

class MockBaseAgentBuilder:
    def __init__(self, model_manager, tool_registry, llm=None, config_root=None, logger=None, document_loader=None, vector_store=None):
        self.model_manager = model_manager
        self.tool_registry = tool_registry
        self.llm = llm
        self.config_root = config_root
        self.logger = logger or MagicMock()
        self.document_loader = document_loader
        self.vector_store = vector_store
        self._tool_lock = MagicMock()
        self._kb_lock = MagicMock()
    
    async def create_single_agent(self, agent_name, agent_config, pre_loaded_tools=None):
        self._ensure_model(agent_config)
        all_tools = list(pre_loaded_tools) if pre_loaded_tools else []
        regular_tools = await self._get_regular_tools(agent_name, agent_config)
        all_tools.extend(regular_tools)
        mcp_tools = await self._load_mcp_tools(agent_name, agent_config)
        all_tools.extend(mcp_tools)
        kb_tools = await self._load_knowledge_base_tools(agent_name, agent_config)
        all_tools.extend(kb_tools)
        return self._create_agent_instance(agent_name, agent_config, all_tools)
        
    async def create_multi_agent_system(self, agent_configs, system_prompt="", session_id="default", pattern="supervisor"):
        agent_definitions = self._normalize_agent_configs(agent_configs, MagicMock())
        base_agent_list, agent_list, sub_agent_tools = await self._create_agents_parallel(agent_definitions, pattern)
        supervisor = self._create_supervisor_agent(pattern, agent_list, sub_agent_tools, system_prompt)
        return supervisor, base_agent_list

    def _ensure_model(self, agent_config):
        pass
        
    async def _get_regular_tools(self, agent_name, agent_config):
        return []
        
    async def _load_mcp_tools(self, agent_name, agent_config):
        return []
        
    async def _load_knowledge_base_tools(self, agent_name, agent_config):
        kb_configs = agent_config.get('knowledge_base', [])
        if not kb_configs:
            return []
        factory_class = self._get_knowledgebase_factory_class()
        if factory_class:
            factory = factory_class(
                knowledge_base_config=kb_configs,
                logger=self.logger,
                project_root=self.config_root,
                llm=self.llm,
                document_loader=self.document_loader,
                vector_store=self.vector_store
            )
            return [factory.create_tool()]
        return []
        
    def _normalize_agent_configs(self, agent_configs, config_manager):
        return []
        
    async def _create_agents_parallel(self, agent_definitions, pattern):
        return [], [], []
        
    def _create_agent_instance(self, agent_name, agent_config, tools):
        pass
        
    def _create_supervisor_agent(self, pattern, agent_list, sub_agent_tools, system_prompt):
        pass
        
    def _get_knowledgebase_factory_class(self):
        return None

class MockBaseResultExtractor:
    def __init__(self, logger=None):
        self.logger = logger or MagicMock()

    def extract_text(self, result):
        return "Mock Text"

    def extract_token_usage(self, result):
        return {}

    def format_response(self, result, session_id, model_id, model_provider, **kwargs):
        return {
            "content": {"text": "Mock Response", "type": "text"},
            "model": {"model_id": model_id, "model_provider": model_provider}
        }
        
    def format_streaming_chunk(self, content, **kwargs):
        return {"content": content}
        
    def extract_execution_metadata(self, result):
        return {}

    def _serialize_result(self, result):
        return str(result)

# Create mock modules
base_tool_registry_module = MagicMock()
base_tool_registry_module.BaseToolRegistry = MockBaseToolRegistry

base_kb_factory_module = MagicMock()
base_kb_factory_module.BaseKnowledgeBaseFactory = MockBaseKnowledgeBaseFactory

base_agent_module = MagicMock()
base_agent_module.BaseAgent = MockBaseAgent

config_manager_module = MagicMock()
config_manager_module.ConfigManager = MockConfigManager

constants_module = MagicMock()
constants_module.Constants = MockConstants

base_model_config_module = MagicMock()
base_model_config_module.BaseModelConfigurationManager = MockBaseModelConfigurationManager

base_agent_builder_module = MagicMock()
base_agent_builder_module.BaseAgentBuilder = MockBaseAgentBuilder

base_result_extractor_module = MagicMock()
base_result_extractor_module.BaseResultExtractor = MockBaseResultExtractor

# Ensure oai_agent_core is imported correctly as a namespace package
try:
    import oai_agent_core
except ImportError:
    # If it can't be imported, create it as a namespace package pointing to local src
    oai_agent_core = types.ModuleType('oai_agent_core')
    oai_agent_core.__path__ = [os.path.join(src_path, 'oai_agent_core')]
    sys.modules['oai_agent_core'] = oai_agent_core

# Patch missing submodules of oai_agent_core (those from agent-core package)
def patch_submodule(name, module):
    full_name = f'oai_agent_core.{name}'
    if full_name not in sys.modules:
        sys.modules[full_name] = module
        # Also set it as an attribute on the parent module
        parts = name.split('.')
        parent = oai_agent_core
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, types.ModuleType(part))
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module)

patch_submodule('core.base_tool_registry', base_tool_registry_module)
patch_submodule('core.base_knowledge_base_factory', base_kb_factory_module)
patch_submodule('core.base_agent', base_agent_module)
patch_submodule('core.base_model_configuration_manager', base_model_config_module)
patch_submodule('core.constants', constants_module)
patch_submodule('builders.base_agent_builder', base_agent_builder_module)
patch_submodule('processing.base_result_extractor', base_result_extractor_module)

# Patch attributes on oai_agent_core.core for direct imports
if not hasattr(oai_agent_core, 'core'):
    oai_agent_core.core = types.ModuleType('core')
    sys.modules['oai_agent_core.core'] = oai_agent_core.core

oai_agent_core.core.BaseToolRegistry = MockBaseToolRegistry
oai_agent_core.core.BaseKnowledgeBaseFactory = MockBaseKnowledgeBaseFactory
oai_agent_core.core.BaseAgent = MockBaseAgent
oai_agent_core.core.BaseModelConfigurationManager = MockBaseModelConfigurationManager
oai_agent_core.core.Constants = MockConstants

# Patch other modules
sys.modules['oai_agent_core.manager'] = MagicMock()
sys.modules['oai_agent_core.manager.config_manager'] = config_manager_module
sys.modules['oai_agent_core.components'] = MagicMock()
sys.modules['oai_agent_core.components.configuration'] = MagicMock()
sys.modules['oai_agent_core.components.configuration.model_config'] = config_manager_module
sys.modules['oai_agent_core.utils'] = MagicMock()
sys.modules['oai_agent_core.utils.dynamic_class_loader'] = MagicMock()
sys.modules['oai_agent_core.processing'] = MagicMock()
sys.modules['oai_agent_core.processing.message_formatter'] = MagicMock()
