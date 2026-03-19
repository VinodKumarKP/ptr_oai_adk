import os
import sys
import types
from typing import Optional, Dict, Any
from unittest.mock import MagicMock

# Add src to sys.path to ensure local packages are discoverable
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# --- Global Mocking Setup ---
# This runs when pytest imports conftest.py, before collecting other tests.

# 1. Mock strands package
if 'strands' not in sys.modules:
    strands = types.ModuleType('strands')
    strands.__path__ = []  # Essential: marks it as a package
    sys.modules['strands'] = strands
    
    # Mock submodules
    strands_tools = types.ModuleType('strands.tools')
    sys.modules['strands.tools'] = strands_tools
    strands.tools = strands_tools # Link to parent

    strands_tools_mcp = types.ModuleType('strands.tools.mcp')
    sys.modules['strands.tools.mcp'] = strands_tools_mcp
    strands_tools.mcp = strands_tools_mcp # Link to parent

    strands_tools_mcp_client = types.ModuleType('strands.tools.mcp.mcp_client')
    sys.modules['strands.tools.mcp.mcp_client'] = strands_tools_mcp_client
    strands_tools_mcp.mcp_client = strands_tools_mcp_client # Link to parent
    
    strands_multiagent = types.ModuleType('strands.multiagent')
    sys.modules['strands.multiagent'] = strands_multiagent
    strands.multiagent = strands_multiagent # Link to parent

    strands_models = types.ModuleType('strands.models')
    sys.modules['strands.models'] = strands_models
    strands.models = strands_models # Link to parent

    strands_models_litellm = types.ModuleType('strands.models.litellm')
    sys.modules['strands.models.litellm'] = strands_models_litellm
    strands_models.litellm = strands_models_litellm # Link to parent

    strands_agent = types.ModuleType('strands.agent')
    sys.modules['strands.agent'] = strands_agent
    strands.agent = strands_agent # Link to parent

    strands_agent_result = types.ModuleType('strands.agent.agent_result')
    sys.modules['strands.agent.agent_result'] = strands_agent_result
    strands_agent.agent_result = strands_agent_result # Link to parent
    
    # Mock attributes
    sys.modules['strands'].Agent = MagicMock()
    sys.modules['strands'].tools.tool = MagicMock(side_effect=lambda x: x)
    sys.modules['strands'].tools.mcp.mcp_client.MCPClient = MagicMock()
    sys.modules['strands'].multiagent.GraphBuilder = MagicMock()
    sys.modules['strands'].multiagent.Swarm = MagicMock()
    sys.modules['strands'].models.litellm.LiteLLMModel = MagicMock()
    
    # Mock result classes
    class MockGraphResult: pass
    class MockSwarmResult: pass
    class MockAgentResult: pass
    
    sys.modules['strands.multiagent.graph'] = types.ModuleType('strands.multiagent.graph')
    sys.modules['strands.multiagent.graph'].GraphResult = MockGraphResult
    strands_multiagent.graph = sys.modules['strands.multiagent.graph']

    sys.modules['strands.multiagent.swarm'] = types.ModuleType('strands.multiagent.swarm')
    sys.modules['strands.multiagent.swarm'].SwarmResult = MockSwarmResult
    strands_multiagent.swarm = sys.modules['strands.multiagent.swarm']

    sys.modules['strands.agent.agent_result'].AgentResult = MockAgentResult

# 2. Mock oai_agent_core package
if 'oai_agent_core' not in sys.modules:
    oai_agent_core = types.ModuleType('oai_agent_core')
    oai_agent_core.__path__ = [os.path.join(src_path, 'oai_agent_core')]
    sys.modules['oai_agent_core'] = oai_agent_core
    
    # Core
    core = types.ModuleType('oai_agent_core.core')
    core.__path__ = []
    sys.modules['oai_agent_core.core'] = core
    sys.modules['oai_agent_core'].core = core
    
    # Processing
    processing = types.ModuleType('oai_agent_core.processing')
    processing.__path__ = []
    sys.modules['oai_agent_core.processing'] = processing
    sys.modules['oai_agent_core'].processing = processing
    
    # Utils
    utils = types.ModuleType('oai_agent_core.utils')
    utils.__path__ = []
    sys.modules['oai_agent_core.utils'] = utils
    sys.modules['oai_agent_core.utils.dynamic_class_loader'] = MagicMock()
    sys.modules['oai_agent_core'].utils = utils
    
    # Components
    components = types.ModuleType('oai_agent_core.components')
    components.__path__ = []
    sys.modules['oai_agent_core.components'] = components
    sys.modules['oai_agent_core'].components = components
    
    # Builders
    builders = types.ModuleType('oai_agent_core.builders')
    builders.__path__ = []
    sys.modules['oai_agent_core.builders'] = builders
    sys.modules['oai_agent_core'].builders = builders

    # Mock specific modules
    
    # Base Agent
    mock_base_agent = types.ModuleType('oai_agent_core.core.base_agent')
    class MockBaseAgent:
        def __init__(self, **kwargs):
            self.logger = MagicMock()
            self.agent_name = kwargs.get('agent_name', 'mock')
            self.session_id = kwargs.get('session_id', 'default')
            self.user_id = kwargs.get('user_id', 'default')
            self.agent_config = kwargs.get('agent_config', {})
            self.langfuse_manager = MagicMock()
            self.langfuse_manager.is_enabled = False
            self._initialized = False
            self.memory_store = MagicMock()
            self.global_kb_factory = None
            self.config_root = kwargs.get('config_root')
            self.llm = kwargs.get('llm')
            self.document_loader = kwargs.get('document_loader')
            self.vector_store = kwargs.get('vector_store')
            self.skill_registry = kwargs.get('skill_registry')
            self.output_model_registry = kwargs.get('output_model_registryf')

        async def _ensure_initialized(self):
            pass

        def _augment_message(self, message: str, original_query: str = None):
            return f"KB Context {original_query}\n\n{message}"

        def _guardrail_input_message(self, message):
            return message

        def _guardrail_output_message(self, message):
            return message
            
        async def _load_tools_and_kb_and_memory(self, kb_factory_class):
            # Simulate KB loading logic
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

    mock_base_agent.BaseAgent = MockBaseAgent
    sys.modules['oai_agent_core.core.base_agent'] = mock_base_agent
    core.BaseAgent = MockBaseAgent
    
    # Constants
    mock_constants = types.ModuleType('oai_agent_core.core.constants')
    class MockConstants:
        AWS_STRANDS = "aws_strands"
        LANGGRAPH = "langgraph"
        REMOTE = "remote"
        PATTERN_SUPERVISOR = 'supervisor'
        PATTERN_GRAPH = 'graph'
        PATTERN_SWARM = 'swarm'
        PATTERN_SEQUENTIAL = 'sequential'
        PATTERN_HIERARCHICAL = 'hierarchical'
        PATTERN_AGENT_AS_TOOL = 'agent-as-tool'
    mock_constants.Constants = MockConstants
    sys.modules['oai_agent_core.core.constants'] = mock_constants
    core.Constants = MockConstants
    
    # Output Serializer
    mock_serializer = types.ModuleType('oai_agent_core.processing.output_serializer')
    mock_serializer.OutputSerializer = MagicMock()
    sys.modules['oai_agent_core.processing.output_serializer'] = mock_serializer
    processing.OutputSerializer = MagicMock()
    
    # Message Formatter
    mock_formatter = types.ModuleType('oai_agent_core.processing.message_formatter')
    mock_formatter.MessageFormatter = MagicMock()
    sys.modules['oai_agent_core.processing.message_formatter'] = mock_formatter
    processing.MessageFormatter = MagicMock()
    
    # Model Config Manager
    mock_model_config = types.ModuleType('oai_agent_core.core.base_model_configuration_manager')
    class MockBaseModelConfigurationManager:
        def __init__(self, default_config=None, logger=None):
            self.default_config = default_config or {}
            self.logger = logger or MagicMock()
        def _merge_with_defaults(self, config): return config or self.default_config
        def _validate_config(self, config): pass
        def get_model_info(self): pass
    mock_model_config.BaseModelConfigurationManager = MockBaseModelConfigurationManager
    sys.modules['oai_agent_core.core.base_model_configuration_manager'] = mock_model_config
    core.BaseModelConfigurationManager = MockBaseModelConfigurationManager

    # Tool Registry
    mock_tool_registry = types.ModuleType('oai_agent_core.core.base_tool_registry')
    class MockBaseToolRegistry:
        def __init__(self, logger=None, project_root=None, enable_lazy_loading: Optional[bool] = False):
            self.logger = logger or MagicMock()
            self.project_root = project_root
            self.tools = {}
            self.mcp_clients = {}
            self.enable_lazy_loading = False
        def _get_mcp_name_list_from_mcp_config(self, config): return list(config.keys())
        def _get_mcp_config(self, config, name): return config.get(name, {})
        def clear(self): 
            self.tools = {}
            self.mcp_clients = []

        @staticmethod
        def _sanitize_headers(headers: Dict[str, Any]) -> Dict[str, str]:
            return headers
    mock_tool_registry.BaseToolRegistry = MockBaseToolRegistry
    sys.modules['oai_agent_core.core.base_tool_registry'] = mock_tool_registry
    core.BaseToolRegistry = MockBaseToolRegistry

    # Knowledge Base Factory
    mock_kb_factory = types.ModuleType('oai_agent_core.core.base_knowledge_base_factory')
    class MockBaseKnowledgeBaseFactory:
        def __init__(self, knowledge_base_config=None, logger=None, **kwargs):
            self.knowledge_base_config = knowledge_base_config
            self.logger = logger or MagicMock()
        def search_custom_knowledge_base(self, query): return "Mock Search Result"
        def create_tool(self): return MagicMock(name="kb_tool")
    mock_kb_factory.BaseKnowledgeBaseFactory = MockBaseKnowledgeBaseFactory
    sys.modules['oai_agent_core.core.base_knowledge_base_factory'] = mock_kb_factory
    core.BaseKnowledgeBaseFactory = MockBaseKnowledgeBaseFactory
    
    # Base Agent Builder
    mock_base_builder = types.ModuleType('oai_agent_core.builders.base_agent_builder')
    class MockBaseAgentBuilder:
        def __init__(self, model_manager,
                     tool_registry, llm=None, config_root=None,
                     logger=None, document_loader=None, vector_store=None,
                     skill_registry=None, structured_output_model_registry=None):
            self.model_manager = model_manager
            self.tool_registry = tool_registry
            self.llm = llm
            self.config_root = config_root
            self.logger = logger or MagicMock()
            self.document_loader = document_loader
            self.vector_store = vector_store
            self._tool_lock = MagicMock()
            self._kb_lock = MagicMock()
            self.skill_registry = MagicMock()
            self.structured_output_model_registry = MagicMock()
        
        async def create_single_agent(self, agent_name, agent_config, pre_loaded_tools=None):
            # Simulate template method
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
            # Simulate template method
            agent_definitions = self._normalize_agent_configs(agent_configs, MagicMock())
            base_agent_list, agent_list, sub_agent_tools = await self._create_agents_parallel(agent_definitions, pattern)
            supervisor = self._create_supervisor_agent(pattern, agent_list, sub_agent_tools, system_prompt)
            return supervisor, base_agent_list

        def _ensure_model(self, agent_config): pass
        async def _get_regular_tools(self, agent_name, agent_config): return []
        async def _load_mcp_tools(self, agent_name, agent_config): return []
        
        async def _load_knowledge_base_tools(self, agent_name, agent_config):
            # Simulate generic implementation
            kb_configs = agent_config.get('knowledge_base', [])
            if not kb_configs:
                return []
            
            # Call abstract method
            factory_class = self._get_knowledgebase_factory_class()
            if factory_class:
                # Simulate factory instantiation and tool creation
                # We assume factory_class is a mock or class that can be instantiated
                factory = factory_class(
                    knowledge_base_config=kb_configs,
                    logger=self.logger,
                    project_root=self.tool_registry.project_root,
                    llm=self.llm,
                    document_loader=self.document_loader,
                    vector_store=self.vector_store
                )
                return [factory.create_tool()]
            return []

        def _normalize_agent_configs(self, agent_configs, config_manager): return []
        async def _create_agents_parallel(self, agent_definitions, pattern): return [], [], []
        def _create_agent_instance(self, agent_name, agent_config, tools): pass
        def _create_supervisor_agent(self, pattern, agent_list, sub_agent_tools, system_prompt): pass
        def _get_knowledgebase_factory_class(self): return None
        
    mock_base_builder.BaseAgentBuilder = MockBaseAgentBuilder
    sys.modules['oai_agent_core.builders.base_agent_builder'] = mock_base_builder
    builders.BaseAgentBuilder = MockBaseAgentBuilder
    
    # Base Result Extractor
    mock_base_result_extractor = types.ModuleType('oai_agent_core.processing.base_result_extractor')
    class MockBaseResultExtractor:
        def __init__(self, logger=None):
            self.logger = logger or MagicMock()
        def extract_text(self, result): return str(result)
        def extract_token_usage(self, result): return {}
        def format_response(self, result, session_id, model_id, model_provider, include_raw=False, input_message=None, original_message=None, final=True):
            response = {
                "content": {
                    "text": self.extract_text(result),
                    "type": "text",
                    "final": final,
                    "session_id": session_id
                },
                "token_usage": self.extract_token_usage(result),
                "model": {
                    "model_id": model_id,
                    "model_provider": model_provider
                }
            }
            if include_raw:
                response["raw_result"] = result
            return response
        def extract_execution_metadata(self, result): return {}
        def extract_results_list(self, result): return []
        def _extract_from_content(self, content):
            if isinstance(content, str): return content
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict) and 'text' in part:
                        text_parts.append(part['text'])
                    elif hasattr(part, 'text'):
                        text_parts.append(part.text)
                    elif isinstance(part, str):
                        text_parts.append(part)
                return "".join(text_parts)
            return str(content)
        def _serialize_result(self, result):
            if hasattr(result, '__dict__'):
                # Filter out private attributes
                return {k: v for k, v in result.__dict__.items() if not k.startswith('_')}
            return result
        def format_streaming_chunk(self, chunk, agent=None, **kwargs):
            return {"content": chunk, "agent": agent, "final": False, **kwargs}

    mock_base_result_extractor.BaseResultExtractor = MockBaseResultExtractor
    sys.modules['oai_agent_core.processing.base_result_extractor'] = mock_base_result_extractor
    processing.BaseResultExtractor = MockBaseResultExtractor

    # Dynamic Class Loader
    sys.modules['oai_agent_core.utils.dynamic_class_loader'] = MagicMock()

# 3. Mock mcp package
if 'mcp' not in sys.modules:
    mcp = types.ModuleType('mcp')
    mcp.__path__ = []
    sys.modules['mcp'] = mcp
    sys.modules['mcp.client'] = types.ModuleType('mcp.client')
    sys.modules['mcp.client.sse'] = MagicMock()
    sys.modules['mcp.client.stdio'] = MagicMock()
    sys.modules['mcp.client.streamable_http'] = MagicMock()
