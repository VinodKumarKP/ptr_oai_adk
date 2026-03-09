import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_core.aws_strands_core.builders.agent_builder import AgentBuilder
from oai_agent_core.core.constants import Constants

@pytest.fixture
def mock_model_manager():
    manager = MagicMock()
    manager.create_model.return_value = "mock_model"
    manager.default_config = {'model_id': 'test-model'}
    return manager

@pytest.fixture
def mock_tool_registry():
    registry = MagicMock()
    registry.get_tools_for_agent.return_value = []
    registry.get_mcp_configs.return_value = {}
    registry.get_mcp_clients.return_value = []
    registry.has_tool.return_value = True
    registry.enable_lazy_loading = False
    registry.project_root = "/tmp"
    # Mock async load methods
    registry.load_mcp_tools_from_config = AsyncMock(return_value=[])
    registry.lazy_loading_required_tools.return_value = []
    return registry

@pytest.fixture
def builder(mock_model_manager, mock_tool_registry):
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.BaseAgentBuilder.__init__') as mock_base_init:
        builder = AgentBuilder(
            model_manager=mock_model_manager,
            tool_registry=mock_tool_registry,
            logger=MagicMock()
        )
        # Manually set attributes
        builder.model_manager = mock_model_manager
        builder.tool_registry = mock_tool_registry
        builder.logger = MagicMock()
        builder.llm = None
        builder.document_loader = None
        builder.vector_store = None
        import asyncio
        builder._tool_lock = asyncio.Lock()
        builder._kb_lock = asyncio.Lock()
        return builder

@pytest.mark.asyncio
async def test_create_agent_basic(builder):
    config = {
        'system_prompt': 'You are a helper',
        'model': {'model_id': 'gpt-4'}
    }
    
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.Agent') as MockAgent:
        # Mock base class methods
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=[])
        builder._load_mcp_tools = AsyncMock(return_value=[])
        builder._load_knowledge_base_tools = AsyncMock(return_value=[])
        
        agent = await builder.create_agent('test_agent', config)
        
        MockAgent.assert_called_once()
        call_args = MockAgent.call_args[1]
        assert call_args['name'] == 'test_agent'
        # The system prompt might be modified by lazy loading logic, so check if it contains the original prompt
        assert 'You are a helper' in call_args['system_prompt']
        
        builder.llm = "mock_model"
        agent = await builder.create_agent('test_agent', config)
        call_args = MockAgent.call_args[1]
        assert call_args['model'] == 'mock_model'

@pytest.mark.asyncio
async def test_create_agent_with_tools(builder):
    config = {
        'system_prompt': 'helper',
        'tools': ['tool1', 'tool2']
    }
    
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.Agent') as MockAgent:
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=['t1', 't2'])
        builder._load_mcp_tools = AsyncMock(return_value=[])
        builder._load_knowledge_base_tools = AsyncMock(return_value=[])
        
        await builder.create_agent('test_agent', config)
        
        call_args = MockAgent.call_args[1]
        # Check if tools are passed correctly. Note that lazy loading might add extra tools.
        # So we check if our tools are present in the list
        tools_arg = call_args['tools']
        if tools_arg is None:
             tools_arg = []
        assert 't1' in tools_arg
        assert 't2' in tools_arg
        builder._get_regular_tools.assert_called_once()

@pytest.mark.asyncio
async def test_create_agent_with_mcps(builder):
    config = {
        'system_prompt': 'helper',
        'mcps': {'mcp1': {}}
    }
    
    mock_mcp_client = MagicMock()
    
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.Agent') as MockAgent:
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=[])
        builder._load_mcp_tools = AsyncMock(return_value=[mock_mcp_client])
        builder._load_knowledge_base_tools = AsyncMock(return_value=[])
        
        await builder.create_agent('test_agent', config)
        
        call_args = MockAgent.call_args[1]
        # Check if mcp client is in tools
        tools_arg = call_args['tools']
        if tools_arg is None:
             tools_arg = []
        assert mock_mcp_client in tools_arg
        builder._load_mcp_tools.assert_called_once()

@pytest.mark.asyncio
async def test_create_agent_with_kb(builder):
    config = {
        'system_prompt': 'helper',
        'knowledge_base': [{'name': 'kb1'}]
    }
    
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.Agent'):
        mock_kb_tool = MagicMock()
        
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=[])
        builder._load_mcp_tools = AsyncMock(return_value=[])
        builder._load_knowledge_base_tools = AsyncMock(return_value=[mock_kb_tool])
        
        await builder.create_agent('test_agent', config)
        
        builder._load_knowledge_base_tools.assert_called_once()

@pytest.mark.asyncio
async def test_create_agents_from_config(builder):
    configs = [
        {'a1': {'system_prompt': 'p1'}},
        {'a2': {'system_prompt': 'p2'}}
    ]
    
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.Agent'):
        # Mock create_single_agent (aliased as create_agent)
        # Note: create_agent calls create_single_agent, which calls _create_agent_instance
        # We can mock create_single_agent directly on the builder instance
        builder.create_single_agent = AsyncMock(return_value=MagicMock())
        
        agents = await builder.create_agents_from_config(configs)
        assert len(agents) == 2
        assert 'a1' in agents
        assert 'a2' in agents
        assert builder.create_single_agent.call_count == 2

@pytest.mark.asyncio
async def test_create_agents_from_config_agent_as_tool(builder):
    configs = [
        {'a1': {'system_prompt': 'p1'}},
        {'a2': {'system_prompt': 'p2'}}
    ]
    
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.Agent') as MockAgent:
        builder.create_single_agent = AsyncMock(return_value=MagicMock())
        
        # Mock _create_agent_as_tool
        with patch.object(builder, '_create_agent_as_tool') as mock_create_tool:
            mock_tool = MagicMock()
            mock_create_tool.return_value = mock_tool
            
            agents = await builder.create_agents_from_config(configs, architecture="agent-as-tool")
            
            assert len(agents) == 2
            assert agents['a1'] == mock_tool
            assert agents['a2'] == mock_tool
            assert mock_create_tool.call_count == 2

def test_create_agent_as_tool(builder):
    mock_agent = MagicMock()
    
    # Mock the tool decorator
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.tool') as mock_decorator:
        # Configure the mock decorator to behave like @tool(name=..., description=...)
        # It should return a wrapper function that takes the decorated function
        
        def decorator_wrapper(name=None, description=None):
            def wrapper(func):
                func.name = name
                func.description = description
                return func
            return wrapper
            
        mock_decorator.side_effect = decorator_wrapper
        
        tool = builder._create_agent_as_tool(mock_agent, "tool_name", "desc")
        
        assert tool.name == "tool_name"
        assert tool.description == "desc"
        assert callable(tool)

def test_extract_context_map(builder):
    configs = [
        {'a1': {'context': ['a2']}},
        {'a2': {}}
    ]
    context_map = builder.extract_context_map(configs)
    assert context_map == {'a1': ['a2']}

def test_validate_agent_config_valid(builder):
    config = {'system_prompt': 'p', 'tools': ['t1']}
    is_valid, errors = builder.validate_agent_config('a1', config)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_agent_config_invalid(builder):
    config = {'tools': ['unknown']}
    builder.tool_registry.has_tool.return_value = False
    
    is_valid, errors = builder.validate_agent_config('a1', config)
    assert is_valid is False
    assert "missing system_prompt" in errors[0]
    assert "unknown tool" in errors[1]

def test_get_agent_summary(builder):
    mock_agent = MagicMock()
    mock_agent.tools = ['t1']
    mock_agent.instructions = "inst"
    mock_agent.name = "agent_name"
    
    summary = builder.get_agent_summary('key', mock_agent)
    assert summary['agent_key'] == 'key'
    assert summary['tool_count'] == 1
    assert summary['has_instructions'] is True

@pytest.mark.asyncio
async def test_load_knowledge_base_tools(builder):
    # Mock the module import using sys.modules
    mock_kb_module = MagicMock()
    mock_kb_factory = MagicMock()
    mock_kb_tool = MagicMock()
    mock_kb_factory.return_value.create_tool.return_value = mock_kb_tool
    mock_kb_module.KnowledgeBaseFactory = mock_kb_factory
    
    # Ensure parent packages exist in sys.modules to avoid import errors
    with patch.dict('sys.modules', {
        'oai_aws_strands_agent_core': MagicMock(),
        'oai_agent_core.aws_strands_core.components': MagicMock(),
        'oai_agent_core.aws_strands_core.components.knowledge': MagicMock(),
        'oai_agent_core.aws_strands_core.components.knowledge.knowledge_base_factory': mock_kb_module
    }):
        # We also need to patch asyncio.to_thread because the code uses it
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            # to_thread will be called with KnowledgeBaseFactory class and args
            # It should return the instance
            mock_to_thread.return_value = mock_kb_factory.return_value
            
            # Call the method on the CLASS to ensure we use the implementation in AgentBuilder
            # passing the builder instance as self
            tools = await AgentBuilder._load_knowledge_base_tools(builder, 'agent1', {'knowledge_base': [{'name': 'kb1'}]})
            
            assert len(tools) == 1
            assert tools[0] == mock_kb_tool

def test_create_supervisor_agent(builder):
    mock_agent = MagicMock()
    with patch('oai_agent_core.aws_strands_core.builders.agent_builder.Agent', return_value=mock_agent):
        supervisor = builder._create_supervisor_agent(
            Constants.PATTERN_AGENT_AS_TOOL, 
            [], 
            ['tool1'], 
            'prompt'
        )
        assert supervisor == mock_agent

def test_create_supervisor_agent_none(builder):
    supervisor = builder._create_supervisor_agent(
        Constants.PATTERN_SUPERVISOR, 
        [], 
        [], 
        'prompt'
    )
    assert supervisor is None
