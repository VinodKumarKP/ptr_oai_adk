import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_core.langgraph_core.builders.agent_builder import AgentBuilder
from oai_agent_core.langgraph_core.components.registry.tool_registry import LangChainToolRegistry
from oai_agent_core.utils.constants import Constants

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model_name = "gpt-4"
    return llm

def dummy():
    pass

@pytest.fixture
def mock_tool_registry():
    registry = MagicMock(spec=LangChainToolRegistry)
    registry.get_tools_for_agent.return_value = []
    registry.load_mcp_tools_from_config = AsyncMock(return_value=[])
    registry.has_tool.return_value = True
    # Add the missing method mock
    registry.generate_lazy_mcp_system_prompt = MagicMock(return_value="")
    registry.enable_lazy_loading = False
    return registry

@pytest.fixture
def builder(mock_llm, mock_tool_registry):
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.BaseAgentBuilder.__init__') as mock_base_init:
        builder = AgentBuilder(
            llm=mock_llm,
            tool_registry=mock_tool_registry,
            config_root="/tmp",
            model_manager=MagicMock(),
            skill_registry=MagicMock(),
            structured_output_model_registry=MagicMock(),
            logger=MagicMock(),
        )
        # Manually set attributes that BaseAgentBuilder would set
        builder.llm = mock_llm
        builder.tool_registry = mock_tool_registry
        builder.config_root = "/tmp"
        builder.model_manager = MagicMock()
        builder.logger = MagicMock()
        builder.document_loader = None
        builder.vector_store = None
        builder.skill_registry = MagicMock()
        builder.structured_output_model_registry=MagicMock()
        builder._tool_lock = asyncio.Lock()
        builder._kb_lock = asyncio.Lock()
        return builder

def test_init(builder, mock_llm, mock_tool_registry):
    assert builder.llm == mock_llm
    assert builder.tool_registry == mock_tool_registry
    assert builder.config_root == "/tmp"

@pytest.mark.asyncio
async def test_create_single_agent(builder):
    agent_config = {
        'system_prompt': 'You are a helper',
        'tools': ['tool1']
    }
    
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_agent') as mock_create_agent:
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        
        # Mock base class methods
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=['tool1_obj'])
        builder._load_mcp_tools = AsyncMock(return_value=[])
        builder._load_knowledge_base_tools = AsyncMock(return_value=[])
        
        agent = await builder.create_single_agent('test_agent', agent_config)
        
        assert agent == mock_agent
        mock_create_agent.assert_called_once()
        builder._get_regular_tools.assert_called_once()

@pytest.mark.asyncio
async def test_create_single_agent_with_dict_tools(builder):
    agent_config = {
        'system_prompt': 'You are a helper',
        'tools': {'tool1': {'arg': 'val'}}
    }
    
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_agent') as mock_create_agent:
        mock_agent = MagicMock()
        mock_create_agent.return_value = mock_agent
        
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=['tool1_obj'])
        builder._load_mcp_tools = AsyncMock(return_value=[])
        builder._load_knowledge_base_tools = AsyncMock(return_value=[])
        
        agent = await builder.create_single_agent('test_agent', agent_config)
        
        assert agent == mock_agent
        builder._get_regular_tools.assert_called_once()

@pytest.mark.asyncio
async def test_create_single_agent_with_mcps(builder):
    agent_config = {
        'system_prompt': 'helper',
        'mcps': {'mcp1': {}}
    }
    
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_agent'):
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=[])
        builder._load_mcp_tools = AsyncMock(return_value=['mcp_tool'])
        builder._load_knowledge_base_tools = AsyncMock(return_value=[])
        
        await builder.create_single_agent('test_agent', agent_config)
        
        builder._load_mcp_tools.assert_called_once()

@pytest.mark.asyncio
async def test_create_single_agent_with_kb(builder):
    agent_config = {
        'system_prompt': 'You are a helper',
        'knowledge_base': [{'name': 'kb1'}]
    }
    
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_agent') as mock_create_agent:
        builder._ensure_model = MagicMock()
        builder._get_regular_tools = AsyncMock(return_value=[])
        builder._load_mcp_tools = AsyncMock(return_value=[])
        builder._load_knowledge_base_tools = AsyncMock(return_value=['kb_tool'])
        
        await builder.create_single_agent('test_agent', agent_config)
        
        builder._load_knowledge_base_tools.assert_called_once()
        
        # Verify create_agent was called with the tool
        call_args = mock_create_agent.call_args
        assert 'kb_tool' in call_args[1]['tools']

@pytest.mark.asyncio
async def test_create_multi_agent_system(builder):
    agent_configs = [
        {'agent1': {'system_prompt': 'p1'}},
        {'agent2': {'system_prompt': 'p2'}}
    ]
    
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_supervisor') as mock_create_supervisor:
        # Mock base class methods
        builder._normalize_agent_configs = MagicMock(return_value=[('agent1', {}), ('agent2', {})])
        builder._create_agents_parallel = AsyncMock(return_value=([], [], []))
        builder._create_supervisor_agent = MagicMock(return_value=MagicMock())
        
        supervisor, base_agents = await builder.create_multi_agent_system(agent_configs)
        
        builder._normalize_agent_configs.assert_called_once()
        builder._create_agents_parallel.assert_called_once()
        builder._create_supervisor_agent.assert_called_once()

def test_validate_agent_config_valid(builder):
    config = {
        'system_prompt': 'prompt',
        'tools': ['tool1']
    }
    is_valid, errors = builder.validate_agent_config('agent1', config)
    assert is_valid is True
    assert len(errors) == 0

def test_validate_agent_config_missing_prompt(builder):
    config = {
        'tools': ['tool1']
    }
    is_valid, errors = builder.validate_agent_config('agent1', config)
    assert is_valid is False
    assert "missing system_prompt" in errors[0]

def test_validate_agent_config_invalid_tool_format(builder):
    config = {
        'system_prompt': 'prompt',
        'tools': 'not_a_list_or_dict'
    }
    is_valid, errors = builder.validate_agent_config('agent1', config)
    assert is_valid is False
    assert "invalid 'tools' configuration" in errors[0]

def test_validate_agent_config_invalid_tool(builder):
    config = {
        'system_prompt': 'prompt',
        'tools': ['unknown_tool']
    }
    builder.tool_registry.has_tool.return_value = False
    
    is_valid, errors = builder.validate_agent_config('agent1', config)
    assert is_valid is False
    assert "references unknown tool" in errors[0]

def test_validate_agent_config_invalid_mcp(builder):
    config = {
        'system_prompt': 'prompt',
        'mcps': 'not_a_dict'
    }
    is_valid, errors = builder.validate_agent_config('agent1', config)
    assert is_valid is False
    assert "invalid 'mcps' configuration" in errors[0]

def test_get_agent_summary(builder):
    config = {
        'system_prompt': 'prompt',
        'tools': ['t1', 't2'],
        'mcps': {'m1': {}}
    }
    summary = builder.get_agent_summary('agent1', config)
    assert summary['agent_name'] == 'agent1'
    assert summary['tool_count'] == 2
    assert summary['mcp_count'] == 1
    assert summary['total_tools'] == 3

def test_get_agent_summary_dict_tools(builder):
    config = {
        'tools': {'t1': {}},
    }
    summary = builder.get_agent_summary('agent1', config)
    assert summary['tool_count'] == 1

def test_get_agent_summary_no_tools(builder):
    config = {}
    summary = builder.get_agent_summary('agent1', config)
    assert summary['tool_count'] == 0

def test_extract_context_map(builder):
    configs = [
        {'agent1': {'context': ['agent2']}},
        {'agent2': {}}
    ]
    context_map = builder.extract_context_map(configs)
    assert context_map == {'agent1': ['agent2']}

def test_repr(builder):
    builder.tool_registry.tools = {'t1': 1}
    repr_str = repr(builder)
    assert "LangChainAgentBuilder" in repr_str
    assert "tools=1" in repr_str

@pytest.mark.asyncio
async def test_load_knowledge_base_tools(builder):
    # Mock the module import using sys.modules
    mock_kb_module = MagicMock()
    mock_kb_factory = MagicMock()
    mock_kb_tool = MagicMock()
    mock_kb_factory.return_value.create_tool.return_value = mock_kb_tool
    mock_kb_module.KnowledgeBaseFactory = mock_kb_factory
    
    with patch.dict('sys.modules', {'oai_agent_core.langgraph_core.components.knowledge.knowledge_base_factory': mock_kb_module}):
        with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
            mock_to_thread.return_value = mock_kb_factory.return_value
            
            # Call the method on the CLASS to ensure we use the implementation in AgentBuilder (via BaseAgentBuilder)
            # passing the builder instance as self
            tools = await AgentBuilder._load_knowledge_base_tools(builder, 'agent1', {'knowledge_base': [{'name': 'kb1'}]})
            
            assert len(tools) == 1
            assert tools[0] == mock_kb_tool

@pytest.mark.asyncio
async def test_create_agent_as_tool(builder):
    mock_agent = MagicMock()
    mock_agent.ainvoke = AsyncMock(return_value={'messages': [MagicMock(content='response')]})
    
    tool = builder._create_agent_as_tool(mock_agent, 'tool_name', 'tool_desc')
    
    assert tool.name == 'tool_name'
    assert tool.description == 'tool_desc'
    
    # Test execution
    result = await tool.ainvoke("query")
    assert result == 'response'
    mock_agent.ainvoke.assert_called_once()

def test_create_supervisor_agent_supervisor(builder):
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_supervisor') as mock_create:
        mock_supervisor = MagicMock()
        mock_create.return_value = mock_supervisor
        mock_supervisor.compile.return_value = 'compiled_supervisor'
        
        result = builder._create_supervisor_agent(
            { "pattern": Constants.PATTERN_SUPERVISOR},
            ['agent1'], 
            [], 
            'prompt'
        )
        
        assert result == 'compiled_supervisor'
        mock_create.assert_called_once()

def test_create_supervisor_agent_agent_as_tool(builder):
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_agent') as mock_create:
        mock_supervisor = MagicMock()
        mock_create.return_value = mock_supervisor
        mock_supervisor.compile.return_value = 'compiled_supervisor'
        
        result = builder._create_supervisor_agent(
            { "pattern":Constants.PATTERN_AGENT_AS_TOOL},
            [], 
            ['tool1'], 
            'prompt'
        )
        
        assert result == 'compiled_supervisor'
        mock_create.assert_called_once()

def test_create_supervisor_agent_swarm(builder):
    with patch('oai_agent_core.langgraph_core.builders.agent_builder.create_supervisor'):
        # Mock langgraph_swarm import
        with patch.dict('sys.modules', {'langgraph_swarm': MagicMock()}):
            import langgraph_swarm
            langgraph_swarm.create_swarm.return_value.compile.return_value = 'compiled_swarm'
            
            mock_agent = MagicMock()
            mock_agent.name = 'agent1'
            
            result = builder._create_supervisor_agent(
                { "pattern":Constants.PATTERN_SWARM},
                [mock_agent], 
                [], 
                'prompt'
            )
            
            assert result == 'compiled_swarm'

def test_create_supervisor_agent_unknown(builder):
    with pytest.raises(ValueError, match="Unknown architecture"):
        builder._create_supervisor_agent(
            { "pattern":"unknown"},
            [], 
            [], 
            'prompt'
        )
