import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_core.crewai_core.builders.crew_builder import CrewBuilder

@pytest.fixture
def mock_registry():
    reg = MagicMock()
    reg.get_tools_for_agent.return_value = []
    reg.get_mcp_configs.return_value = {}
    reg.load_mcp_tools_from_config.return_value = []
    return reg

@pytest.fixture
def mock_output_model_registry():
    reg = MagicMock()
    reg.get_model.return_value = MagicMock()
    return reg

@pytest.fixture
def builder(mock_registry, mock_output_model_registry):
    config = {
        'agent_list': [{'agent1': {'role': 'role1', 'goal': 'goal1', 'backstory': 'backstory1'}}],
        'task_list': [{'task1': {'description': 'desc', 'expected_output': 'out', 'agent': 'agent1'}}],
        'crew_config': {'process': 'sequential'}
    }
    return CrewBuilder(config, mock_registry, MagicMock(), MagicMock(), structured_output_model_registry=mock_output_model_registry)


# ──────────────────────────────────────────────────────────────────────────────
# Tests for _resolve_agent_entry (string vs dict entries in agent_list)
# ──────────────────────────────────────────────────────────────────────────────

def test_resolve_agent_entry_dict(builder):
    """Dict entries are resolved directly without any file I/O."""
    entry = {'researcher': {'role': 'Researcher', 'goal': 'Research', 'backstory': 'Expert'}}
    agent_key, agent_data = builder._resolve_agent_entry(entry)
    assert agent_key == 'researcher'
    assert agent_data['role'] == 'Researcher'


def test_resolve_agent_entry_string_loads_external_config(builder):
    """String entries load config via ConfigManager."""
    external_config = {'role': 'Writer', 'goal': 'Write', 'backstory': 'Skilled writer'}
    with patch('oai_agent_core.crewai_core.builders.crew_builder.ConfigManager') as MockCM:
        MockCM.return_value.load_agent_config.return_value = external_config
        agent_key, agent_data = builder._resolve_agent_entry('writer')
    assert agent_key == 'writer'
    assert agent_data['role'] == 'Writer'
    MockCM.return_value.load_agent_config.assert_called_once_with(agent_name='writer')


def test_resolve_agent_entry_string_file_not_found(builder):
    """String entries raise FileNotFoundError when the YAML file is missing."""
    with patch('oai_agent_core.crewai_core.builders.crew_builder.ConfigManager') as MockCM:
        MockCM.return_value.load_agent_config.side_effect = FileNotFoundError("not found")
        with pytest.raises(FileNotFoundError, match="agents_config/missing_agent.yaml"):
            builder._resolve_agent_entry('missing_agent')


def test_resolve_agent_entry_invalid_type(builder):
    """Non-string, non-dict entries raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported agent_list entry type"):
        builder._resolve_agent_entry(42)


def test_build_crew_with_string_entry(builder):
    """build_crew works when agent_list contains a string entry."""
    external_config = {'role': 'Analyst', 'goal': 'Analyse', 'backstory': 'Expert analyst'}
    builder.config['agent_list'] = ['analyst']
    builder.config['task_list'] = [{'task1': {'description': 'analyse', 'expected_output': 'report', 'agent': 'analyst'}}]

    with patch('oai_agent_core.crewai_core.builders.crew_builder.ConfigManager') as MockCM, \
         patch('oai_agent_core.crewai_core.builders.crew_builder.Agent') as MockAgent, \
         patch('oai_agent_core.crewai_core.builders.task_builder.Task'), \
         patch('oai_agent_core.crewai_core.builders.crew_builder.Crew') as MockCrew:
        MockCM.return_value.load_agent_config.return_value = external_config
        crew = builder.build_crew("sess_1")
        MockAgent.assert_called_once()
        MockCrew.assert_called_once()
        assert crew == MockCrew.return_value


def test_build_crew_async_with_string_entry(builder):
    """build_crew_async works when agent_list contains a string entry."""
    external_config = {'role': 'Analyst', 'goal': 'Analyse', 'backstory': 'Expert analyst'}
    builder.config['agent_list'] = ['analyst']
    builder.config['task_list'] = [{'task1': {'description': 'analyse', 'expected_output': 'report', 'agent': 'analyst'}}]

    async def run():
        with patch('oai_agent_core.crewai_core.builders.crew_builder.ConfigManager') as MockCM, \
             patch('oai_agent_core.crewai_core.builders.crew_builder.Agent') as MockAgent, \
             patch('oai_agent_core.crewai_core.builders.task_builder.Task'), \
             patch('oai_agent_core.crewai_core.builders.crew_builder.Crew') as MockCrew:
            MockCM.return_value.load_agent_config.return_value = external_config
            crew = await builder.build_crew_async("sess_1")
            MockAgent.assert_called_once()
            MockCrew.assert_called_once()
            assert crew == MockCrew.return_value

    asyncio.run(run())



def test_build_crew(builder):
    # We need to patch Task in task_builder.py as well because CrewBuilder uses TaskBuilder
    with patch('oai_agent_core.crewai_core.builders.crew_builder.Agent') as MockAgent, \
         patch('oai_agent_core.crewai_core.builders.task_builder.Task') as MockTask, \
         patch('oai_agent_core.crewai_core.builders.crew_builder.Crew') as MockCrew:
        
        # build_crew calls _build_crew_sync internally
        crew = builder.build_crew("sess_1")
        
        MockAgent.assert_called()
        MockTask.assert_called()
        MockCrew.assert_called()
        assert crew == MockCrew.return_value

def test_build_crew_async(builder):
    async def run():
        with patch('oai_agent_core.crewai_core.builders.crew_builder.Agent') as MockAgent, \
             patch('oai_agent_core.crewai_core.builders.task_builder.Task') as MockTask, \
             patch('oai_agent_core.crewai_core.builders.crew_builder.Crew') as MockCrew:
            
            crew = await builder.build_crew_async("sess_1")
            
            MockAgent.assert_called()
            MockTask.assert_called()
            MockCrew.assert_called()
            assert crew == MockCrew.return_value
            
    asyncio.run(run())

def test_build_crew_no_tasks(builder):
    builder.config['task_list'] = []
    # Also clear agent tasks if any
    builder.config['agent_list'][0]['agent1'].pop('tasks', None)
    
    with patch('oai_agent_core.crewai_core.builders.crew_builder.Agent'):
        with pytest.raises(ValueError, match="No tasks defined"):
            builder.build_crew("sess_1")

def test_build_crew_async_no_tasks(builder):
    builder.config['task_list'] = []
    
    async def run():
        with patch('oai_agent_core.crewai_core.builders.crew_builder.Agent'):
            with pytest.raises(ValueError, match="No tasks defined"):
                await builder.build_crew_async("sess_1")
                
    asyncio.run(run())

def test_prepare_tools(builder):
    builder.config['agent_list'][0]['agent1']['tools'] = ['t1']
    builder.config['agent_list'][0]['agent1']['mcps'] = {'m1': {}}
    builder.config['agent_list'][0]['agent1']['knowledge_base'] = [{'name': 'kb1'}]
    
    builder.tool_registry.get_tools_for_agent.return_value = ['tool_obj']
    builder.tool_registry.load_mcp_tools_from_config.return_value = ['mcp_obj']
    
    with patch('oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory.KnowledgeBaseFactory') as MockKB:
        MockKB.return_value.create_tool.return_value = "kb_tool"
        
        tools, mcps = builder._prepare_tools('agent1', builder.config['agent_list'][0]['agent1'])
        
        assert 'tool_obj' in tools
        assert 'kb_tool' in tools
        assert 'mcp_obj' in mcps
        MockKB.assert_called()

def test_prepare_tools_kb_error(builder):
    builder.config['agent_list'][0]['agent1']['knowledge_base'] = [{'name': 'kb1'}]
    
    with patch('oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory.KnowledgeBaseFactory') as MockKB:
        MockKB.side_effect = ImportError("Missing dep")
        
        tools, mcps = builder._prepare_tools('agent1', builder.config['agent_list'][0]['agent1'])
        
        # Should catch exception and log warning
        assert len(tools) == 0
        builder.logger.warning.assert_called()

def test_prepare_tools_kb_generic_error(builder):
    builder.config['agent_list'][0]['agent1']['knowledge_base'] = [{'name': 'kb1'}]
    
    with patch('oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory.KnowledgeBaseFactory') as MockKB:
        MockKB.side_effect = Exception("Generic error")
        
        tools, mcps = builder._prepare_tools('agent1', builder.config['agent_list'][0]['agent1'])
        
        assert len(tools) == 0
        builder.logger.error.assert_called()

def test_prepare_tools_async(builder):
    builder.config['agent_list'][0]['agent1']['tools'] = ['t1']
    builder.config['agent_list'][0]['agent1']['mcps'] = {'m1': {}}
    builder.config['agent_list'][0]['agent1']['knowledge_base'] = [{'name': 'kb1'}]
    
    builder.tool_registry.get_tools_for_agent.return_value = ['tool_obj']
    
    async def run():
        with patch('oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory.KnowledgeBaseFactory') as MockKB:
            MockKB.return_value.create_tool.return_value = "kb_tool"
            
            # Mock to_thread for MCP loading and KB init
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                # Configure side effects for to_thread calls
                # 1. load_mcp_tools_from_config -> ['mcp_obj']
                # 2. KnowledgeBaseFactory -> MockKB.return_value
                
                mock_to_thread.side_effect = [['mcp_obj'], MockKB.return_value]
                
                tools, mcps = await builder._prepare_tools_async('agent1', builder.config['agent_list'][0]['agent1'])
                
                assert 'tool_obj' in tools
                assert 'kb_tool' in tools
                assert 'mcp_obj' in mcps
                
    asyncio.run(run())

def test_prepare_tools_async_kb_error(builder):
    builder.config['agent_list'][0]['agent1']['knowledge_base'] = [{'name': 'kb1'}]
    
    async def run():
        with patch('oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory.KnowledgeBaseFactory'):
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                # First call (MCP) succeeds (empty list), second call (KB) raises
                mock_to_thread.side_effect = [[], ImportError("Missing dep")]
                
                tools, mcps = await builder._prepare_tools_async('agent1', builder.config['agent_list'][0]['agent1'])
                
                assert len(tools) == 0
                builder.logger.warning.assert_called()
                
    asyncio.run(run())

def test_prepare_tools_async_kb_generic_error(builder):
    builder.config['agent_list'][0]['agent1']['knowledge_base'] = [{'name': 'kb1'}]
    
    async def run():
        with patch('oai_agent_core.crewai_core.components.knowledge.knowledge_base_factory.KnowledgeBaseFactory'):
            with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
                mock_to_thread.side_effect = [[], Exception("Generic error")]
                
                tools, mcps = await builder._prepare_tools_async('agent1', builder.config['agent_list'][0]['agent1'])
                
                assert len(tools) == 0
                builder.logger.error.assert_called()
                
    asyncio.run(run())

def test_create_task_invalid_agent(builder):
    builder.config['task_list'][0]['task1']['agent'] = 'unknown'
    
    with patch('oai_agent_core.crewai_core.builders.crew_builder.Agent'):
        with pytest.raises(ValueError, match="Agent 'unknown' not found"):
            builder.build_crew("sess_1")

def test_validate_configuration(builder):
    builder.config['tools'] = {'t1': {}}
    builder.config['agent_list'][0]['agent1']['tools'] = ['t1']
    
    diag = builder.validate_configuration()
    
    assert diag['agent_count'] == 1
    assert diag['task_count'] == 1
    assert diag['tool_count'] == 1
    assert len(diag['missing_tools']) == 0

def test_validate_configuration_missing_tool(builder):
    builder.config['agent_list'][0]['agent1']['tools'] = ['unknown']
    
    diag = builder.validate_configuration()
    
    assert len(diag['missing_tools']) == 1
    assert "unknown" in diag['missing_tools'][0]

def test_extract_input_variables(builder):
    task_data = {
        'description': 'Do {task} for {person}',
        'expected_output': 'Report for {person}'
    }
    vars_set = set()
    builder._extract_input_variables(task_data, vars_set)
    assert 'task' in vars_set
    assert 'person' in vars_set
    assert len(vars_set) == 2
