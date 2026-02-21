import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
import types

# Robust mocking of oai_agent_core package structure
if 'oai_agent_core' not in sys.modules:
    oai_agent_core = types.ModuleType('oai_agent_core')
    oai_agent_core.__path__ = []
    sys.modules['oai_agent_core'] = oai_agent_core

if 'oai_agent_core.core' not in sys.modules:
    core_pkg = types.ModuleType('oai_agent_core.core')
    core_pkg.__path__ = []
    sys.modules['oai_agent_core.core'] = core_pkg
    sys.modules['oai_agent_core'].core = core_pkg

if 'oai_agent_core.processing' not in sys.modules:
    proc_pkg = types.ModuleType('oai_agent_core.processing')
    proc_pkg.__path__ = []
    sys.modules['oai_agent_core.processing'] = proc_pkg
    sys.modules['oai_agent_core.processing'].processing = proc_pkg

# Mock OutputSerializer and MessageFormatter
mock_serializer = MagicMock()
mock_formatter = MagicMock()
sys.modules['oai_agent_core.processing'].OutputSerializer = mock_serializer
sys.modules['oai_agent_core.processing'].MessageFormatter = mock_formatter

from oai_agent_core.crewai_core.agents.crewai_agent import CrewAIAgent

@pytest.fixture
def agent():
    config = {
        'mode': 'crew',
        'agent_list': [{'a1': {}}],
        'task_list': [{'t1': {'agent': 'a1'}}]
    }
    with patch('oai_agent_core.crewai_core.agents.crewai_agent.ModelConfigurationManager'), \
         patch('oai_agent_core.crewai_core.agents.crewai_agent.ToolRegistry'), \
         patch('oai_agent_core.crewai_core.agents.crewai_agent.OutputSerializer'), \
         patch('oai_agent_core.crewai_core.agents.crewai_agent.MessageFormatter'), \
         patch('oai_agent_core.crewai_core.agents.crewai_agent.CrewBuilder') as MockCrewBuilder, \
         patch('oai_agent_core.crewai_core.agents.crewai_agent.ResultExtractor'):
        
        # Configure the mock builder instance
        mock_builder_instance = MockCrewBuilder.return_value
        # Ensure build_crew_async is an AsyncMock so it can be awaited
        mock_builder_instance.build_crew_async = AsyncMock(return_value="crew_obj")
        
        return CrewAIAgent("test_agent", config, llm=MagicMock())

def test_init(agent):
    assert agent.agent_name == "test_agent"
    assert agent.execution_mode == "crew"
    assert agent._initialized is False

def test_initialize_crew(agent):
    # Ensure build_crew_async is set up correctly on the agent's builder instance
    # The fixture sets it up, but let's be explicit if needed or just verify
    
    async def run():
        res = await agent.initialize("sess_1")
        assert res == "crew_obj"
        assert agent._initialized is True
        agent.agent_builder.build_crew_async.assert_called()
        
    asyncio.run(run())

def test_initialize_flow(agent):
    agent.execution_mode = 'flow'
    agent.flow_builder = MagicMock()
    agent.flow_builder.build_flow.return_value = "flow_obj"
    
    async def run():
        res = await agent.initialize("sess_1")
        assert res == "flow_obj"
        agent.flow_builder.build_flow.assert_called()
        
    asyncio.run(run())

def test_process_request_crew(agent):
    agent.message_formatter.get_inputs.return_value = {'in': 'val'}
    agent.result_extractor.format_execution_result.return_value = {'res': 'ok'}
    
    # Mock initialize to return a mock context with kickoff
    mock_context = MagicMock()
    mock_context.kickoff_async = AsyncMock(return_value="result")
    agent.initialize = AsyncMock(return_value=mock_context)
    
    async def run():
        res = await agent.process_request("msg")
        assert res['res'] == 'ok'
        mock_context.kickoff_async.assert_called_with(inputs={'in': 'val'})
        
    asyncio.run(run())

def test_process_request_with_kb(agent):
    agent.global_kb_factory = MagicMock()
    agent.global_kb_factory.search_custom_knowledge_base.return_value = "KB Context"
    agent.message_formatter.get_inputs.return_value = {'topic': 'AI'}
    
    mock_context = MagicMock()
    mock_context.kickoff_async = AsyncMock(return_value="result")
    agent.initialize = AsyncMock(return_value=mock_context)
    
    async def run():
        await agent.process_request("msg")
        
        # Check inputs augmented
        call_args = mock_context.kickoff_async.call_args
        inputs = call_args[1]['inputs']
        assert "KB Context" in inputs['topic']
        
    asyncio.run(run())

def test_process_request_error(agent):
    agent.message_formatter.get_inputs.return_value = {}
    agent.initialize = AsyncMock(side_effect=ValueError("Init failed"))
    agent.result_extractor.extract_error_details.return_value = {'error_message': 'Init failed'}
    
    async def run():
        with pytest.raises(ValueError):
            await agent.process_request("msg")
            
    asyncio.run(run())

def test_ainvoke(agent):
    agent.process_request = AsyncMock(return_value={'session_id': 's1', 'result': 'r1'})
    agent.result_extractor.get_response.return_value = {'final': 'response'}
    agent.result_extractor.format_response.return_value = {'content': {'text': 'res'}}
    
    async def run():
        res = await agent.ainvoke("msg")
        assert res['content']['text'] == 'res'
        
    asyncio.run(run())

def test_invoke_sync(agent):
    agent.ainvoke = AsyncMock(return_value={'res': 'sync'})

    with patch('asyncio.run', return_value={'res': 'sync'}) as mock_run:
        res = agent.invoke("msg")
        assert res['res'] == 'sync'
        mock_run.assert_called_once()

def test_astream(agent):
    agent.process_request = AsyncMock(return_value={'session_id': 's1', 'result': 'r1'})
    
    async def mock_stream(*args):
        yield {'chunk': 1}
        
    agent.result_extractor.stream_response = mock_stream
    
    async def run():
        chunks = []
        async for chunk in agent.astream("msg"):
            chunks.append(chunk)
        assert len(chunks) == 1
        
    asyncio.run(run())

def test_validate_tasks(agent):
    agent.agent_builder.validate_configuration.return_value = {'valid': True}
    res = agent.validate_tasks()
    assert res['valid'] is True

def test_get_agent_info(agent):
    agent.llm.model = "gpt-4"
    agent.llm.provider = "openai"
    agent.validate_tasks = MagicMock(return_value={})
    
    info = agent.get_agent_info()
    assert info['agent_name'] == "test_agent"
    assert info['model']['name'] == "gpt-4"

def test_stream_warning(agent):
    agent.ainvoke = AsyncMock(return_value={'res': 'ok'})
    
    async def run():
        res = await agent.stream("msg")
        assert res['res'] == 'ok'
        agent.logger.warning.assert_called()
        
    asyncio.run(run())

def test_process_request_with_memory(agent):
    agent.memory_store = MagicMock()
    # Mock _get_conversation_context directly as it's inherited/mocked
    agent._get_conversation_context = MagicMock(return_value="Mem Context")
    agent.message_formatter.get_inputs.return_value = {'topic': 'AI'}
    
    mock_context = MagicMock()
    mock_context.kickoff_async = AsyncMock(return_value="result")
    agent.initialize = AsyncMock(return_value=mock_context)
    
    async def run():
        await agent.process_request("msg")
        
        # Check inputs augmented
        call_args = mock_context.kickoff_async.call_args
        inputs = call_args[1]['inputs']
        assert "Mem Context" in inputs['topic']
        
    asyncio.run(run())

def test_process_request_tracing(agent):
    agent.langfuse_manager.is_enabled = True
    agent.langfuse_manager.trace_generation.return_value.__enter__.return_value = MagicMock()
    
    agent.message_formatter.get_inputs.return_value = {'in': 'val'}
    
    mock_context = MagicMock()
    mock_context.kickoff_async = AsyncMock(return_value="result")
    agent.initialize = AsyncMock(return_value=mock_context)
    
    async def run():
        await agent.process_request("msg")
        agent.langfuse_manager.trace_generation.assert_called()
        
    asyncio.run(run())
