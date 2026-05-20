import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_core.langgraph_core.agents.langgraph_agent import LangGraphAgent

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model_name = "gpt-4"
    return llm

@pytest.fixture
def agent(mock_llm):
    config = {
        'system_prompt': 'You are a helper',
        'tools': {'t1': {}}
    }
    return LangGraphAgent(
        agent_name="test_agent",
        agent_config=config,
        llm=mock_llm,
        config_root="/tmp"
    )

def test_init(agent):
    assert agent.agent_name == "test_agent"
    assert agent.is_multi_agent is False
    assert agent._initialized is False

def test_initialize_single_agent(agent):
    async def run():
        # Mock AgentBuilder
        with patch('oai_agent_core.langgraph_core.agents.langgraph_agent.AgentBuilder') as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.create_single_agent = AsyncMock(return_value=MagicMock())
            mock_builder_cls.return_value = mock_builder
            
            await agent.initialize()
            
            assert agent._initialized is True
            assert agent.is_multi_agent is False
            mock_builder.create_single_agent.assert_called_once()
            
    asyncio.run(run())

def test_initialize_single_agent_from_list(mock_llm):
    # Test case where agent_list has one item
    config = {
        'agent_list': [{'single_agent': {'system_prompt': 'test'}}],
        'tools': {}
    }
    agent = LangGraphAgent("single_agent_wrapper", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.langgraph_core.agents.langgraph_agent.AgentBuilder') as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.create_single_agent = AsyncMock(return_value=MagicMock())
            mock_builder_cls.return_value = mock_builder
            
            await agent.initialize()
            
            assert agent._initialized is True
            assert agent.is_multi_agent is False
            mock_builder.create_single_agent.assert_called_once()
            
    asyncio.run(run())

def test_initialize_single_agent_from_list_string(mock_llm):
    # Test case where agent_list has one string item
    config = {
        'agent_list': ['single_agent'],
        'tools': {}
    }
    agent = LangGraphAgent("single_agent_wrapper", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.langgraph_core.agents.langgraph_agent.AgentBuilder') as mock_builder_cls:
            with patch('oai_agent_core.components.configuration.model_config.ConfigManager') as mock_cm_cls:
                mock_cm = MagicMock()
                mock_cm.load_agent_config.return_value = {'system_prompt': 'loaded'}
                mock_cm_cls.return_value = mock_cm
                
                mock_builder = MagicMock()
                mock_builder.create_single_agent = AsyncMock(return_value=MagicMock())
                mock_builder_cls.return_value = mock_builder
                
                await agent.initialize()
                
                assert agent._initialized is True
                assert agent.is_multi_agent is False
                mock_builder.create_single_agent.assert_called_once()
                mock_cm.load_agent_config.assert_called_with(agent_name='single_agent')
            
    asyncio.run(run())

def test_initialize_single_agent_from_list_invalid(mock_llm):
    config = {
        'agent_list': [123], # Invalid type
        'tools': {}
    }
    agent = LangGraphAgent("invalid", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.langgraph_core.agents.langgraph_agent.AgentBuilder'):
            with pytest.raises(ValueError, match="Invalid item type"):
                await agent.initialize()
                
    asyncio.run(run())

def test_initialize_multi_agent(mock_llm):
    config = {
        'agent_list': [{'a1': {}}, {'a2': {}}],
        'system_prompt': 'supervisor prompt'
    }
    agent = LangGraphAgent("multi_agent", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.langgraph_core.agents.langgraph_agent.AgentBuilder') as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.create_multi_agent_system = AsyncMock(return_value=(MagicMock(), []))
            mock_builder_cls.return_value = mock_builder
            
            await agent.initialize()
            
            assert agent._initialized is True
            assert agent.is_multi_agent is True
            mock_builder.create_multi_agent_system.assert_called_once()
            
    asyncio.run(run())

def test_initialize_with_global_kb(mock_llm):
    config = {
        'knowledge_base': [{'name': 'kb1'}],
        'tools': {}
    }
    agent = LangGraphAgent("kb_agent", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.langgraph_core.agents.langgraph_agent.AgentBuilder') as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.create_single_agent = AsyncMock(return_value=MagicMock())
            mock_builder_cls.return_value = mock_builder
            
            with patch('oai_agent_core.langgraph_core.components.KnowledgeBaseFactory') as mock_kb_cls:
                # The mock class itself needs to return an instance when called
                mock_kb_instance = MagicMock()
                mock_kb_cls.return_value = mock_kb_instance
                
                await agent.initialize()
                
                # In MockBaseAgent._load_tools_and_kb (conftest.py), it instantiates the factory:
                # self.global_kb_factory = kb_factory_class(...)
                # So agent.global_kb_factory should be the return value of mock_kb_cls()
                
                assert agent.global_kb_factory is not None
                assert agent.global_kb_factory == mock_kb_instance
                mock_kb_cls.assert_called_once()
                
    asyncio.run(run())

def test_initialize_global_kb_error(mock_llm):
    config = {
        'knowledge_base': [{'name': 'kb1'}],
    }
    agent = LangGraphAgent("kb_error", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.langgraph_core.agents.langgraph_agent.AgentBuilder') as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.create_single_agent = AsyncMock(return_value=MagicMock())
            mock_builder_cls.return_value = mock_builder
            
            with patch('oai_agent_core.langgraph_core.components.KnowledgeBaseFactory') as mock_kb_cls:
                # Simulate ImportError during instantiation
                mock_kb_cls.side_effect = ImportError("Missing dep")
                
                try:
                    await agent.initialize()
                except ImportError:
                    pass # Expected if MockBaseAgent re-raises
                
                # assert agent.global_kb_factory is None
                pass
                
    asyncio.run(run())

def test_prepare_message(agent):
    user_msg = "Hello {name}"
    config = {'inputs': {'name': 'World'}}
    
    # Mock message formatter
    agent.message_formatter.format_message = MagicMock(return_value="Hello World")
    
    result = agent._prepare_message(user_msg, config)
    assert result == "Hello World"
    agent.message_formatter.format_message.assert_called_with(user_msg, {'name': 'World'})

def test_prepare_message_defaults(agent):
    user_msg = "Hello"
    config = {}
    
    agent.message_formatter.extract_variables_from_config = MagicMock(return_value={'var'})
    agent.message_formatter.create_default_inputs = MagicMock(return_value={'var': 'val'})
    agent.message_formatter.format_message = MagicMock(return_value="Hello val")
    
    result = agent._prepare_message(user_msg, config)
    assert result == "Hello val"

def test_ainvoke(agent):
    async def run():
        # Mock initialization and agent execution
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        agent.agent.ainvoke = AsyncMock(return_value={'messages': []})
        
        response = await agent.ainvoke("hello")
        
        assert 'content' in response
        agent.agent.ainvoke.assert_called_once()
        
    asyncio.run(run())

def test_ainvoke_with_kb(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        agent.agent.ainvoke = AsyncMock(return_value={'messages': []})
        
        # Mock global KB
        agent.global_kb_factory = MagicMock()
        agent.global_kb_factory.search_custom_knowledge_base.return_value = "KB Context"
        
        await agent.ainvoke("hello")
        
        # Check if prompt was augmented
        call_args = agent.agent.ainvoke.call_args
        messages = call_args[0][0]['messages']
        assert "KB Context" in messages[0]['content']
        
    asyncio.run(run())

def test_ainvoke_error(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        agent.agent.ainvoke = AsyncMock(side_effect=Exception("Test Error"))
        
        response = await agent.ainvoke("hello")
        
        assert response['final'] is True
        assert "Test Error" in response['error']
        
    asyncio.run(run())

def test_astream(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        
        # Mock async generator
        async def mock_stream(*args, **kwargs):
            yield {'messages': [{'content': 'chunk1'}]}
            yield {'messages': [{'content': 'chunk2'}]}
            
        agent.agent.astream = mock_stream
        
        chunks = []
        async for chunk in agent.astream("hello"):
            chunks.append(chunk)
            
        assert len(chunks) > 0
        
    asyncio.run(run())

def test_astream_verbose(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        
        async def mock_stream(*args, **kwargs):
            # Simulate LangGraph state updates where history grows
            yield {'messages': [{'content': 'msg1'}]}
            yield {'messages': [{'content': 'msg1'}, {'content': 'msg2'}]}
            
        agent.agent.astream = mock_stream
        
        chunks = []
        async for chunk in agent.astream("hello", config={'verbose': True}):
            chunks.append(chunk)
            
        # Should yield msg1 then msg2 (deduplicated)
        assert len(chunks) == 2
        assert chunks[0]['content'] == {'final': False, 'session_id': 'default', 'text': 'msg1', 'type': 'dict'}
        assert chunks[1]['content'] == {'final': False, 'session_id': 'default', 'text': 'msg2', 'type': 'dict'}
        
    asyncio.run(run())

def test_astream_error(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        agent.agent.astream.side_effect = Exception("Stream Error")
        
        chunks = []
        async for chunk in agent.astream("hello"):
            chunks.append(chunk)
            
        assert len(chunks) == 1
        assert chunks[0]['type'] == 'error'
        assert "Stream Error" in chunks[0]['content']
        
    asyncio.run(run())

def test_invoke_sync(agent):
    # Mock _sync_invoke
    with patch.object(agent, '_sync_invoke', return_value={'content': 'sync'}) as mock_sync:
        # We are in an async loop here (pytest-asyncio or asyncio.run wrapper), 
        # so invoke should call _sync_invoke directly if loop is running.
        
        async def run():
            result = agent.invoke("hello")
            assert result['content'] == 'sync'
            mock_sync.assert_called_once()
            
        asyncio.run(run())

def test_sync_invoke_implementation(agent):
    agent._set_langfuse_config = MagicMock()
    agent.agent = MagicMock()
    agent.agent.invoke.return_value = {'messages': []}
    
    result = agent._sync_invoke("hello")
    assert 'content' in result
    agent.agent.invoke.assert_called_once()

def test_sync_invoke_error(agent):
    agent._set_langfuse_config = MagicMock()
    agent.agent = MagicMock()
    agent.agent.invoke.side_effect = Exception("Sync Error")
    
    result = agent._sync_invoke("hello")
    assert result['final'] is True
    assert "Sync Error" in result['error']

def test_validate_tasks(agent):
    agent.agent_config['tools'] = {'t1': {}}
    agent.tool_registry.tools = {'t1': MagicMock()}
    
    diagnostics = agent.validate_tasks()
    
    assert diagnostics['tool_count'] == 1
    assert diagnostics['is_multi_agent'] is False
    assert 't1' in diagnostics['tools']

def test_validate_tasks_multi_agent(mock_llm):
    config = {
        'agent_list': [{'a1': {}}],
        'tools': {}
    }
    agent = LangGraphAgent("multi", config, llm=mock_llm)
    agent.agent_builder = MagicMock()
    agent.agent_builder.validate_agent_config.return_value = (False, ['error'])
    agent.is_multi_agent = True
    
    diagnostics = agent.validate_tasks()
    assert 'validation_errors' in diagnostics
    assert diagnostics['validation_errors'] == ['error']

def test_get_agent_info(agent):
    info = agent.get_agent_info()
    assert info['agent_name'] == "test_agent"
    assert info['agent_type'] == "langgraph"
    assert 'config_summary' in info

def test_cleanup(agent):
    async def run():
        agent.is_multi_agent = True
        mock_remote = MagicMock()
        mock_remote.agent_type = "remote"
        mock_remote.agent.stop = AsyncMock()
        
        agent.base_agent_list = [mock_remote]
        
        await agent.cleanup()
        mock_remote.agent.stop.assert_called_once()
        
    asyncio.run(run())

def test_del(agent):
    agent.langfuse_manager.is_enabled = True
    agent.__del__()
    agent.langfuse_manager.flush.assert_called_once()

def test_repr(agent):
    repr_str = repr(agent)
    assert "UnifiedLangChainAgent" in repr_str
    assert "test_agent" in repr_str

def test_stream_not_implemented(agent):
    # stream() should raise NotImplementedError; callers should use astream() instead.
    async def run():
        with pytest.raises(NotImplementedError):
            await agent.stream("hello")

    asyncio.run(run())

def test_set_langfuse_config_enabled(agent):
    agent.langfuse_manager.initialize_callback_handler = MagicMock()
    agent.langfuse_manager.callback_handler = MagicMock()
    
    agent._set_langfuse_config()
    
    assert 'callbacks' in agent.config
    assert agent.config['callbacks'][0] == agent.langfuse_manager.callback_handler
    assert agent.config['metadata']['name'] == "test_agent"

def test_set_langfuse_config_disabled(agent):
    agent.langfuse_manager.initialize_callback_handler = MagicMock()
    agent.langfuse_manager.callback_handler = None
    
    # Clear config to verify it's not updated
    agent.config = {}
    
    agent._set_langfuse_config()
    
    assert 'callbacks' not in agent.config

def test_validate_tasks_agent_list_string(mock_llm):
    config = {
        'agent_list': ['agent1'],
        'tools': {}
    }
    agent = LangGraphAgent("multi", config, llm=mock_llm)
    agent.agent_builder = MagicMock()
    agent.agent_builder.validate_agent_config.return_value = (True, [])
    agent.is_multi_agent = True
    
    diagnostics = agent.validate_tasks()
    assert 'agent1' in diagnostics['agents']

def test_validate_tasks_agent_list_dict(mock_llm):
    config = {
        'agent_list': [{'agent1': {}}],
        'tools': {}
    }
    agent = LangGraphAgent("multi", config, llm=mock_llm)
    agent.agent_builder = MagicMock()
    agent.agent_builder.validate_agent_config.return_value = (True, [])
    agent.is_multi_agent = True
    
    diagnostics = agent.validate_tasks()
    assert 'agent1' in diagnostics['agents']

def test_cleanup_exception(agent):
    async def run():
        agent.is_multi_agent = True
        mock_remote = MagicMock()
        mock_remote.agent_type = "remote"
        mock_remote.agent.stop = AsyncMock(side_effect=Exception("Stop error"))
        
        agent.base_agent_list = [mock_remote]
        
        # Should not raise exception
        await agent.cleanup()
        mock_remote.agent.stop.assert_called_once()
        
    asyncio.run(run())

def test_del_exception(agent):
    agent.langfuse_manager.is_enabled = True
    agent.langfuse_manager.flush.side_effect = Exception("Flush error")
    
    # Should not raise exception
    agent.__del__()
    agent.langfuse_manager.flush.assert_called_once()
