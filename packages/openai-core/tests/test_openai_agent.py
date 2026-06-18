import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_core.openai_core.agents.openai_agent import OpenAIAgent

@pytest.fixture
def mock_llm():
    llm = MagicMock()
    llm.model = "gpt-4o"
    return llm

@pytest.fixture
def agent(mock_llm):
    config = {
        'system_prompt': 'You are a helper',
        'tools': {'t1': {}}
    }
    return OpenAIAgent(
        agent_name="test_agent",
        agent_config=config,
        llm=mock_llm,
        config_root="/tmp"
    )

def test_init(agent):
    assert agent.agent_name == "test_agent"
    assert agent.is_multi_agent is False
    assert agent._initialized is False
    assert agent.agent_type == 'openai'

def test_initialize_single_agent(agent):
    async def run():
        # Mock AgentBuilder
        with patch('oai_agent_core.openai_core.agents.openai_agent.AgentBuilder') as mock_builder_cls:
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
    agent = OpenAIAgent("single_agent_wrapper", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.openai_core.agents.openai_agent.AgentBuilder') as mock_builder_cls:
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
    agent = OpenAIAgent("single_agent_wrapper", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.openai_core.agents.openai_agent.AgentBuilder') as mock_builder_cls:
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

def test_initialize_multi_agent(mock_llm):
    config = {
        'agent_list': [{'a1': {}}, {'a2': {}}],
        'system_prompt': 'supervisor prompt'
    }
    agent = OpenAIAgent("multi_agent", config, llm=mock_llm)
    
    async def run():
        with patch('oai_agent_core.openai_core.agents.openai_agent.AgentBuilder') as mock_builder_cls:
            mock_builder = MagicMock()
            mock_builder.create_multi_agent_system = AsyncMock(return_value=(MagicMock(), []))
            mock_builder_cls.return_value = mock_builder
            
            await agent.initialize()
            
            assert agent._initialized is True
            assert agent.is_multi_agent is True
            mock_builder.create_multi_agent_system.assert_called_once()
            
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
        
        # Mock Runner.run
        with patch('agents.Runner.run', new_callable=AsyncMock) as mock_run:
            mock_result = MagicMock()
            mock_result.final_output = "response"
            mock_run.return_value = mock_result
            
            # Mock MCP context
            agent._mcp_context = MagicMock()
            agent._mcp_context.return_value.__aenter__.return_value = None
            agent._mcp_context.return_value.__aexit__.return_value = None
            
            response = await agent.ainvoke("hello")
            
            assert 'content' in response
            mock_run.assert_called_once()
        
    asyncio.run(run())

def test_ainvoke_error(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        
        # Mock Runner.run to raise exception
        with patch('agents.Runner.run', new_callable=AsyncMock) as mock_run:
            mock_run.side_effect = Exception("Test Error")
            
            # Mock MCP context
            agent._mcp_context = MagicMock()
            agent._mcp_context.return_value.__aenter__.return_value = None
            agent._mcp_context.return_value.__aexit__.return_value = None
            
            with pytest.raises(Exception, match="Test Error"):
                await agent.ainvoke("hello")
        
    asyncio.run(run())

def test_astream(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        
        # Mock Runner.run_streamed
        with patch('agents.Runner.run_streamed') as mock_run_streamed:
            mock_result = MagicMock()
            
            # Mock stream events
            async def mock_events():
                event1 = MagicMock()
                event1.type = "run_item_stream_event"
                event1.item.type = "message_output_item"
                event1.item.content = "chunk1"
                yield event1
                
            mock_result.stream_events = mock_events
            mock_result.context_wrapper.usage = MagicMock()
            mock_run_streamed.return_value = mock_result
            
            # Mock MCP context
            agent._mcp_context = MagicMock()
            agent._mcp_context.return_value.__aenter__.return_value = None
            agent._mcp_context.return_value.__aexit__.return_value = None
            
            # Mock ItemHelpers
            with patch('agents.ItemHelpers.text_message_output', return_value="chunk1"):
                chunks = []
                async for chunk in agent.astream("hello"):
                    chunks.append(chunk)
                
                assert len(chunks) > 0
        
    asyncio.run(run())

def test_astream_error(agent):
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        
        # Mock Runner.run_streamed to raise exception
        with patch('agents.Runner.run_streamed') as mock_run_streamed:
            mock_run_streamed.side_effect = Exception("Stream Error")
            
            # Mock MCP context
            agent._mcp_context = MagicMock()
            agent._mcp_context.return_value.__aenter__.return_value = None
            agent._mcp_context.return_value.__aexit__.return_value = None
            
            chunks = []
            async for chunk in agent.astream("hello"):
                chunks.append(chunk)

            # The error path must honor the dict contract, not yield a raw string.
            assert len(chunks) == 1
            error_chunk = chunks[0]
            assert isinstance(error_chunk, dict)
            assert error_chunk['content']['type'] == 'error'
            assert "Error: Stream Error" in error_chunk['content']['text']
            assert error_chunk['final'] is True
            assert error_chunk['error'] == "Stream Error"
            assert 'session_id' in error_chunk
            assert 'model' in error_chunk

    asyncio.run(run())

def test_invoke_no_running_loop(agent):
    # No event loop running on this thread -> invoke() drives the coroutine
    # directly via asyncio.run().
    agent.ainvoke = AsyncMock(return_value={'content': 'sync'})

    result = agent.invoke("hello")

    assert result['content'] == 'sync'
    agent.ainvoke.assert_called_once()

def test_invoke_within_running_loop(agent):
    # When a loop is already running, invoke() must not raise; it runs ainvoke()
    # in a worker thread instead of calling asyncio.run() on the live loop.
    agent.ainvoke = AsyncMock(return_value={'content': 'async'})

    async def run():
        # Called synchronously from inside a running event loop.
        return agent.invoke("hello")

    result = asyncio.run(run())
    assert result['content'] == 'async'
    agent.ainvoke.assert_called_once()

def test_mcp_context(agent):
    async def run():
        agent.tool_registry.get_mcp_clients = MagicMock(return_value={'agent': ['client']})
        agent.agent = MagicMock()
        agent.agent.name = 'agent'
        
        with patch('agents.mcp.MCPServerManager') as mock_manager:
            mock_manager.return_value.__aenter__.return_value = None
            mock_manager.return_value.__aexit__.return_value = None
            
            async with agent._mcp_context():
                pass

            mock_manager.assert_called_once()

    asyncio.run(run())

def _make_message_chunk_event(text):
    """Build a fake message_output_item stream event yielding `text`."""
    event = MagicMock()
    event.type = "run_item_stream_event"
    event.item.type = "message_output_item"
    event.item.raw_item.status = "completed"
    return event

def test_astream_original_message_fallback(agent):
    # Regression: when config exists but lacks 'original_message', the streaming
    # path must fall back to the passed-in original_message (not None). Verify it
    # is what gets stored in memory.
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        agent.memory_store = MagicMock()
        agent.result_extractor.extract_text = MagicMock(return_value="answer")

        with patch('agents.Runner.run_streamed') as mock_run_streamed:
            mock_result = MagicMock()

            async def mock_events():
                yield _make_message_chunk_event("hi")

            mock_result.stream_events = mock_events
            mock_result.context_wrapper.usage = MagicMock()
            mock_result.final_output = "answer"
            mock_run_streamed.return_value = mock_result

            agent._mcp_context = MagicMock()
            agent._mcp_context.return_value.__aenter__.return_value = None
            agent._mcp_context.return_value.__aexit__.return_value = None

            with patch('agents.ItemHelpers.text_message_output', return_value="hi"):
                # config present but WITHOUT 'original_message'
                async for _ in agent.astream("user question", config={'verbose': True}):
                    pass

        agent.memory_store.add_turn.assert_called_once()
        kwargs = agent.memory_store.add_turn.call_args.kwargs
        assert kwargs['user_message'] == "user question"

    asyncio.run(run())

def test_astream_with_tracing_collects_dict_content(agent):
    # Regression: trace output aggregation must read content from the dict shape
    # returned by format_response (content is a dict, not a list).
    async def run():
        agent._ensure_initialized = AsyncMock()
        agent.agent = MagicMock()
        agent.memory_store = None
        agent.langfuse_manager = MagicMock()
        agent.langfuse_manager.is_enabled = True

        span = MagicMock()
        agent.langfuse_manager.trace_generation.return_value.__enter__.return_value = span
        agent.langfuse_manager.trace_generation.return_value.__exit__.return_value = None

        async def fake_inner(message, original_message, config):
            yield {'content': {'text': 'Hello ', 'type': 'text'}}
            yield {'content': {'text': 'world', 'type': 'text'}}

        agent._astream_without_tracing = fake_inner

        chunks = []
        async for chunk in agent._astream_with_tracing("msg", "orig", None):
            chunks.append(chunk)

        assert len(chunks) == 2
        # update_trace must be called with the aggregated text, not empty.
        agent.langfuse_manager.update_trace.assert_called_once()
        kwargs = agent.langfuse_manager.update_trace.call_args.kwargs
        assert kwargs['output_data'] == "Hello world"

    asyncio.run(run())
