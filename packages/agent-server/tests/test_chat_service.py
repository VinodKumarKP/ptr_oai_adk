import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from oai_agent_server.services.chat_service import ChatService
from oai_agent_server.models.requests import ChatRequest, StreamChatRequest
from oai_agent_server.exceptions import StreamingException
from fastapi import HTTPException

@pytest.fixture
def chat_service(mock_agent, mock_db_logger):
    return ChatService(mock_agent, mock_db_logger, MagicMock())

@pytest.mark.asyncio
async def test_process_chat_success(chat_service):
    request = ChatRequest(message="Hello", user_id="user1")
    headers = {"header": "value"}
    
    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "Response"})
    
    response = await chat_service.process_chat(request, headers)
    
    assert response.status_code == 200
    content = response.body.decode()
    assert "Response" in content
    chat_service.db_logger.log_interaction.assert_called_once()

@pytest.mark.asyncio
async def test_process_chat_with_files(chat_service):
    request = ChatRequest(message="Analyze this", user_id="user1")
    headers = {}
    files = ["/tmp/file1.txt"]
    
    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "File analyzed"})
    
    await chat_service.process_chat(request, headers, files=files)
    
    call_args = chat_service.agent.ainvoke.call_args
    assert call_args.kwargs['config']['uploaded_files'] == files

@pytest.mark.asyncio
async def test_process_chat_with_files_in_message(chat_service):
    request = ChatRequest(message={"content": "Analyze", "uploaded_files": ["/tmp/file2.txt"]}, user_id="user1")
    headers = {}
    
    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "File analyzed"})
    
    await chat_service.process_chat(request, headers)
    
    call_args = chat_service.agent.ainvoke.call_args
    assert call_args.kwargs['config']['uploaded_files'] == ["/tmp/file2.txt"]

@pytest.mark.asyncio
async def test_process_chat_exception(chat_service):
    request = ChatRequest(message="Hello", user_id="user1")
    chat_service.agent.ainvoke = AsyncMock(side_effect=Exception("Agent error"))
    
    with pytest.raises(HTTPException) as excinfo:
        await chat_service.process_chat(request, {})
    assert excinfo.value.status_code == 500

@pytest.mark.asyncio
async def test_process_stream_chat(chat_service):
    request = StreamChatRequest(message="Stream me", user_id="user1")
    headers = {}
    
    async def mock_stream(*args, **kwargs):
        yield {"content": "Chunk 1"}
        yield {"content": "Chunk 2"}
        
    chat_service.agent.astream = mock_stream
    
    response = await chat_service.process_stream_chat(request, headers)
    
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode('utf-8')
        chunks.append(chunk)
        
    assert len(chunks) > 0
    assert any("Chunk 1" in chunk for chunk in chunks)
    
    chat_service.db_logger.log_stream_chunks_batch.assert_called_once()
    chat_service.db_logger.log_interaction.assert_called_once()

@pytest.mark.asyncio
async def test_process_stream_chat_non_serializable(chat_service):
    request = StreamChatRequest(message="Stream me", user_id="user1")
    
    class NonSerializable:
        pass
        
    async def mock_stream(*args, **kwargs):
        # This will cause make_serializable to return string representation
        # But json.dumps handles strings fine.
        # To trigger the TypeError in json.dumps, we need something make_serializable doesn't handle?
        # make_serializable handles almost everything by converting to str as last resort.
        # The code block:
        # try: json.dumps(...) except: try: make_serializable(...) except: ...
        # So we need json.dumps to fail first.
        # NonSerializable() object will fail json.dumps.
        yield {"content": NonSerializable()}
        
    chat_service.agent.astream = mock_stream
    
    response = await chat_service.process_stream_chat(request, {})
    
    chunks = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, bytes):
            chunk = chunk.decode('utf-8')
        chunks.append(chunk)
    
    # make_serializable converts it to string, so it succeeds in the second try block.
    # It does NOT go to the third block (serialization_warning) unless make_serializable fails.
    # make_serializable is very robust.
    # To trigger the warning, we need make_serializable to fail or json.dumps of serializable content to fail.
    # Let's mock make_serializable to raise exception.
    
    with patch('oai_agent_server.services.chat_service.make_serializable', side_effect=Exception("Serialize error")):
        response = await chat_service.process_stream_chat(request, {})
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')
            chunks.append(chunk)
            
        assert any("serialization_warning" in chunk for chunk in chunks)

@pytest.mark.asyncio
async def test_process_stream_chat_exception(chat_service):
    request = StreamChatRequest(message="Stream me", user_id="user1")
    # The exception happens when calling astream, which is inside generate_response
    # But generate_response is called when iterating the response.
    # Wait, process_stream_chat calls astream inside generate_response?
    # No, it calls astream inside generate_response.
    # So we need to iterate the response to trigger the error.
    
    chat_service.agent.astream = MagicMock(side_effect=Exception("Stream init error"))
    
    response = await chat_service.process_stream_chat(request, {})
    
    # The exception is raised inside the generator, so iterating it should raise StreamingException
    # But StreamingResponse might handle it or suppress it?
    # Actually, process_stream_chat returns StreamingResponse.
    # The generator is passed to StreamingResponse.
    # If the generator raises, StreamingResponse stops.
    # However, the test framework might not see the exception unless we iterate.
    
    # Wait, the implementation wraps the whole thing in try/except and raises StreamingException.
    # But the try/except block in process_stream_chat only covers the setup, NOT the generator execution?
    # Let's check the code.
    # try: ... return StreamingResponse(...) except: raise StreamingException
    # The generator is defined inside.
    # If astream raises immediately when called (not when iterated), then it depends on where it's called.
    # It's called inside generate_response.
    # generate_response is called by StreamingResponse when iterating.
    # So process_stream_chat returns successfully.
    # The exception happens during iteration.
    
    # To test the try/except block in process_stream_chat, we need an error BEFORE return StreamingResponse.
    # e.g. _handle_reinitialization failure.
    
    chat_service._handle_reinitialization = AsyncMock(side_effect=Exception("Init error"))
    
    with pytest.raises(StreamingException):
        await chat_service.process_stream_chat(request, {})

@pytest.mark.asyncio
async def test_handle_reinitialization(chat_service):
    with patch('os.environ.get', return_value='true'):
        chat_service.agent.initialize = AsyncMock()
        await chat_service._handle_reinitialization("user1")
        chat_service.agent.initialize.assert_called_once()
        assert chat_service.agent.user_id == "user1"

@pytest.mark.asyncio
async def test_handle_reinitialization_no_user(chat_service):
    with patch('os.environ.get', return_value='true'):
        chat_service.agent.initialize = AsyncMock()
        await chat_service._handle_reinitialization(None)
        assert chat_service.agent.user_id == "default_user"
