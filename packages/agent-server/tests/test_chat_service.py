import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from oai_agent_server.services.chat_service import ChatService
from oai_agent_server.models.requests import ChatRequest, StreamChatRequest
from oai_agent_server.exceptions import StreamingException
from fastapi import HTTPException, Request


class MockState:
    def __init__(self, user_email=None, user_id=None, user_role=None):
        if user_email:
            self.user_email = user_email
        if user_id:
            self.user_id = user_id
        if user_role:
            self.user_role = user_role


class MockRequest:
    def __init__(self, state=None, headers=None):
        self._state = state or MockState()
        self._headers = headers or {}

    @property
    def state(self):
        return self._state

    @property
    def headers(self):
        return self._headers


@pytest.fixture
def chat_service(mock_agent, mock_db_logger):
    return ChatService(mock_agent, mock_db_logger, MagicMock())


@pytest.mark.asyncio
async def test_process_chat_success(chat_service):
    chat_request = ChatRequest(message="Hello", user_id="user1")
    http_request = MockRequest(headers={"header": "value"})

    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "Response"})

    response = await chat_service.process_chat(http_request, chat_request, http_request.headers)

    assert response.status_code == 200
    content = response.body.decode()
    assert "Response" in content
    chat_service.db_logger.log_interaction.assert_called_once()


@pytest.mark.asyncio
async def test_process_chat_with_saml_user(chat_service):
    chat_request = ChatRequest(message="Hello", user_id="user1")
    http_request = MockRequest(state=MockState(user_email="saml.user@example.com"))

    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "Response"})

    await chat_service.process_chat(http_request, chat_request, http_request.headers)

    call_args = chat_service.db_logger.log_interaction.call_args
    assert call_args.kwargs['user_id'] == "saml.user@example.com"


@pytest.mark.asyncio
async def test_process_chat_with_api_token_user(chat_service):
    chat_request = ChatRequest(message="Hello") # No user_id in body
    http_request = MockRequest(state=MockState(user_id="api_user_from_token"))

    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "Response"})

    await chat_service.process_chat(http_request, chat_request, http_request.headers)

    call_args = chat_service.db_logger.log_interaction.call_args
    assert call_args.kwargs['user_id'] == "api_user_from_token"


@pytest.mark.asyncio
async def test_process_chat_with_non_saml_user(chat_service):
    chat_request = ChatRequest(message="Hello", user_id="api_user")
    http_request = MockRequest()

    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "Response"})

    await chat_service.process_chat(http_request, chat_request, http_request.headers)

    call_args = chat_service.db_logger.log_interaction.call_args
    assert call_args.kwargs['user_id'] == "api_user"


@pytest.mark.asyncio
async def test_process_chat_with_files(chat_service):
    chat_request = ChatRequest(message="Analyze this", user_id="user1")
    http_request = MockRequest()
    files = ["/tmp/file1.txt"]

    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "File analyzed"})

    await chat_service.process_chat(http_request, chat_request, http_request.headers, files=files)

    call_args = chat_service.agent.ainvoke.call_args
    assert call_args.kwargs['config']['uploaded_files'] == files


@pytest.mark.asyncio
async def test_process_chat_with_files_in_message(chat_service):
    chat_request = ChatRequest(message={"content": "Analyze", "uploaded_files": ["/tmp/file2.txt"]}, user_id="user1")
    http_request = MockRequest()

    chat_service.agent.ainvoke = AsyncMock(return_value={"content": "File analyzed"})

    await chat_service.process_chat(http_request, chat_request, http_request.headers)

    call_args = chat_service.agent.ainvoke.call_args
    assert call_args.kwargs['config']['uploaded_files'] == ["/tmp/file2.txt"]


@pytest.mark.asyncio
async def test_process_chat_exception(chat_service):
    chat_request = ChatRequest(message="Hello", user_id="user1")
    http_request = MockRequest()
    chat_service.agent.ainvoke = AsyncMock(side_effect=Exception("Agent error"))

    with pytest.raises(HTTPException) as excinfo:
        await chat_service.process_chat(http_request, chat_request, http_request.headers)
    assert excinfo.value.status_code == 500


@pytest.mark.asyncio
async def test_process_stream_chat(chat_service):
    stream_request = StreamChatRequest(message="Stream me", user_id="user1")
    http_request = MockRequest()

    async def mock_stream(*args, **kwargs):
        yield {"content": "Chunk 1"}
        yield {"content": "Chunk 2"}

    chat_service.agent.astream = mock_stream

    response = await chat_service.process_stream_chat(http_request, stream_request, http_request.headers)

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
    stream_request = StreamChatRequest(message="Stream me", user_id="user1")
    http_request = MockRequest()

    class NonSerializable:
        pass

    async def mock_stream(*args, **kwargs):
        yield {"content": NonSerializable()}

    chat_service.agent.astream = mock_stream

    with patch('oai_agent_server.services.chat_service.make_serializable', side_effect=Exception("Serialize error")):
        response = await chat_service.process_stream_chat(http_request, stream_request, http_request.headers)
        chunks = []
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode('utf-8')
            chunks.append(chunk)

        assert any("serialization_warning" in chunk for chunk in chunks)


@pytest.mark.asyncio
async def test_process_stream_chat_exception(chat_service):
    stream_request = StreamChatRequest(message="Stream me", user_id="user1")
    http_request = MockRequest()

    chat_service.agent.astream = MagicMock(side_effect=Exception("Stream init error"))

    chat_service._handle_reinitialization = AsyncMock(side_effect=Exception("Init error"))

    with pytest.raises(StreamingException):
        await chat_service.process_stream_chat(http_request, stream_request, http_request.headers)


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
