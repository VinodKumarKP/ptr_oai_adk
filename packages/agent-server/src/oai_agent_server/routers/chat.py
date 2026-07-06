import os
import shutil
import tempfile
from typing import Any, List, Optional, Union, Tuple

from fastapi import APIRouter, Request, Depends, UploadFile, File, Form, BackgroundTasks

from oai_agent_server.models.requests import ChatRequest, StreamChatRequest
from oai_agent_server.security.dependencies import verify_api_key
from oai_agent_server.utils.file_manager import FileUploadManager

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False
    Limiter = None
    get_remote_address = None

# Module-level limiter so the decorators can reference it at import time.
# The app-level limiter (attached in main.py) handles the actual enforcement.
_chat_rate_limit = os.environ.get("RATE_LIMIT_CHAT", "60/minute")
if _SLOWAPI_AVAILABLE:
    _limiter = Limiter(key_func=get_remote_address)
else:
    _limiter = None


def _maybe_limit(rate: str):
    """Return a no-op decorator when slowapi is unavailable."""
    if _limiter is None:
        def _noop(func):
            return func
        return _noop
    return _limiter.limit(rate)


def create_chat_router(chat_service: Any, allowed_modes: Optional[List[str]] = None) -> APIRouter:
    """Create the chat router with configured endpoints."""
    router = APIRouter(prefix="/chat", tags=["chat"], dependencies=[Depends(verify_api_key)])

    if allowed_modes is None:
        allowed_modes = ["chat"]

    if "chat" in allowed_modes:
        @router.post("")
        @_maybe_limit(_chat_rate_limit)
        async def chat(request: Request, chat_request: ChatRequest, background_tasks: BackgroundTasks):
            """Process a synchronous chat request.

            Supports both old and new request formats:
            - Old: { "message": "...", "session_id": "...", "user_id": "..." }
            - New: { "assistant_id": "xyz", "input": { "message": "...", ... } }

            If assistant_id is not provided, defaults to the server's configured agent.

            Returns a free-form dict ``{"content": ..., "session_id": str, "interaction_id": str}``.
            ``content`` is provider-dependent (string or dict with model/token_usage), so the
            response shape isn't constrained by a Pydantic model.
            """
            # Extract agent name: use assistant_id if provided, otherwise use configured agent
            agent_name = chat_request.assistant_id or chat_service.agent_name

            original_message = chat_request.message
            chat_request.message = FileUploadManager.append_files_to_message(
                chat_request.message, [], chat_request.session_id
            )
            return await chat_service.process_chat(http_request=request,
                                                   chat_request=chat_request,
                                                   background_tasks=background_tasks,
                                                   headers=request.headers,
                                                   original_message=original_message,
                                                   agent_name=agent_name)

        _with_files_openapi = {
            "requestBody": {
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                                "session_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string", "format": "binary"},
                                },
                            },
                            "required": ["message"],
                        }
                    }
                }
            }
        }

        @router.post("/with-files", openapi_extra=_with_files_openapi)
        async def chat_with_files(
                http_request: Request,
                background_tasks: BackgroundTasks,
                message: str = Form(...),
                session_id: Optional[str] = Form(None),
                user_id: Optional[str] = Form("user"),
                assistant_id: Optional[str] = Form(None),
                files: List[UploadFile] = File(default=[])
        ):
            """Process a synchronous chat request with file uploads.

            Optionally accepts assistant_id to route to a specific agent.
            If assistant_id is not provided, defaults to the server's configured agent.
            """
            file_paths, temp_dir = await FileUploadManager.save_files(files)

            if temp_dir:
                background_tasks.add_task(FileUploadManager.cleanup, temp_dir)

            chat_request = ChatRequest(
                message=message,
                session_id=session_id,
                user_id=user_id,
                assistant_id=assistant_id
            )

            # Extract agent name: use assistant_id if provided, otherwise use configured agent
            agent_name = assistant_id or chat_service.agent_name

            original_message = chat_request.message
            chat_request.message = FileUploadManager.append_files_to_message(
                chat_request.message, file_paths, session_id
            )
            return await chat_service.process_chat(http_request=http_request,
                                                   chat_request=chat_request,
                                                   background_tasks=background_tasks,
                                                   headers=dict(http_request.headers),
                                                   files=file_paths,
                                                   original_message=original_message,
                                                   agent_name=agent_name)

        @router.post("/stream")
        @_maybe_limit(_chat_rate_limit)
        async def chat_stream(request: Request, stream_request: StreamChatRequest,
                              background_tasks: BackgroundTasks):
            """Process a streaming chat request.

            Supports both old and new request formats:
            - Old: { "message": "...", "session_id": "...", "user_id": "..." }
            - New: { "assistant_id": "xyz", "input": { "message": "...", ... } }

            If assistant_id is not provided, defaults to the server's configured agent.
            """
            # Extract agent name: use assistant_id if provided, otherwise use configured agent
            agent_name = stream_request.assistant_id or chat_service.agent_name

            original_message = stream_request.message
            stream_request.message = FileUploadManager.append_files_to_message(
                stream_request.message, [], stream_request.session_id
            )
            return await chat_service.process_stream_chat(http_request=request,
                                                          stream_request=stream_request,
                                                          background_tasks=background_tasks,
                                                          headers=request.headers,
                                                          original_message=original_message,
                                                          agent_name=agent_name)

        _stream_with_files_openapi = {
            "requestBody": {
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "message": {"type": "string"},
                                "session_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "verbose": {"type": "boolean"},
                                "files": {
                                    "type": "array",
                                    "items": {"type": "string", "format": "binary"},
                                },
                            },
                            "required": ["message"],
                        }
                    }
                }
            }
        }

        @router.post("/stream/with-files", openapi_extra=_stream_with_files_openapi)
        async def chat_stream_with_files(
                http_request: Request,
                background_tasks: BackgroundTasks,
                message: str = Form(...),
                session_id: Optional[str] = Form(None),
                user_id: Optional[str] = Form("user"),
                verbose: bool = Form(False),
                assistant_id: Optional[str] = Form(None),
                files: List[UploadFile] = File(default=[])
        ):
            """Process a streaming chat request with file uploads.

            Optionally accepts assistant_id to route to a specific agent.
            If assistant_id is not provided, defaults to the server's configured agent.
            """
            file_paths, temp_dir = await FileUploadManager.save_files(files)

            if temp_dir:
                background_tasks.add_task(FileUploadManager.cleanup, temp_dir)

            stream_request = StreamChatRequest(
                message=message,
                session_id=session_id,
                user_id=user_id,
                verbose=verbose,
                assistant_id=assistant_id
            )

            # Extract agent name: use assistant_id if provided, otherwise use configured agent
            agent_name = assistant_id or chat_service.agent_name

            original_message = stream_request.message
            stream_request.message = FileUploadManager.append_files_to_message(
                stream_request.message, file_paths, session_id
            )
            return await chat_service.process_stream_chat(http_request=http_request,
                                                          stream_request=stream_request,
                                                          background_tasks=background_tasks,
                                                          headers=dict(http_request.headers),
                                                          files=file_paths,
                                                          original_message=original_message,
                                                          agent_name=agent_name)

    return router
