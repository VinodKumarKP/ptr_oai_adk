import os
import shutil
import tempfile
from typing import Any, List, Optional, Union, Tuple

from fastapi import APIRouter, Request, Depends, UploadFile, File, Form, BackgroundTasks

from oai_agent_server.models.requests import ChatRequest, StreamChatRequest
from oai_agent_server.models.responses import ChatResponse
from oai_agent_server.security.dependencies import verify_api_key

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
        def _save_files(files: List[UploadFile]) -> Tuple[List[str], Optional[str]]:
            """Save uploaded files to a temporary directory."""
            if not files:
                return [], None
            temp_dir = tempfile.mkdtemp()
            file_paths = []
            for file in files:
                file_path = os.path.join(temp_dir, file.filename)
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                file_paths.append(file_path)
            return file_paths, temp_dir

        def _cleanup_files(temp_dir: Optional[str]):
            """Cleanup temporary directory and files."""
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        def _append_files_to_message(message: Union[str, dict], file_paths: List[str],
                                     session_id: Optional[str] = None) -> Union[str, dict]:
            """Append file paths and session ID to the message content."""
            extra_content = ""
            if file_paths:
                extra_content += "\nUploaded Files:\n" + "\n".join(file_paths)

            if session_id:
                extra_content += f"\nSession ID: {session_id}"

            if not extra_content:
                return message

            if isinstance(message, str):
                return message + extra_content
            elif isinstance(message, dict):
                if "content" in message and isinstance(message["content"], str):
                    message["content"] += extra_content
                elif "text" in message and isinstance(message["text"], str):
                    message["text"] += extra_content
                # We don't need to add uploaded_files to message dict anymore as we pass it separately
            return message

        @router.post("", response_model=ChatResponse)
        @_maybe_limit(_chat_rate_limit)
        async def chat(request: Request, chat_request: ChatRequest, background_tasks: BackgroundTasks):
            """Process a synchronous chat request."""
            original_message = chat_request.message
            chat_request.message = _append_files_to_message(chat_request.message, [], chat_request.session_id)
            return await chat_service.process_chat(http_request=request,
                                                   chat_request=chat_request,
                                                   background_tasks=background_tasks,
                                                   headers=request.headers,
                                                   original_message=original_message)

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

        @router.post("/with-files", response_model=ChatResponse, openapi_extra=_with_files_openapi)
        async def chat_with_files(
                http_request: Request,
                background_tasks: BackgroundTasks,
                message: str = Form(...),
                session_id: Optional[str] = Form(None),
                user_id: Optional[str] = Form("user"),
                files: List[UploadFile] = File(default=[])
        ):
            """Process a synchronous chat request with file uploads."""
            file_paths, temp_dir = _save_files(files)

            if temp_dir:
                background_tasks.add_task(_cleanup_files, temp_dir)

            chat_request = ChatRequest(
                message=message,
                session_id=session_id,
                user_id=user_id
            )

            original_message = chat_request.message
            chat_request.message = _append_files_to_message(chat_request.message, file_paths, session_id)
            return await chat_service.process_chat(http_request=http_request,
                                                   chat_request=chat_request,
                                                   background_tasks=background_tasks,
                                                   headers=dict(http_request.headers),
                                                   files=file_paths,
                                                   original_message=original_message)

        @router.post("/stream")
        @_maybe_limit(_chat_rate_limit)
        async def chat_stream(request: Request, stream_request: StreamChatRequest,
                              background_tasks: BackgroundTasks):
            """Process a streaming chat request."""
            original_message = stream_request.message
            stream_request.message = _append_files_to_message(stream_request.message, [], stream_request.session_id)
            return await chat_service.process_stream_chat(http_request=request,
                                                          stream_request=stream_request,
                                                          background_tasks=background_tasks,
                                                          headers=request.headers,
                                                          original_message=original_message)

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
                files: List[UploadFile] = File(default=[])
        ):
            """Process a streaming chat request with file uploads."""
            file_paths, temp_dir = _save_files(files)

            if temp_dir:
                background_tasks.add_task(_cleanup_files, temp_dir)

            stream_request = StreamChatRequest(
                message=message,
                session_id=session_id,
                user_id=user_id,
                verbose=verbose
            )

            original_message = stream_request.message
            stream_request.message = _append_files_to_message(stream_request.message, file_paths, session_id)
            return await chat_service.process_stream_chat(http_request=http_request,
                                                          stream_request=stream_request,
                                                          background_tasks=background_tasks,
                                                          headers=dict(http_request.headers),
                                                          files=file_paths,
                                                          original_message=original_message)

    return router
