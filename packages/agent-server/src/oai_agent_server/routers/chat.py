import os
import shutil
import tempfile
from typing import Any, List, Optional, Union, Tuple

from fastapi import APIRouter, Request, Depends, UploadFile, File, Form, BackgroundTasks

from oai_agent_server.models.requests import ChatRequest, StreamChatRequest
from oai_agent_server.models.responses import ChatResponse
from oai_agent_server.security.dependencies import verify_api_key


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

        def _append_files_to_message(message: Union[str, dict], file_paths: List[str], session_id: Optional[str] = None) -> Union[str, dict]:
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
        async def chat(request: ChatRequest, req: Request):
            """Process a synchronous chat request."""
            original_message = request.message
            request.message = _append_files_to_message(request.message, [], request.session_id)
            return await chat_service.process_chat(request=request,
                                                   headers=req.headers,
                                                   original_message=original_message)

        @router.post("/with-files", response_model=ChatResponse)
        async def chat_with_files(
                req: Request,
                background_tasks: BackgroundTasks,
                message: str = Form(...),
                session_id: Optional[str] = Form(None),
                user_id: Optional[str] = Form("user"),
                files: List[UploadFile] = File(None)
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
            return await chat_service.process_chat(request=chat_request,
                                                   headers=dict(req.headers),
                                                   files=file_paths,
                                                   original_message=original_message)

        @router.post("/stream")
        async def chat_stream(request: StreamChatRequest, req: Request):
            """Process a streaming chat request."""
            original_message = request.message
            request.message = _append_files_to_message(request.message, [], request.session_id)
            return await chat_service.process_stream_chat(request=request,
                                                          headers=req.headers,
                                                          original_message=original_message)

        @router.post("/stream/with-files")
        async def chat_stream_with_files(
                req: Request,
                background_tasks: BackgroundTasks,
                message: str = Form(...),
                session_id: Optional[str] = Form(None),
                user_id: Optional[str] = Form("user"),
                verbose: bool = Form(False),
                files: List[UploadFile] = File(None)
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
            return await chat_service.process_stream_chat(request=stream_request,
                                                          headers=dict(req.headers),
                                                          files=file_paths,
                                                          original_message=original_message)

    return router
