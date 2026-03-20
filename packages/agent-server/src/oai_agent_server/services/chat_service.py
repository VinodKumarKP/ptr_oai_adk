import inspect
import json
import time
import uuid
from typing import Optional, AsyncGenerator, List

from fastapi import HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse

from oai_agent_server.exceptions import StreamingException
from oai_agent_server.models.requests import ChatRequest, StreamChatRequest
from oai_agent_server.utils.response_extractor import ResponseContentExtractor, extract_output_text, extract_chunk_text
from oai_agent_server.utils.serialization import make_serializable


class ChatService:
    """Service for handling chat interactions."""

    def __init__(self, agent, db_logger, logger):
        self.agent = agent
        self.db_logger = db_logger
        self.logger = logger
        self.agent_name = agent.agent_name

    async def process_chat(self, http_request: Request,
                           chat_request: ChatRequest,
                           headers,
                           files: Optional[List[str]] = None,
                           original_message: Optional[str] = None):
        """Process a synchronous chat request.

        Args:
            http_request: The FastAPI request object.
            chat_request: The chat request object.
            headers: Request headers.
            files: Optional list of uploaded file paths.
            original_message: Optional original message.
        Returns:
            JSONResponse containing the chat response.
        """
        start_time = time.time()
        session_id = chat_request.session_id or str(uuid.uuid4())
        user_id = getattr(http_request.state, 'user_email', None) or chat_request.user_id or "user"
        status = "success"

        # Prepare config
        config = {
            "session_id": session_id,
            "user_id": user_id,
            "original_message": original_message
        }

        # Add files to config if provided
        if files:
            config["uploaded_files"] = files
        # Fallback: Extract uploaded files if present in message dict (legacy/alternative way)
        elif isinstance(chat_request.message, dict) and "uploaded_files" in chat_request.message:
            config["uploaded_files"] = chat_request.message["uploaded_files"]

        try:
            await self._handle_reinitialization(user_id)

            response = await self.agent.ainvoke(
                user_message=chat_request.message,
                config=config
            )

            response_extractor = ResponseContentExtractor(self.agent)
            # Extract the response content from the agent response
            content = response_extractor.extract_content(response)

            response_time_ms = (time.time() - start_time) * 1000

            output_response = extract_output_text(content)

            # Log to database
            await self.db_logger.log_interaction(
                agent_name=self.agent_name,
                session_id=session_id,
                user_id=user_id,
                endpoint="/chat",
                input_message=chat_request.message,
                output_response=output_response,
                request_headers=headers,
                model_info=content.get('model') if isinstance(content, dict) else None,
                token_usage=content.get('token_usage') if isinstance(content, dict) else None,
                response_time_ms=response_time_ms,
                status=status
            )

            return JSONResponse(
                content={
                    "content": content,
                    "session_id": session_id,
                })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    async def process_stream_chat(self, http_request: Request,
                                  stream_request: StreamChatRequest,
                                  headers,
                                  files: Optional[List[str]] = None,
                                  original_message: Optional[str] = None):
        """Process a streaming chat request.

        Args:
            http_request: The FastAPI request object.
            stream_request: The streaming chat request object.
            headers: Request headers.
            files: Optional list of uploaded file paths.
            original_message: Optional original message.

        Returns:
            StreamingResponse yielding chat chunks.
        """
        start_time = time.time()
        session_id = stream_request.session_id or str(uuid.uuid4())
        user_id = getattr(http_request.state, 'user_email', None) or stream_request.user_id or "user"

        # Prepare config
        config = {
            "session_id": session_id,
            "user_id": user_id,
            "verbose": stream_request.verbose,
            "original_message": original_message
        }

        # Add files to config if provided
        if files:
            config["uploaded_files"] = files
        # Fallback: Extract uploaded files if present in message dict
        elif isinstance(stream_request.message, dict) and "uploaded_files" in stream_request.message:
            config["uploaded_files"] = stream_request.message["uploaded_files"]

        try:
            # Handle reinitialization BEFORE generating response
            await self._handle_reinitialization(user_id)

            response_extractor = ResponseContentExtractor(self.agent)

            async def generate_response() -> AsyncGenerator[str, None]:
                # Call astream to get the result
                stream_result = self.agent.astream(
                    user_message=stream_request.message,
                    config=config
                )

                # Check if the result is an async generator or needs to be awaited
                if inspect.isasyncgen(stream_result):
                    # It's already an async generator, use it directly
                    stream = stream_result
                elif inspect.iscoroutine(stream_result):
                    # It's a coroutine that needs to be awaited
                    stream = await stream_result
                else:
                    # Fallback: assume it's already the stream
                    stream = stream_result

                last_response = {}
                chunk_sequence = 0

                # Collect all chunks for batch insert at the end
                activity_chunks = []

                async for chunk in stream:
                    # Extract content from the chunk
                    content = response_extractor.extract_content(chunk)

                    # Track last response for logging
                    last_response = content

                    # Initialize variables for activity logging
                    chunk_text = None
                    serialization_warning = None

                    # Format the chunk as JSON for streaming
                    if content:
                        try:
                            # Try to serialize directly
                            json_data = json.dumps({
                                'content': content,
                                'session_id': session_id
                            })

                            # Extract text from content if possible
                            chunk_text = extract_chunk_text(content)

                            yield f"data: {json_data}\n\n"

                        except (TypeError, ValueError) as e:
                            # Handle non-serializable content
                            try:
                                # Try to convert content to a serializable format
                                serializable_content = make_serializable(content)
                                json_data = json.dumps({
                                    'content': serializable_content,
                                    'session_id': session_id
                                })
                                chunk_text = str(serializable_content)
                                yield f"data: {json_data}\n\n"

                            except Exception as serialize_error:
                                # Last resort: send as string
                                self.logger.warning(f"Content not JSON serializable: {e}, converting to string")
                                serialization_warning = f"Content was converted to string: {str(e)}"
                                chunk_text = str(content)
                                json_data = json.dumps({
                                    'content': str(content),
                                    'session_id': session_id,
                                    'serialization_warning': serialization_warning
                                })
                                yield f"data: {json_data}\n\n"

                        # Collect chunk data for batch insert
                        from datetime import datetime
                        activity_chunks.append({
                            'chunk_sequence': chunk_sequence,
                            'chunk_content': content,
                            'chunk_text': chunk_text,
                            'serialization_warning': serialization_warning,
                            'timestamp': datetime.utcnow()
                        })

                        chunk_sequence += 1

                output_response = extract_output_text(last_response)

                # Batch insert all collected chunks to agent_activity_log
                if activity_chunks:
                    await self.db_logger.log_stream_chunks_batch(
                        agent_name=self.agent_name,
                        session_id=session_id,
                        user_id=user_id,
                        endpoint="/chat/stream",
                        chunks=activity_chunks,
                        request_headers=headers
                    )

                # Log to database after streaming completes (summary log in chat_logs)
                response_time_ms = (time.time() - start_time) * 1000
                await self.db_logger.log_interaction(
                    agent_name=self.agent_name,
                    session_id=session_id,
                    user_id=user_id,
                    endpoint="/chat/stream",
                    input_message=stream_request.message,
                    output_response=output_response,
                    request_headers=headers,
                    model_info=last_response.get('model') if isinstance(last_response, dict) else None,
                    token_usage=last_response.get('token_usage') if isinstance(last_response,
                                                                               dict) else None,
                    response_time_ms=response_time_ms,
                    status="success"
                )

                # Send end signal
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                generate_response(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "Content-Type": "text/event-stream"
                }
            )
        except Exception as e:
            raise StreamingException(reason=str(e))

    async def _handle_reinitialization(self, user_id: Optional[str]):
        """Handle agent reinitialization if requested via environment variable"""
        import os
        agent_reinit = os.environ.get('AGENT_REINITIALIZE', 'false')
        if isinstance(agent_reinit, str) and agent_reinit.lower() == 'true':
            self.logger.info(f"Reinitializing agent {self.agent_name}")
            if hasattr(self.agent, "base_agent_list"):
                self.agent.base_agent_list = []
            if hasattr(self.agent, "agent_list"):
                self.agent.agent_list = []
            await self.agent.initialize()

            if user_id is not None:
                self.agent.user_id = user_id
            else:
                self.agent.user_id = "default_user"
