import inspect
import json
import time
import uuid
from datetime import datetime
from typing import Optional, AsyncGenerator, List

from fastapi import HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from oai_agent_server.exceptions import StreamingException
from oai_agent_server.models.requests import ChatRequest, StreamChatRequest
from oai_agent_server.services.llm_judge_service import LLMJudgeService
from oai_agent_server.utils.response_extractor import ResponseContentExtractor, extract_output_text, extract_chunk_text
from oai_agent_server.utils.serialization import make_serializable


class ChatService:
    """Service for handling chat interactions."""

    def __init__(self, agent, db_logger, logger, llm_judge_service: LLMJudgeService, allowed_modes: List[str]):
        self.agent = agent
        self.db_logger = db_logger
        self.logger = logger
        self.agent_name = agent.agent_name
        self.llm_judge_service = llm_judge_service
        self.allowed_modes = allowed_modes

    async def process_chat(self, http_request: Request,
                           chat_request: ChatRequest,
                           background_tasks: BackgroundTasks,
                           headers,
                           files: Optional[List[str]] = None,
                           original_message: Optional[str] = None):
        """Process a synchronous chat request."""
        start_time = time.time()
        interaction_id = str(uuid.uuid4())
        session_id = chat_request.session_id or str(uuid.uuid4())
        user_id = (getattr(http_request.state, 'user_email', None) or
                   getattr(http_request.state, 'user_id', None) or
                   chat_request.user_id or
                   "user")
        status = "success"

        config = {
            "session_id": session_id,
            "user_id": user_id,
            "original_message": original_message
        }
        if files:
            config["uploaded_files"] = files
        elif isinstance(chat_request.message, dict) and "uploaded_files" in chat_request.message:
            config["uploaded_files"] = chat_request.message["uploaded_files"]

        try:
            await self._handle_reinitialization(user_id)
            response = await self.agent.ainvoke(user_message=chat_request.message, config=config)
            content = ResponseContentExtractor(self.agent).extract_content(response)
            output_response = extract_output_text(content)
            response_time_ms = (time.time() - start_time) * 1000

            await self.db_logger.log_interaction(
                interaction_id=interaction_id,
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

            if "monitoring" in self.allowed_modes:
                background_tasks.add_task(
                    self.llm_judge_service.judge_interaction,
                    interaction_id=interaction_id,
                    user_message=chat_request.message,
                    agent_response=output_response,
                    session_id=session_id,
                    user_id=user_id
                )

            return JSONResponse(
                content={"content": content, "session_id": session_id, "interaction_id": interaction_id})
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

    async def process_stream_chat(self, http_request: Request,
                                  stream_request: StreamChatRequest,
                                  background_tasks: BackgroundTasks,
                                  headers,
                                  files: Optional[List[str]] = None,
                                  original_message: Optional[str] = None):
        """Process a streaming chat request."""
        start_time = time.time()
        interaction_id = str(uuid.uuid4())
        session_id = stream_request.session_id or str(uuid.uuid4())
        user_id = (getattr(http_request.state, 'user_email', None) or
                   getattr(http_request.state, 'user_id', None) or
                   stream_request.user_id or
                   "user")

        config = {
            "session_id": session_id,
            "user_id": user_id,
            "verbose": stream_request.verbose,
            "original_message": original_message
        }
        if files:
            config["uploaded_files"] = files
        elif isinstance(stream_request.message, dict) and "uploaded_files" in stream_request.message:
            config["uploaded_files"] = stream_request.message["uploaded_files"]

        try:
            await self._handle_reinitialization(user_id)
            response_extractor = ResponseContentExtractor(self.agent)

            async def generate_response() -> AsyncGenerator[str, None]:
                stream = self.agent.astream(user_message=stream_request.message, config=config)
                if inspect.iscoroutine(stream):
                    stream = await stream

                last_response = {}
                activity_chunks = []
                chunk_sequence = 0
                async for chunk in stream:
                    content = response_extractor.extract_content(chunk)
                    last_response = content
                    chunk_text = extract_chunk_text(content)
                    json_data, serialization_warning = self._serialize_chunk(content, session_id, interaction_id)
                    yield f"data: {json_data}\n\n"
                    activity_chunks.append({
                        'chunk_sequence': chunk_sequence, 'chunk_content': content,
                        'chunk_text': chunk_text, 'serialization_warning': serialization_warning,
                        'timestamp': datetime.utcnow()
                    })
                    chunk_sequence += 1

                output_response = extract_output_text(last_response)
                response_time_ms = (time.time() - start_time) * 1000

                if "monitoring" in self.allowed_modes:
                    background_tasks.add_task(
                        self.llm_judge_service.judge_interaction,
                        interaction_id=interaction_id,
                        user_message=stream_request.message,
                        agent_response=output_response,
                        session_id=session_id,
                        user_id=user_id
                    )

                if activity_chunks:
                    await self.db_logger.log_stream_chunks_batch(
                        interaction_id=interaction_id,
                        agent_name=self.agent_name, session_id=session_id, user_id=user_id,
                        endpoint="/chat/stream", chunks=activity_chunks, request_headers=headers
                    )

                await self.db_logger.log_interaction(
                    interaction_id=interaction_id,
                    agent_name=self.agent_name, session_id=session_id, user_id=user_id,
                    endpoint="/chat/stream", input_message=stream_request.message,
                    output_response=output_response, request_headers=headers,
                    model_info=last_response.get('model') if isinstance(last_response, dict) else None,
                    token_usage=last_response.get('token_usage') if isinstance(last_response, dict) else None,
                    response_time_ms=response_time_ms, status="success"
                )
                yield "data: [DONE]\n\n"

            return StreamingResponse(generate_response(), media_type="text/event-stream")
        except Exception as e:
            raise StreamingException(reason=str(e))

    def _serialize_chunk(self, content, session_id, interaction_id):
        base_data = {'content': content, 'session_id': session_id, 'interaction_id': interaction_id}
        try:
            return json.dumps(base_data), None
        except (TypeError, ValueError):
            try:
                base_data['content'] = make_serializable(content)
                return json.dumps(base_data), None
            except Exception as e:
                warning = f"Content not JSON serializable: {e}, converting to string"
                self.logger.warning(warning)
                base_data['content'] = str(content)
                base_data['serialization_warning'] = warning
                return json.dumps(base_data), warning

    async def _handle_reinitialization(self, user_id: Optional[str]):
        import os
        if os.environ.get('AGENT_REINITIALIZE', 'false').lower() == 'true':
            self.logger.info(f"Reinitializing agent {self.agent_name}")
            if hasattr(self.agent, "base_agent_list"): self.agent.base_agent_list = []
            if hasattr(self.agent, "agent_list"): self.agent.agent_list = []
            await self.agent.initialize()
            self.agent.user_id = user_id or "default_user"
