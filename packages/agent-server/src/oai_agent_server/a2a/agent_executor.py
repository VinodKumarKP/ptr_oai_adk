"""
BaseAgentExecutor — bridges the a2a-sdk AgentExecutor interface
to your existing BaseAgent (ainvoke / astream).

Install requirement:
    pip install "a2a-sdk[http-server]"

This is the ONLY file you write. The SDK owns everything else:
  - Task store       → InMemoryTaskStore (or DatabaseTaskStore)
  - Protocol routing → DefaultRequestHandler
  - HTTP server      → A2AStarletteApplication mounted on your FastAPI app
  - SSE streaming    → EventQueue / EventConsumer
  - Push notifs      → PushNotificationSender
"""

import asyncio
from datetime import datetime
import inspect
import logging
import time
from typing import Optional, List

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from typing_extensions import override

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_server.services.llm_judge_service import LLMJudgeService
from oai_agent_server.utils.database_logger import DatabaseLogger

logger = logging.getLogger(__name__)


class BaseAgentExecutor(AgentExecutor):
    """
    Adapter between a2a-sdk's AgentExecutor interface and your BaseAgent.

    Supports both:
      • Synchronous invoke  → ainvoke()  (tasks/send)
      • Streaming invoke    → astream()  (tasks/stream / sendSubscribe)

    Both code paths funnel through the same execute() method — the SDK
    decides how to deliver the events to the caller (batch vs SSE).
    """

    def __init__(self, agent: BaseAgent, db_logger: DatabaseLogger, llm_judge_service: LLMJudgeService,
                 allowed_modes: List[str], use_streaming: bool = True):
        """
        Args:
            agent:         Your BaseAgent instance.
            use_streaming: If True and agent has astream(), use it so the
                           SDK can stream partial results to the caller.
                           Set False to always use ainvoke().
        """
        self.agent = agent
        self.db_logger = db_logger
        self.llm_judge_service = llm_judge_service
        self.allowed_modes = allowed_modes
        self.use_streaming = use_streaming and hasattr(agent, "astream")
        self._cancelled_tasks: set[str] = set()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response) -> str:
        """
        Pull a plain string out of whatever BaseAgent returns.
        Handles: str, dict with common output keys, anything else → str().
        """
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            for key in ("output", "content", "text", "result", "answer", "response"):
                val = response.get(key)
                if isinstance(val, str) and val:
                    return val
                if isinstance(val, dict):
                    inner = val.get("text") or val.get("content") or ""
                    if inner:
                        return str(inner)
        return str(response)

    def _build_config(self, context: RequestContext) -> dict:
        """Map A2A RequestContext fields to your agent's config dict."""
        return {
            "session_id": context.context_id or "",
            "user_id": "a2a",
        }

    def _is_cancelled(self, task_id: Optional[str]) -> bool:
        return task_id is not None and task_id in self._cancelled_tasks

    # ------------------------------------------------------------------
    # execute() — called for BOTH tasks/send and tasks/sendSubscribe
    # ------------------------------------------------------------------

    @override
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Entry point called by DefaultRequestHandler for every incoming task.

        Flow:
          1. Emit the Task object so the SDK registers it in the TaskStore.
          2. Emit TaskStatusUpdateEvent(working).
          3. Call ainvoke() or astream() on your BaseAgent.
          4. Emit TaskArtifactUpdateEvent(s) with the response content.
          5. Emit TaskStatusUpdateEvent(completed | failed).
        """
        query = context.get_user_input()
        config = self._build_config(context)

        # 1. Register the task (required for stateful tracking)
        if not context.current_task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        task_id = context.task_id
        context_id = context.context_id

        # 2. Signal working
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.working,
                    message=new_agent_text_message("Processing your request…"),
                ),
                final=False,
            )
        )

        try:
            if self.use_streaming:
                await self._execute_streaming(
                    query, config, task_id, context_id, event_queue
                )
            else:
                await self._execute_batch(
                    query, config, task_id, context_id, event_queue
                )

        except Exception as e:
            logger.exception(f"Agent execution failed for task '{task_id}': {e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(f"Error: {str(e)}"),
                    ),
                    final=True,
                )
            )

    # ------------------------------------------------------------------
    # Batch path  (ainvoke)
    # ------------------------------------------------------------------

    async def _execute_batch(
        self, query: str, config: dict,
        task_id: str, context_id: str,
        event_queue: EventQueue,
    ) -> None:
        """Call ainvoke(), emit a single artifact, then complete."""
        start_time = time.time()
        response = await self.agent.ainvoke(user_message=query, config=config)
        output_text = self._extract_text(response)
        response_time_ms = (time.time() - start_time) * 1000

        await self.db_logger.log_interaction(
            interaction_id=task_id,
            agent_name=self.agent.agent_name,
            session_id=context_id,
            user_id="a2a",
            endpoint="/a2a/tasks/send",
            input_message=query,
            output_response=output_text,
            request_headers={},
            model_info=response.get('model') if isinstance(response, dict) else None,
            token_usage=response.get('token_usage') if isinstance(response, dict) else None,
            response_time_ms=response_time_ms,
            status="success"
        )

        if "monitoring" in self.allowed_modes:
            asyncio.create_task(
                self.llm_judge_service.judge_interaction(
                    interaction_id=task_id,
                    user_message=query,
                    agent_response=output_text,
                    session_id=context_id,
                    user_id="a2a"
                )
            )

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=new_text_artifact(
                    name="response",
                    text=output_text,
                ),
                append=False,
                last_chunk=True,
            )
        )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=new_agent_text_message(output_text),
                ),
                final=True,
            )
        )

    # ------------------------------------------------------------------
    # Streaming path  (astream)
    # ------------------------------------------------------------------

    async def _execute_streaming(
        self, query: str, config: dict,
        task_id: str, context_id: str,
        event_queue: EventQueue,
    ) -> None:
        """
        Call astream(), emit one TaskArtifactUpdateEvent per chunk,
        then a final completed status event.

        If the agent doesn't actually stream (returns a coroutine instead
        of an async generator), falls back to batch gracefully.
        """
        start_time = time.time()
        stream = self.agent.astream(user_message=query, config=config)

        # astream() might be a coroutine that returns an async generator
        if inspect.iscoroutine(stream):
            stream = await stream

        # Fallback: if it's not an async generator, treat as batch
        if not hasattr(stream, "__aiter__"):
            response = stream
            output_text = self._extract_text(response)
            response_time_ms = (time.time() - start_time) * 1000
            await self.db_logger.log_interaction(
                interaction_id=task_id,
                agent_name=self.agent.agent_name,
                session_id=context_id,
                user_id="a2a",
                endpoint="/a2a/tasks/stream",
                input_message=query,
                output_response=output_text,
                request_headers={},
                model_info=response.get('model') if isinstance(response, dict) else None,
                token_usage=response.get('token_usage') if isinstance(response, dict) else None,
                response_time_ms=response_time_ms,
                status="success"
            )
            if "monitoring" in self.allowed_modes:
                asyncio.create_task(
                    self.llm_judge_service.judge_interaction(
                        interaction_id=task_id,
                        user_message=query,
                        agent_response=output_text,
                        session_id=context_id,
                        user_id="a2a"
                    )
                )
            await self._emit_single_artifact(output_text, task_id, context_id, event_queue)
            return

        chunk_index = 0
        last_text = ""
        last_response = {}
        activity_chunks = []

        async for chunk in stream:
            # Check for cancellation between chunks
            if self._is_cancelled(task_id):
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        status=TaskStatus(state=TaskState.canceled),
                        final=True,
                    )
                )
                return

            chunk_text = self._extract_text(chunk)
            if not chunk_text:
                continue

            last_text = chunk_text
            last_response = chunk
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    artifact=new_text_artifact(
                        name="response",
                        text=chunk_text,
                    ),
                    append=chunk_index > 0,   # first chunk replaces, rest append
                    last_chunk=False,
                )
            )
            activity_chunks.append({
                'chunk_sequence': chunk_index, 'chunk_content': chunk,
                'chunk_text': chunk_text, 'serialization_warning': '',
                'timestamp': datetime.utcnow()
            })
            chunk_index += 1

        response_time_ms = (time.time() - start_time) * 1000

        if activity_chunks:
            await self.db_logger.log_stream_chunks_batch(
                interaction_id=task_id,
                agent_name=self.agent.agent_name, session_id=context_id, user_id='a2a',
                endpoint="/a2a/tasks/stream", chunks=activity_chunks, request_headers={}
            )

        await self.db_logger.log_interaction(
            interaction_id=task_id,
            agent_name=self.agent.agent_name,
            session_id=context_id,
            user_id="a2a",
            endpoint="/a2a/tasks/stream",
            input_message=query,
            output_response=last_text,
            request_headers={},
            model_info=last_response.get('model') if isinstance(last_response, dict) else None,
            token_usage=last_response.get('token_usage') if isinstance(last_response, dict) else None,
            response_time_ms=response_time_ms,
            status="success"
        )

        if "monitoring" in self.allowed_modes:
            asyncio.create_task(
                self.llm_judge_service.judge_interaction(
                    interaction_id=task_id,
                    user_message=query,
                    agent_response=last_text,
                    session_id=context_id,
                    user_id="a2a"
                )
            )

        # Final chunk marker
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=new_text_artifact(name="response", text=""),
                append=True,
                last_chunk=True,
            )
        )

        # Completed status
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=new_agent_text_message(last_text),
                ),
                final=True,
            )
        )

    async def _emit_single_artifact(
        self, text: str, task_id: str, context_id: str, event_queue: EventQueue
    ) -> None:
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=new_text_artifact(name="response", text=text),
                append=False,
                last_chunk=True,
            )
        )
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.completed,
                    message=new_agent_text_message(text),
                ),
                final=True,
            )
        )

    # ------------------------------------------------------------------
    # cancel()
    # ------------------------------------------------------------------

    @override
    async def cancel(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """
        Signal cancellation. The streaming loop checks _cancelled_tasks
        between chunks and exits cleanly.
        """
        task_id = context.task_id
        if task_id:
            self._cancelled_tasks.add(task_id)
            logger.info(f"Cancellation requested for task '{task_id}'")

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.canceled),
                final=True,
            )
        )
