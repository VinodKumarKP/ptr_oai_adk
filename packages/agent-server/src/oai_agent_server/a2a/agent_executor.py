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

File Attachment Support (A2A protocol):
  Incoming messages may contain parts of kind='file'. Each file part has:
    - file.name      (str)            — original filename
    - file.mimeType  (str)            — MIME type
    - file.bytes     (str | None)     — base64-encoded content (inline)
    - file.uri       (str | None)     — remote URI (alternative to bytes)

  _extract_input() assembles a rich input dict passed to ainvoke/astream:
    {
      "text":  "<user text>",
      "files": [
        {
          "name":      "report.pdf",
          "mime_type": "application/pdf",
          "bytes":     "<base64 string>",   # if inline
          "uri":       None,                # if URI-based
          "data":      b"<raw bytes>",      # decoded bytes (inline only)
        },
        ...
      ]
    }

  Your BaseAgent.ainvoke / astream should accept `user_message` as either
  a plain str (backward-compatible) or this dict when files are present.
"""

import asyncio
import base64
import logging
import os
import shutil
import tempfile
import time
import inspect
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    Part,
    Message,
    Part,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent
)
from a2a.helpers import new_task, new_text_artifact, new_text_message
from typing_extensions import override

from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_server.services.llm_judge_service import LLMJudgeService
from oai_agent_server.utils.database_logger import DatabaseLogger

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers for parsing A2A message parts
# ---------------------------------------------------------------------------

def _parse_file_part(file_part: Part) -> Dict[str, Any]:
    """
    Convert an A2A FilePart into a normalised dict your agent can consume.

    Returns:
        {
            "name":      str,         # original filename (may be empty string)
            "mime_type": str,         # MIME type (may be empty string)
            "bytes":     str | None,  # raw base64 string as sent over the wire
            "uri":       str | None,  # remote URI (mutually exclusive with bytes)
            "data":      bytes | None # decoded bytes (only when inline bytes present)
        }
    """
    file = file_part.filename
    raw_b64: Optional[str] = getattr(file_part, "raw", None)
    uri: Optional[str] = getattr(file, "uri", None)

    return {
        "name": getattr(file_part, "filename", "") or "",
        "mime_type": getattr(file_part, "mimeType", "") or "",
        "bytes": raw_b64,
        "uri": uri,
        "data": raw_b64,
    }


def _extract_input(message: Message) -> Dict[str, Any]:
    """
    Walk all parts of an A2A Message and return a structured input dict:

        {
            "text":  "<concatenated text from all TextParts>",
            "files": [ <parsed file dicts> ]
        }

    When there are no file parts the "files" list is empty, so callers can
    check  `bool(result["files"])`  to decide whether to forward attachments.
    """
    texts: List[str] = []
    files: List[Dict[str, Any]] = []

    parts: List[Part] = getattr(message, "parts", []) or []
    for part in parts:
        # A2A SDK wraps each part in a union; access via .root or directly
        actual = getattr(part, "root", part)

        if isinstance(actual, Part):
            if actual.text:
                texts.append(actual.text)
            elif actual.filename:
                files.append(_parse_file_part(actual))

        # elif isinstance(actual, Part):
        #     try:
        #         files.append(_parse_file_part(actual))
        #     except Exception as exc:
        #         logger.warning("Failed to parse file part: %s", exc)

    return {
        "text": "\n".join(texts),
        "files": files,
    }


# ---------------------------------------------------------------------------
# BaseAgentExecutor
# ---------------------------------------------------------------------------

class BaseAgentExecutor(AgentExecutor):
    """
    Adapter between a2a-sdk's AgentExecutor interface and your BaseAgent.

    Supports both:
      • Synchronous invoke  → ainvoke()  (tasks/send)
      • Streaming invoke    → astream()  (tasks/stream / sendSubscribe)

    Both code paths funnel through the same execute() method — the SDK
    decides how to deliver the events to the caller (batch vs SSE).

    File attachments arriving as A2A FilePart(s) are decoded and forwarded
    to the agent as part of the `user_message` dict (see module docstring).
    """

    def __init__(
        self,
        agent: BaseAgent,
        db_logger: DatabaseLogger,
        llm_judge_service: LLMJudgeService,
        allowed_modes: List[str],
        use_streaming: bool = True,
    ):
        """
        Args:
            agent:              Your BaseAgent instance.
            db_logger:          DatabaseLogger for interaction logging.
            llm_judge_service:  LLMJudgeService for quality monitoring.
            allowed_modes:      Feature flags (e.g. ["monitoring"]).
            use_streaming:      If True and agent has astream(), use it so the
                                SDK can stream partial results to the caller.
                                Set False to always use ainvoke().
        """
        self.agent = agent
        self.db_logger = db_logger
        self.llm_judge_service = llm_judge_service
        self.allowed_modes = allowed_modes
        self.use_streaming = use_streaming and hasattr(agent, "astream")
        # TTL-bounded cancellation registry: {task_id: timestamp_added}
        # Pruned opportunistically; entries older than _CANCELLED_TASK_TTL
        # are evicted on every access.
        self._cancelled_tasks: Dict[str, float] = {}
        self._CANCELLED_TASK_TTL = 300.0  # 5 minutes

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

    def _prune_cancelled_tasks(self) -> None:
        """Evict cancelled-task records older than _CANCELLED_TASK_TTL."""
        if not self._cancelled_tasks:
            return
        cutoff = time.time() - self._CANCELLED_TASK_TTL
        stale = [tid for tid, ts in self._cancelled_tasks.items() if ts < cutoff]
        for tid in stale:
            self._cancelled_tasks.pop(tid, None)

    def _is_cancelled(self, task_id: Optional[str]) -> bool:
        self._prune_cancelled_tasks()
        return task_id is not None and task_id in self._cancelled_tasks

    @staticmethod
    def _log_input_summary(extracted: Dict[str, Any], task_id: str) -> None:
        """Emit a structured log line summarising what arrived."""
        file_summary = [
            f"{f['name'] or '<unnamed>'} ({f['mime_type'] or 'unknown'}, "
            f"{'inline ' + str(len(f['data'])) + 'B' if f['data'] else 'uri=' + str(f['uri'])})"
            for f in extracted["files"]
        ]
        logger.info(
            "Task '%s' — text length=%d, attachments=[%s]",
            task_id,
            len(extracted["text"]),
            ", ".join(file_summary) if file_summary else "none",
        )

    # ------------------------------------------------------------------
    # execute() — called for BOTH tasks/send and tasks/sendSubscribe
    # ------------------------------------------------------------------

    def _save_files(self, files: List[Dict[str, Any]]) -> Tuple[List[str], Optional[str]]:
        """
        Save file data to a temporary directory.

        Args:
            files: A list of file dictionaries, each from _parse_file_part.

        Returns:
            A tuple containing:
            - A list of absolute paths to the saved files.
            - The path to the temporary directory created, or None.
        """
        if not files:
            return [], None

        temp_dir = tempfile.mkdtemp()
        file_paths = []

        for file_info in files:
            data = file_info.get('data')
            name = file_info.get('name')
            if data and name:
                file_path = os.path.join(temp_dir, name)
                try:
                    with open(file_path, 'wb') as f:
                        f.write(data)
                    file_paths.append(file_path)
                except IOError as e:
                    logger.error(f"Error writing file {name} to {temp_dir}: {e}")
        return file_paths, temp_dir

    async def _cleanup_directory(self, path: str):
        """Asynchronously removes a directory and its contents."""
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, shutil.rmtree, path)
            logger.info(f"Successfully scheduled cleanup for temp directory: {path}")
        except Exception as e:
            logger.error(f"Failed to cleanup temp directory {path}: {e}")

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
          2. Parse all message parts (text + files) via _extract_input().
          3. Emit TaskStatusUpdateEvent(working).
          4. Call ainvoke() or astream() on your BaseAgent.
          5. Emit TaskArtifactUpdateEvent(s) with the response content.
          6. Emit TaskStatusUpdateEvent(completed | failed).
        """
        if not context.current_task:
            task = new_task(context.task_id, context.context_id, state=TaskState.TASK_STATE_WORKING)
            await event_queue.enqueue_event(task)

        task_id = context.task_id
        context_id = context.context_id

        extracted = _extract_input(context.message)
        self._log_input_summary(extracted, task_id)
        query_text: str = extracted["text"]
        config = self._build_config(context)
        temp_directory = None
        query_for_agent = query_text

        try:
            if extracted.get('files'):
                file_paths, temp_directory = self._save_files(extracted['files'])
                if file_paths:
                    file_list = ", ".join(file_paths)
                    query_for_agent = f"{query_text}\n\nUploaded files: {file_list}"

            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_WORKING,
                        message=new_text_message("Processing your request…"),
                    )
                )
            )

            if self.use_streaming:
                await self._execute_streaming(
                    query_for_agent, query_text, config, task_id, context_id, event_queue
                )
            else:
                await self._execute_batch(
                    query_for_agent, query_text, config, task_id, context_id, event_queue
                )

        except Exception as e:
            logger.exception(f"Agent execution failed for task '{task_id}': {e}")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=task_id,
                    context_id=context_id,
                    status=TaskStatus(
                        state=TaskState.TASK_STATE_FAILED,
                        message=new_text_message(f"Error: {str(e)}"),
                    )
                )
            )
        finally:
            if temp_directory:
                asyncio.create_task(self._cleanup_directory(temp_directory))

    # ------------------------------------------------------------------
    # Batch path  (ainvoke)
    # ------------------------------------------------------------------

    async def _execute_batch(
            self, query_for_agent: str, query_for_log: str, config: dict,
            task_id: str, context_id: str,
            event_queue: EventQueue,
    ) -> None:
        """Call ainvoke(), emit a single artifact, then complete."""
        start_time = time.time()
        response = await self.agent.ainvoke(user_message=query_for_agent, config=config)
        output_text = self._extract_text(response)
        response_time_ms = (time.time() - start_time) * 1000

        await self.db_logger.log_interaction(
            interaction_id=task_id,
            agent_name=self.agent.agent_name,
            session_id=context_id,
            user_id="a2a",
            endpoint="/a2a/tasks/send",
            input_message=query_for_log,
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
                    agent_name=self.agent.agent_name,
                    session_id=context_id,
                    user_message=query_for_log,
                    agent_response=output_text,
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
                    state=TaskState.TASK_STATE_COMPLETED,
                    message=new_text_message(output_text),
                )
            )
        )

    # ------------------------------------------------------------------
    # Streaming path  (astream)
    # ------------------------------------------------------------------

    async def _execute_streaming(
            self, query_for_agent: str, query_for_log: str, config: dict,
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
        stream = self.agent.astream(user_message=query_for_agent, config=config)

        if inspect.iscoroutine(stream):
            stream = await stream

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
                input_message=query_for_log,
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
                        agent_name=self.agent.agent_name,
                        session_id=context_id,
                        user_message=query_for_log,
                        agent_response=output_text,
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
            if self._is_cancelled(task_id):
                await event_queue.enqueue_event(
                    TaskStatusUpdateEvent(
                        task_id=task_id,
                        context_id=context_id,
                        status=TaskStatus(state=TaskState.TASK_STATE_CANCELED),
                    )
                )
                self._cancelled_tasks.pop(task_id, None)
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
                    append=chunk_index > 0,
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
            input_message=query_for_log,
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
                    agent_name=self.agent.agent_name,
                    session_id=context_id,
                    user_message=query_for_log,
                    agent_response=last_text,
                    user_id="a2a"
                )
            )

        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                artifact=new_text_artifact(name="response", text=""),
                append=True,
                last_chunk=True,
            )
        )

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context_id,
                status=TaskStatus(
                    state=TaskState.TASK_STATE_COMPLETED,
                    message=new_text_message(last_text),
                )
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
                    state=TaskState.TASK_STATE_COMPLETED,
                    message=new_text_message(text),
                ),
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
            self._cancelled_tasks[task_id] = time.time()
            self._prune_cancelled_tasks()
            logger.info(f"Cancellation requested for task '{task_id}'")

        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_CANCELED)
            )
        )
