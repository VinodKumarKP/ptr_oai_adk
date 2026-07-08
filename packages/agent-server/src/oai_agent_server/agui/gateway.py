"""AG-UI gateway: translates A2A streaming responses into AG-UI events.

Lets any AG-UI client (CopilotKit, ``@ag-ui/client`` ``HttpAgent``, ...) drive
this server's agent without speaking A2A::

    AG-UI client --- RunAgentInput ---> POST /agui ---> POST /a2a  (SendStreamingMessage)
    AG-UI client <-- SSE AG-UI events -- translator <-- SSE StreamResponse events

Only the envelope is translated. A2UI payloads (``application/json+a2ui`` /
``application/a2ui+json`` data parts) pass through verbatim as AG-UI CUSTOM
events named ``a2ui.message`` so an A2UI renderer on the client consumes the
exact JSON it would receive over plain A2A.

Event mapping (A2A ``StreamResponse`` -> AG-UI):

    task                      capture taskId/contextId (+ translate any artifacts)
    statusUpdate WORKING      STEP_STARTED("working")
    statusUpdate terminal     STEP_FINISHED("working"), ends the run
    artifactUpdate text part  TEXT_MESSAGE_START / _CONTENT / _END (per artifactId)
    artifactUpdate a2ui part  CUSTOM {name: "a2ui.message", value: <part data>}
    artifactUpdate other data CUSTOM {name: "a2a.data", value: <part data>}
    JSON-RPC error            RUN_ERROR

The gateway targets this server's own /a2a endpoint by default; point
``AGUI_A2A_URL`` at any other A2A agent to front it instead.
"""

import json
import os
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx

from ag_ui.core import (
    BaseEvent,
    CustomEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
)
from ag_ui.encoder import EventEncoder

from oai_agent_core.utils.logger import get_logger

# Mirrors oai_agent_server.a2a.a2ui_support — kept local so this module has no
# import-time dependency on the a2a-sdk.
A2UI_MIME_TYPES = {"application/a2ui+json", "application/json+a2ui"}
A2UI_EXTENSION_URI = "https://a2ui.org/a2a-extension/a2ui/v0.9"

# Key in RunAgentInput.forwardedProps carrying an A2UI user action (the JSON a
# renderer produces when e.g. a Book button is pressed). Sent to the agent as
# an A2UI data part, exactly like a native A2UI client would.
A2UI_ACTION_PROP = "a2uiAction"

CUSTOM_EVENT_A2UI = "a2ui.message"
CUSTOM_EVENT_DATA = "a2a.data"

WORKING_STEP_NAME = "working"

TERMINAL_TASK_STATES = {
    "TASK_STATE_COMPLETED",
    "TASK_STATE_FAILED",
    "TASK_STATE_CANCELED",
    "TASK_STATE_REJECTED",
    "TASK_STATE_INPUT_REQUIRED",
    "TASK_STATE_AUTH_REQUIRED",
}
FAILED_TASK_STATES = {"TASK_STATE_FAILED", "TASK_STATE_REJECTED"}


def build_a2a_request(run_input, context_id: Optional[str] = None) -> Dict[str, Any]:
    """Builds the JSON-RPC SendStreamingMessage payload for a RunAgentInput.

    The last user message becomes the text part; a pending A2UI action in
    ``forwardedProps.a2uiAction`` becomes an A2UI data part. ``context_id``
    (from a previous run on the same thread) continues the A2A conversation.
    """
    # Only send text when the run was triggered by a fresh user message (the
    # history then ends with it). An action-only run (e.g. a Book button, sent
    # via forwardedProps) ends with an assistant message — re-sending the
    # previous prompt alongside the action would make the agent answer it again.
    text = None
    messages = run_input.messages or []
    if messages and getattr(messages[-1], "role", None) == "user":
        text = getattr(messages[-1], "content", None)

    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"text": text})

    forwarded = run_input.forwarded_props or {}
    action = forwarded.get(A2UI_ACTION_PROP) if isinstance(forwarded, dict) else None
    if action is not None:
        mime = "application/json+a2ui"
        parts.append({
            "data": action,
            "mediaType": mime,
            "metadata": {"mimeType": mime},
        })

    message: Dict[str, Any] = {
        "messageId": str(uuid.uuid4()),
        "role": "ROLE_USER",
        "parts": parts,
    }
    if context_id:
        message["contextId"] = context_id

    return {
        "jsonrpc": "2.0",
        "id": run_input.run_id or "1",
        "method": "SendStreamingMessage",
        "params": {"message": message},
    }


class A2AStreamTranslator:
    """Stateful per-run translator from A2A StreamResponse dicts to AG-UI events.

    Feed it each JSON-RPC ``result`` object from the A2A SSE stream via
    :meth:`translate`; call :meth:`close` once the stream ends to close any
    text messages / steps left open (e.g. when the stream is cut short).
    """

    def __init__(self):
        self.context_id: Optional[str] = None
        self.task_id: Optional[str] = None
        self.finished = False
        self.error_message: Optional[str] = None
        self._open_messages: List[str] = []
        self._working = False

    def translate(self, result: Dict[str, Any]) -> List[BaseEvent]:
        events: List[BaseEvent] = []
        if "task" in result:
            self._on_task(result["task"], events)
        elif "statusUpdate" in result:
            self._on_status(result["statusUpdate"], events)
        elif "artifactUpdate" in result:
            self._on_artifact_update(result["artifactUpdate"], events)
        elif "message" in result:
            self._on_message(result["message"], events)
        return events

    def close(self) -> List[BaseEvent]:
        """Ends any text message / step still open. Safe to call twice."""
        events: List[BaseEvent] = []
        for message_id in self._open_messages:
            events.append(TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END, message_id=message_id,
            ))
        self._open_messages = []
        if self._working:
            self._working = False
            events.append(StepFinishedEvent(
                type=EventType.STEP_FINISHED, step_name=WORKING_STEP_NAME,
            ))
        return events

    # -- A2A event kinds ---------------------------------------------------

    def _on_task(self, task: Dict[str, Any], events: List[BaseEvent]) -> None:
        self.task_id = task.get("id") or self.task_id
        self.context_id = task.get("contextId") or self.context_id
        # Non-streaming servers may answer with a single completed Task that
        # already carries the artifacts.
        for artifact in task.get("artifacts") or []:
            self._emit_parts(artifact.get("parts") or [],
                             message_id=artifact.get("artifactId"),
                             last_chunk=True, events=events)
        state = (task.get("status") or {}).get("state")
        self._apply_state(state, task.get("status") or {}, events)

    def _on_status(self, update: Dict[str, Any], events: List[BaseEvent]) -> None:
        self.task_id = update.get("taskId") or self.task_id
        self.context_id = update.get("contextId") or self.context_id
        status = update.get("status") or {}
        # Agents sometimes attach a progress message to the status itself.
        status_message = status.get("message")
        if status_message:
            self._emit_parts(status_message.get("parts") or [],
                             message_id=status_message.get("messageId"),
                             last_chunk=True, events=events)
        self._apply_state(status.get("state"), status, events)

    def _on_artifact_update(self, update: Dict[str, Any], events: List[BaseEvent]) -> None:
        self.task_id = update.get("taskId") or self.task_id
        self.context_id = update.get("contextId") or self.context_id
        artifact = update.get("artifact") or {}
        self._emit_parts(artifact.get("parts") or [],
                         message_id=artifact.get("artifactId"),
                         last_chunk=bool(update.get("lastChunk")), events=events)

    def _on_message(self, message: Dict[str, Any], events: List[BaseEvent]) -> None:
        self.context_id = message.get("contextId") or self.context_id
        self._emit_parts(message.get("parts") or [],
                         message_id=message.get("messageId"),
                         last_chunk=True, events=events)

    # -- helpers -------------------------------------------------------------

    def _apply_state(self, state: Optional[str], status: Dict[str, Any],
                     events: List[BaseEvent]) -> None:
        if state == "TASK_STATE_WORKING" and not self._working:
            self._working = True
            events.append(StepStartedEvent(
                type=EventType.STEP_STARTED, step_name=WORKING_STEP_NAME,
            ))
        elif state in TERMINAL_TASK_STATES:
            events.extend(self.close())
            self.finished = True
            if state in FAILED_TASK_STATES:
                self.error_message = self._status_text(status) or f"task ended in {state}"

    @staticmethod
    def _status_text(status: Dict[str, Any]) -> Optional[str]:
        parts = (status.get("message") or {}).get("parts") or []
        texts = [p["text"] for p in parts if p.get("text")]
        return " ".join(texts) or None

    def _emit_parts(self, parts: List[Dict[str, Any]], message_id: Optional[str],
                    last_chunk: bool, events: List[BaseEvent]) -> None:
        message_id = message_id or str(uuid.uuid4())
        for part in parts:
            text = part.get("text")
            if text:
                if message_id not in self._open_messages:
                    self._open_messages.append(message_id)
                    events.append(TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id, role="assistant",
                    ))
                events.append(TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id, delta=text,
                ))
                continue
            data = part.get("data")
            if data is not None:
                mime = part.get("mediaType") or (part.get("metadata") or {}).get("mimeType")
                name = CUSTOM_EVENT_A2UI if mime in A2UI_MIME_TYPES else CUSTOM_EVENT_DATA
                events.append(CustomEvent(type=EventType.CUSTOM, name=name, value=data))
        if last_chunk and message_id in self._open_messages:
            self._open_messages.remove(message_id)
            events.append(TextMessageEndEvent(
                type=EventType.TEXT_MESSAGE_END, message_id=message_id,
            ))


async def _iter_sse_json(response: httpx.Response) -> AsyncIterator[Dict[str, Any]]:
    """Yields the JSON object of each SSE `data:` event on the response."""
    data_lines: List[str] = []
    async for line in response.aiter_lines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line == "" and data_lines:
            payload = "\n".join(data_lines)
            data_lines = []
            try:
                yield json.loads(payload)
            except json.JSONDecodeError:
                get_logger().warning("AG-UI gateway: skipping non-JSON SSE event")
    if data_lines:  # stream ended without the final blank line
        try:
            yield json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            pass


async def stream_agui_events(
    a2a_url: str,
    run_input,
    thread_contexts: Dict[str, str],
    timeout_seconds: float = 300.0,
) -> AsyncIterator[str]:
    """Runs one AG-UI run against an A2A agent, yielding encoded SSE events.

    ``thread_contexts`` maps AG-UI threadIds to A2A contextIds so follow-up
    runs on the same thread continue the same A2A conversation.
    """
    logger = get_logger()
    encoder = EventEncoder()
    translator = A2AStreamTranslator()
    thread_id = run_input.thread_id or str(uuid.uuid4())

    yield encoder.encode(RunStartedEvent(
        type=EventType.RUN_STARTED, thread_id=thread_id, run_id=run_input.run_id,
    ))

    payload = build_a2a_request(run_input, context_id=thread_contexts.get(thread_id))
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream, application/json",
        "A2A-Version": "1.0",
        "A2A-Extensions": A2UI_EXTENSION_URI,
    }

    error_message: Optional[str] = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds)) as client:
            async with client.stream("POST", a2a_url, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    body = (await response.aread()).decode(errors="replace")
                    error_message = f"A2A endpoint returned {response.status_code}: {body[:500]}"
                elif "text/event-stream" not in response.headers.get("content-type", ""):
                    # Non-streaming server: single JSON-RPC response.
                    body = json.loads((await response.aread()).decode())
                    if body.get("error"):
                        error_message = str(body["error"].get("message") or body["error"])
                    else:
                        for event in translator.translate(body.get("result") or {}):
                            yield encoder.encode(event)
                else:
                    async for rpc in _iter_sse_json(response):
                        if rpc.get("error"):
                            error_message = str(rpc["error"].get("message") or rpc["error"])
                            break
                        for event in translator.translate(rpc.get("result") or {}):
                            yield encoder.encode(event)
                        if translator.finished:
                            break
    except httpx.HTTPError as exc:
        error_message = f"A2A request failed: {exc}"
        logger.warning("AG-UI gateway: %s", error_message)

    for event in translator.close():
        yield encoder.encode(event)

    if translator.context_id:
        thread_contexts[thread_id] = translator.context_id

    if error_message or translator.error_message:
        yield encoder.encode(RunErrorEvent(
            type=EventType.RUN_ERROR,
            message=error_message or translator.error_message,
        ))
    else:
        yield encoder.encode(RunFinishedEvent(
            type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_input.run_id,
        ))
