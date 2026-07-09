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
    StateSnapshotEvent,
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

# Clients that can't set forwardedProps (e.g. CopilotKit's chat) may instead
# send the action as a user message with this prefix followed by the action
# JSON; the gateway converts it to the same A2UI data part.
A2UI_ACTION_TEXT_PREFIX = "[UI action] "

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

    forwarded = run_input.forwarded_props or {}
    action = forwarded.get(A2UI_ACTION_PROP) if isinstance(forwarded, dict) else None

    # "[UI action] {...}" text convention → same data part as forwardedProps.
    if action is None and isinstance(text, str) and text.startswith(A2UI_ACTION_TEXT_PREFIX):
        try:
            action = json.loads(text[len(A2UI_ACTION_TEXT_PREFIX):])
            text = None
        except json.JSONDecodeError:
            pass  # malformed — leave it as plain text for the agent to read

    parts: List[Dict[str, Any]] = []
    if text:
        parts.append({"text": text})
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


def _norm(text: str) -> str:
    return " ".join(text.split())


def _looks_like_data_dump(text: str) -> bool:
    """True for text parts that are really machine payloads (tool-result
    JSON, Python message reprs) leaked into the conversation."""
    t = text.strip()
    if t.startswith("The user triggered the UI action"):
        return True
    if not t[:1] in "{[":
        return False
    try:
        json.loads(t)
        return True
    except json.JSONDecodeError:
        return "'type':" in t or "'content':" in t


class A2AStreamTranslator:
    """Stateful per-run translator from A2A StreamResponse dicts to AG-UI events.

    Feed it each JSON-RPC ``result`` object from the A2A SSE stream via
    :meth:`translate`; call :meth:`close` once the stream ends to close any
    text messages / steps left open (e.g. when the stream is cut short).
    """

    def __init__(
        self,
        initial_a2ui_messages: Optional[List[Any]] = None,
        user_text: Optional[str] = None,
        tidy: bool = True,
    ):
        self.context_id: Optional[str] = None
        self.task_id: Optional[str] = None
        self.finished = False
        self.error_message: Optional[str] = None
        self._open_messages: List[str] = []
        self._working = False
        # Chat hygiene (disable with tidy=False / AGUI_RAW=true): agents echo
        # the user's prompt, leak raw tool-result JSON, and re-send a final
        # summary artifact duplicating what was already streamed. Filtering
        # here fixes every AG-UI client at once — off-the-shelf UIs like
        # CopilotKit render the stream verbatim.
        self._tidy = tidy
        self._user_text_norm = _norm(user_text) if user_text else None
        self._message_text: Dict[str, str] = {}   # open message -> text so far
        self._completed_norm: List[str] = []      # normalized finished messages
        # All A2UI messages seen on this thread, mirrored into AG-UI shared
        # state (STATE_SNAPSHOT.a2uiMessages) so state-centric clients — e.g.
        # CopilotKit's useCoAgent — receive them without handling CUSTOM
        # events. Seeded with earlier runs' messages so the client state
        # keeps the whole conversation's surfaces, not just the last run's.
        # Event-centric clients keep using the CUSTOM channel.
        self.a2ui_messages: List[Any] = list(initial_a2ui_messages or [])

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
            self._completed_norm.append(_norm(self._message_text.pop(message_id, "")))
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

    def _skip_text(self, text: str, message_id: str, last_chunk: bool) -> bool:
        if not self._tidy:
            return False
        norm = _norm(text)
        # Echo of the user's own prompt / leaked machine payload.
        if norm == self._user_text_norm or _looks_like_data_dump(text):
            return True
        # Single-shot artifact repeating an already-streamed message (agents
        # often close a run with a consolidated summary of the same text).
        if last_chunk and message_id not in self._open_messages:
            return any(norm in done or done in norm for done in self._completed_norm if done)
        return False

    def _emit_parts(self, parts: List[Dict[str, Any]], message_id: Optional[str],
                    last_chunk: bool, events: List[BaseEvent]) -> None:
        message_id = message_id or str(uuid.uuid4())
        for part in parts:
            text = part.get("text")
            if text:
                if self._skip_text(text, message_id, last_chunk):
                    continue
                if message_id not in self._open_messages:
                    self._open_messages.append(message_id)
                    events.append(TextMessageStartEvent(
                        type=EventType.TEXT_MESSAGE_START,
                        message_id=message_id, role="assistant",
                    ))
                # Dropped segments (echo/tool dumps) can butt two paragraphs
                # together — restore a break at a clear sentence boundary.
                # Whitespace-carrying deltas are untouched, so token-level
                # streams are unaffected.
                so_far = self._message_text.get(message_id, "")
                if self._tidy and so_far and so_far[-1] in ".!?…" and text[:1].isupper():
                    text = "\n\n" + text
                self._message_text[message_id] = so_far + text
                events.append(TextMessageContentEvent(
                    type=EventType.TEXT_MESSAGE_CONTENT,
                    message_id=message_id, delta=text,
                ))
                continue
            data = part.get("data")
            if data is not None:
                mime = part.get("mediaType") or (part.get("metadata") or {}).get("mimeType")
                if mime in A2UI_MIME_TYPES:
                    events.append(CustomEvent(type=EventType.CUSTOM, name=CUSTOM_EVENT_A2UI, value=data))
                    self.a2ui_messages.append(data)
                    events.append(StateSnapshotEvent(
                        type=EventType.STATE_SNAPSHOT,
                        snapshot={"a2uiMessages": list(self.a2ui_messages)},
                    ))
                else:
                    events.append(CustomEvent(type=EventType.CUSTOM, name=CUSTOM_EVENT_DATA, value=data))
        if last_chunk and message_id in self._open_messages:
            self._open_messages.remove(message_id)
            self._completed_norm.append(_norm(self._message_text.pop(message_id, "")))
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
    thread_a2ui: Optional[Dict[str, List[Any]]] = None,
    timeout_seconds: float = 300.0,
) -> AsyncIterator[str]:
    """Runs one AG-UI run against an A2A agent, yielding encoded SSE events.

    ``thread_contexts`` maps AG-UI threadIds to A2A contextIds so follow-up
    runs on the same thread continue the same A2A conversation.
    ``thread_a2ui`` accumulates each thread's A2UI messages across runs so
    state snapshots always describe the whole conversation's surfaces.
    """
    logger = get_logger()
    encoder = EventEncoder()
    thread_id = run_input.thread_id or str(uuid.uuid4())
    thread_a2ui = thread_a2ui if thread_a2ui is not None else {}
    messages = run_input.messages or []
    user_text = (
        messages[-1].content
        if messages and getattr(messages[-1], "role", None) == "user"
        else None
    )
    translator = A2AStreamTranslator(
        initial_a2ui_messages=thread_a2ui.get(thread_id),
        user_text=user_text if isinstance(user_text, str) else None,
        tidy=os.environ.get("AGUI_RAW", "false").lower() != "true",
    )

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
    if translator.a2ui_messages:
        thread_a2ui[thread_id] = translator.a2ui_messages

    if error_message or translator.error_message:
        yield encoder.encode(RunErrorEvent(
            type=EventType.RUN_ERROR,
            message=error_message or translator.error_message,
        ))
    else:
        yield encoder.encode(RunFinishedEvent(
            type=EventType.RUN_FINISHED, thread_id=thread_id, run_id=run_input.run_id,
        ))
