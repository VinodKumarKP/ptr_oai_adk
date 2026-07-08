"""Tests for the AG-UI gateway (A2A -> AG-UI event translation)."""

import pytest

pytest.importorskip("ag_ui")

from ag_ui.core import EventType, RunAgentInput

from oai_agent_server.agui.gateway import (
    A2AStreamTranslator,
    build_a2a_request,
    CUSTOM_EVENT_A2UI,
    CUSTOM_EVENT_DATA,
)
from oai_agent_server.routers.agui import create_agui_router


def make_run_input(**overrides):
    payload = {
        "threadId": "thread-1",
        "runId": "run-1",
        "state": None,
        "messages": [
            {"id": "m1", "role": "user", "content": "Plan a trip to NYC"},
        ],
        "tools": [],
        "context": [],
        "forwardedProps": {},
    }
    payload.update(overrides)
    return RunAgentInput.model_validate(payload)


# ---------------------------------------------------------------------------
# build_a2a_request
# ---------------------------------------------------------------------------

class TestBuildA2ARequest:
    def test_maps_last_user_message_to_text_part(self):
        run_input = make_run_input(messages=[
            {"id": "m1", "role": "user", "content": "first"},
            {"id": "m2", "role": "assistant", "content": "reply"},
            {"id": "m3", "role": "user", "content": "second"},
        ])
        request = build_a2a_request(run_input)

        assert request["method"] == "SendStreamingMessage"
        assert request["id"] == "run-1"
        parts = request["params"]["message"]["parts"]
        assert parts == [{"text": "second"}]
        assert request["params"]["message"]["role"] == "ROLE_USER"
        assert "contextId" not in request["params"]["message"]

    def test_context_id_continues_conversation(self):
        request = build_a2a_request(make_run_input(), context_id="ctx-9")
        assert request["params"]["message"]["contextId"] == "ctx-9"

    def test_a2ui_action_becomes_data_part(self):
        action = {"version": "v0.9", "action": {"name": "book_hotel"}}
        run_input = make_run_input(forwardedProps={"a2uiAction": action})
        parts = build_a2a_request(run_input)["params"]["message"]["parts"]

        data_parts = [p for p in parts if "data" in p]
        assert len(data_parts) == 1
        assert data_parts[0]["data"] == action
        assert data_parts[0]["mediaType"] == "application/json+a2ui"
        assert data_parts[0]["metadata"] == {"mimeType": "application/json+a2ui"}

    def test_action_only_run_omits_stale_prompt(self):
        # After a normal run the client history ends with the assistant reply;
        # a Book-button run must not re-send the previous user prompt.
        action = {"action": {"name": "book_hotel"}}
        run_input = make_run_input(
            messages=[
                {"id": "m1", "role": "user", "content": "find hotels"},
                {"id": "m2", "role": "assistant", "content": "here are hotels"},
            ],
            forwardedProps={"a2uiAction": action},
        )
        parts = build_a2a_request(run_input)["params"]["message"]["parts"]
        assert parts == [{
            "data": action,
            "mediaType": "application/json+a2ui",
            "metadata": {"mimeType": "application/json+a2ui"},
        }]


# ---------------------------------------------------------------------------
# A2AStreamTranslator
# ---------------------------------------------------------------------------

class TestTranslator:
    def test_task_event_captures_ids_only(self):
        translator = A2AStreamTranslator()
        events = translator.translate({"task": {
            "id": "task-1", "contextId": "ctx-1",
            "status": {"state": "TASK_STATE_SUBMITTED"},
        }})
        assert events == []
        assert translator.task_id == "task-1"
        assert translator.context_id == "ctx-1"
        assert not translator.finished

    def test_working_status_starts_step_once(self):
        translator = A2AStreamTranslator()
        first = translator.translate({"statusUpdate": {
            "taskId": "t", "status": {"state": "TASK_STATE_WORKING"},
        }})
        second = translator.translate({"statusUpdate": {
            "taskId": "t", "status": {"state": "TASK_STATE_WORKING"},
        }})
        assert [e.type for e in first] == [EventType.STEP_STARTED]
        assert second == []

    def test_streamed_text_chunks_share_one_message(self):
        translator = A2AStreamTranslator()
        chunk1 = translator.translate({"artifactUpdate": {
            "taskId": "t",
            "artifact": {"artifactId": "a1", "parts": [{"text": "Hello"}]},
        }})
        chunk2 = translator.translate({"artifactUpdate": {
            "taskId": "t", "append": True, "lastChunk": True,
            "artifact": {"artifactId": "a1", "parts": [{"text": " world"}]},
        }})

        assert [e.type for e in chunk1] == [
            EventType.TEXT_MESSAGE_START, EventType.TEXT_MESSAGE_CONTENT,
        ]
        assert [e.type for e in chunk2] == [
            EventType.TEXT_MESSAGE_CONTENT, EventType.TEXT_MESSAGE_END,
        ]
        assert chunk1[0].message_id == "a1"
        assert chunk1[1].delta == "Hello"
        assert chunk2[0].delta == " world"

    def test_a2ui_part_passes_through_as_custom_event(self):
        a2ui_message = {"createSurface": {"surfaceId": "s1"}}
        translator = A2AStreamTranslator()
        events = translator.translate({"artifactUpdate": {
            "taskId": "t", "lastChunk": True,
            "artifact": {"artifactId": "a2", "parts": [{
                "data": a2ui_message,
                "mediaType": "application/json+a2ui",
            }]},
        }})

        assert [e.type for e in events] == [EventType.CUSTOM]
        assert events[0].name == CUSTOM_EVENT_A2UI
        assert events[0].value == a2ui_message

    def test_a2ui_mime_in_metadata_only_is_recognized(self):
        translator = A2AStreamTranslator()
        events = translator.translate({"artifactUpdate": {
            "artifact": {"artifactId": "a", "parts": [{
                "data": {"beginRendering": {}},
                "metadata": {"mimeType": "application/a2ui+json"},
            }]},
        }})
        assert events[0].name == CUSTOM_EVENT_A2UI

    def test_unknown_data_part_becomes_generic_custom_event(self):
        translator = A2AStreamTranslator()
        events = translator.translate({"artifactUpdate": {
            "artifact": {"artifactId": "a", "parts": [{"data": {"x": 1}}]},
        }})
        assert events[0].name == CUSTOM_EVENT_DATA

    def test_completion_closes_open_messages_and_step(self):
        translator = A2AStreamTranslator()
        translator.translate({"statusUpdate": {
            "status": {"state": "TASK_STATE_WORKING"},
        }})
        translator.translate({"artifactUpdate": {
            "artifact": {"artifactId": "a1", "parts": [{"text": "partial"}]},
        }})
        events = translator.translate({"statusUpdate": {
            "status": {"state": "TASK_STATE_COMPLETED"},
        }})

        assert [e.type for e in events] == [
            EventType.TEXT_MESSAGE_END, EventType.STEP_FINISHED,
        ]
        assert translator.finished
        assert translator.error_message is None

    def test_failed_state_records_error(self):
        translator = A2AStreamTranslator()
        translator.translate({"statusUpdate": {"status": {
            "state": "TASK_STATE_FAILED",
            "message": {"parts": [{"text": "tool exploded"}]},
        }}})
        assert translator.finished
        assert translator.error_message == "tool exploded"

    def test_completed_task_with_artifacts_translates_them(self):
        # Non-streaming fallback: one Task carrying everything.
        translator = A2AStreamTranslator()
        events = translator.translate({"task": {
            "id": "t", "contextId": "c",
            "status": {"state": "TASK_STATE_COMPLETED"},
            "artifacts": [{"artifactId": "a1", "parts": [
                {"text": "done"},
                {"data": {"createSurface": {}}, "mediaType": "application/json+a2ui"},
            ]}],
        }})
        types = [e.type for e in events]
        assert EventType.TEXT_MESSAGE_START in types
        assert EventType.TEXT_MESSAGE_CONTENT in types
        assert EventType.CUSTOM in types
        assert EventType.TEXT_MESSAGE_END in types
        assert translator.finished

    def test_close_is_idempotent(self):
        translator = A2AStreamTranslator()
        translator.translate({"artifactUpdate": {
            "artifact": {"artifactId": "a1", "parts": [{"text": "x"}]},
        }})
        first = translator.close()
        assert [e.type for e in first] == [EventType.TEXT_MESSAGE_END]
        assert translator.close() == []


# ---------------------------------------------------------------------------
# Router gating
# ---------------------------------------------------------------------------

class TestRouterGating:
    def test_requires_a2a_mode(self):
        assert create_agui_router(allowed_modes=["chat"]) is None

    def test_enabled_with_a2a_mode(self):
        router = create_agui_router(allowed_modes=["a2a"])
        assert router is not None
        assert any(r.path == "/" for r in router.routes)

    def test_env_kill_switch(self, monkeypatch):
        monkeypatch.setenv("ENABLE_AGUI_GATEWAY", "false")
        assert create_agui_router(allowed_modes=["a2a"]) is None
