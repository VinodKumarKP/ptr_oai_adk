"""Scheduled job management using APScheduler + FastAPI.

Provides REST endpoints for the UI to create, list, pause, resume, and delete
scheduled jobs. Each job calls the same agent.astream() path used by
/chat/stream, so results are identical SSE format.
"""

import asyncio
import inspect
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from oai_agent_server.utils.response_extractor import ResponseContentExtractor
from oai_agent_server.utils.serialization import make_serializable
from pydantic import BaseModel, field_validator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for the schedule API
# ---------------------------------------------------------------------------

class ScheduleRequest(BaseModel):
    """Payload the UI sends to create / update a schedule.

    Either *cron_expression* (recurring) or *run_at* (one-shot) must be set.
    If *run_now* is True the job also fires immediately in the background.
    """
    job_id: Optional[str] = None
    cron_expression: Optional[str] = None
    run_at: Optional[str] = None
    prompt: str # Changed to required, as it's the core task for the agent
    session_id: Optional[str] = None
    user_id: str = "user"
    run_now: bool = False
    enabled: bool = True

    @field_validator("cron_expression")
    @classmethod
    def validate_cron(cls, v):
        if v is not None:
            parts = v.strip().split()
            if len(parts) != 5:
                raise ValueError("cron_expression must have 5 fields: minute hour day month day_of_week")
        return v


class ScheduleRunRequest(BaseModel):
    """Payload for POST /schedule/run — triggers an immediate run and streams
    the result back in the same SSE format as /chat/stream."""
    job_id: Optional[str] = None
    prompt: str # Changed to required
    session_id: Optional[str] = None
    user_id: str = "user"


class ScheduleResponse(BaseModel):
    job_id: str
    schedule: Optional[str] = None
    next_run: Optional[str] = None
    status: str


class ScheduleInfo(BaseModel):
    job_id: str
    cron_expression: Optional[str] = None
    prompt: str
    session_id: str
    user_id: str
    enabled: bool
    next_run: Optional[str] = None
    active: bool


# ---------------------------------------------------------------------------
# In-memory stores (swap with DB / DynamoDB for production)
# ---------------------------------------------------------------------------
schedule_store: Dict[str, dict] = {}
result_store: Dict[str, List[dict]] = {}
scheduler = AsyncIOScheduler()


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_schedule_router(agent, allowed_modes: Optional[List[str]] = None) -> Optional[APIRouter]:
    """Build & return the /schedule router bound to *agent*."""

    if allowed_modes is None:
        allowed_modes = ["schedule"]

    if "schedule" not in allowed_modes:
        return None

    router = APIRouter(prefix="/schedule", tags=["schedule"])
    response_extractor = ResponseContentExtractor(agent)

    # ---- helper: stream agent response as SSE (same format as /chat/stream) ----
    async def _stream_agent(message: str, session_id: str,
                            user_id: str,
                            job_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Yield SSE events identical to /chat/stream.

        When *job_id* is given the collected chunks are stored in ``result_store``.
        """
        config = {"session_id": session_id, "user_id": user_id}

        stream_result = agent.astream(user_message=message, config=config)
        if inspect.isasyncgen(stream_result):
            stream = stream_result
        elif inspect.iscoroutine(stream_result):
            stream = await stream_result
        else:
            stream = stream_result

        collected_chunks: List[dict] = []
        async for chunk in stream:
            content = response_extractor.extract_content(chunk)
            if content:
                try:
                    json_data = json.dumps({
                        "content": content,
                        "session_id": session_id,
                    })
                    collected_chunks.append(content)
                except (TypeError, ValueError):
                    try:
                        serializable = make_serializable(content)
                        json_data = json.dumps({
                            "content": serializable,
                            "session_id": session_id,
                        })
                        collected_chunks.append(serializable)
                    except Exception:
                        json_data = json.dumps({
                            "content": str(content),
                            "session_id": session_id,
                        })
                yield f"data: {json_data}\n\n"

        # Persist results for later retrieval
        if job_id:
            run_id = uuid.uuid4().hex[:8]
            if job_id not in result_store:
                result_store[job_id] = []
            result_store[job_id].append({
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "status": "completed",
                "stream_chunks": collected_chunks,
            })

        yield "data: [DONE]\n\n"

    # ---- helper: background execution (stores full result for later retrieval) ----
    async def _execute_agent_job_bg(job_id: str, prompt: str,
                                  session_id: str, user_id: str):
        message = prompt
        run_id = uuid.uuid4().hex[:8]
        logger.info("[Scheduler] job=%s run=%s session=%s running: %s",
                    job_id, run_id, session_id, message)
        try:
            config = {"session_id": session_id, "user_id": user_id}

            # Stream to capture ALL chunks (tool results, events, final text)
            stream_chunks: List[dict] = []
            stream_result = agent.astream(user_message=message, config=config)
            if inspect.isasyncgen(stream_result):
                stream = stream_result
            elif inspect.iscoroutine(stream_result):
                stream = await stream_result
            else:
                stream = stream_result

            async for chunk in stream:
                content = response_extractor.extract_content(chunk)
                if content:
                    try:
                        serialized = json.loads(json.dumps(content))
                    except (TypeError, ValueError):
                        serialized = make_serializable(content)
                    stream_chunks.append(serialized)

            if job_id not in result_store:
                result_store[job_id] = []
            result_store[job_id].append({
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "status": "completed",
                "stream_chunks": stream_chunks,
            })
            logger.info("[Scheduler] job=%s run=%s completed", job_id, run_id)
        except Exception as exc:
            logger.error("[Scheduler] job=%s run=%s failed: %s",
                         job_id, run_id, exc, exc_info=True)
            if job_id not in result_store:
                result_store[job_id] = []
            result_store[job_id].append({
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "session_id": session_id,
                "error": str(exc),
                "status": "failed",
                "stream_chunks": [],
            })

    # ---- helper: build the right trigger ----
    def _build_trigger(req: ScheduleRequest):
        if req.cron_expression:
            parts = req.cron_expression.strip().split()
            return CronTrigger(
                minute=parts[0], hour=parts[1],
                day=parts[2], month=parts[3],
                day_of_week=parts[4],
            )
        if req.run_at:
            return DateTrigger(run_date=req.run_at)
        return None

    # ---- endpoints ----

    @router.post("", response_model=ScheduleResponse)
    async def create_schedule(request: ScheduleRequest):
        """Create a new scheduled agent job.

        Example UI payloads::

            Recurring:  {"cron_expression": "0 9 * * 1-5", "prompt": "daily report"}
            One-time:   {"run_at": "2026-04-05T09:00:00Z", "prompt": "monthly summary"}
            Immediate:  {"prompt": "quick task", "run_now": true}
        """
        job_id = request.job_id or uuid.uuid4().hex[:8]
        session_id = request.session_id or f"schedule-{job_id}"
        trigger = _build_trigger(request)

        if trigger:
            scheduler.add_job(
                _execute_agent_job_bg,
                trigger=trigger,
                args=[job_id, request.prompt, session_id, request.user_id],
                id=job_id,
                replace_existing=True,
            )

        if not scheduler.running:
            scheduler.start()

        if request.run_now or not trigger:
            asyncio.create_task(
                _execute_agent_job_bg(job_id, request.prompt, session_id, request.user_id)
            )

        next_run = None
        if trigger:
            job = scheduler.get_job(job_id)
            if job and job.next_run_time:
                next_run = job.next_run_time.isoformat()

        schedule_store[job_id] = {
            "cron_expression": request.cron_expression,
            "prompt": request.prompt,
            "session_id": session_id,
            "user_id": request.user_id,
            "enabled": request.enabled,
        }

        return ScheduleResponse(
            job_id=job_id,
            schedule=request.cron_expression,
            next_run=next_run,
            status="scheduled" if trigger else "running",
        )

    @router.post("/run")
    async def run_now_stream(request: ScheduleRunRequest):
        """Run an agent job immediately and stream SSE back — same
        format as POST /chat/stream so the UI can display results identically.

        UI calls this instead of /chat/stream when it wants a schedule-triggered run.
        """
        session_id = request.session_id or f"run-{uuid.uuid4().hex[:8]}"
        job_id = request.job_id or f"run-{uuid.uuid4().hex[:8]}"
        message = request.prompt

        return StreamingResponse(
            _stream_agent(message, session_id, request.user_id, job_id=job_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )

    @router.get("", response_model=dict)
    async def list_schedules():
        """List all registered schedules."""
        jobs = []
        for job_id, cfg in schedule_store.items():
            job = scheduler.get_job(job_id)
            jobs.append(ScheduleInfo(
                job_id=job_id,
                cron_expression=cfg.get("cron_expression"),
                prompt=cfg["prompt"],
                session_id=cfg["session_id"],
                user_id=cfg["user_id"],
                enabled=cfg["enabled"],
                next_run=job.next_run_time.isoformat() if job and job.next_run_time else None,
                active=job is not None,
            ).model_dump())
        return {"schedules": jobs}

    @router.get("/results/{job_id}")
    async def get_results(job_id: str, limit: int = 10):
        """Return the last *limit* run results for a job.

        Each run includes:
        - run_id, timestamp, status
        - stream_chunks: raw output from the agent stream
        """
        results = result_store.get(job_id, [])
        sanitized = []
        for r in results[-limit:]:
            sanitized.append({
                "run_id": r.get("run_id"),
                "timestamp": r.get("timestamp"),
                "session_id": r.get("session_id"),
                "status": r.get("status"),
                "error": r.get("error"),
            })
        return {"job_id": job_id, "total_runs": len(results), "runs": sanitized}

    @router.get("/results/{job_id}/stream")
    async def get_results_stream(job_id: str, run_index: int = -1):
        """Replay ALL stored stream chunks of a past run as SSE — identical
        to what /chat/stream would have returned.

        *run_index* selects which run (-1 = latest).
        """
        results = result_store.get(job_id, [])
        if not results:
            raise HTTPException(status_code=404, detail="No results for this job")
        try:
            entry = results[run_index]
        except IndexError:
            raise HTTPException(status_code=404, detail=f"Run index {run_index} not found")
        if entry.get("status") == "failed":
            raise HTTPException(status_code=500, detail=entry.get("error", "Unknown error"))

        async def _replay() -> AsyncGenerator[str, None]:
            chunks = entry.get("stream_chunks", [])
            session_id = entry.get("session_id", "")
            for chunk_content in chunks:
                json_data = json.dumps({
                    "content": chunk_content,
                    "session_id": session_id,
                })
                yield f"data: {json_data}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            _replay(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @router.put("/{job_id}/pause")
    async def pause_schedule(job_id: str):
        """Pause a recurring schedule."""
        try:
            scheduler.pause_job(job_id)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return {"job_id": job_id, "status": "paused"}

    @router.put("/{job_id}/resume")
    async def resume_schedule(job_id: str):
        """Resume a paused schedule."""
        try:
            scheduler.resume_job(job_id)
        except Exception:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return {"job_id": job_id, "status": "resumed"}

    @router.delete("/{job_id}")
    async def delete_schedule(job_id: str):
        """Delete a schedule and discard its stored results."""
        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass  # already gone
        schedule_store.pop(job_id, None)
        result_store.pop(job_id, None)
        return {"job_id": job_id, "status": "deleted"}

    @router.get("/results")
    async def list_result_job_ids():
        """List all job_ids currently in result_store (debug endpoint)."""
        return {"job_ids": list(result_store.keys())}

    return router
