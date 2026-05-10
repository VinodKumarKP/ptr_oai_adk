"""Scheduled job management using APScheduler + FastAPI.

Provides REST endpoints for the UI to create, list, pause, resume, and delete
scheduled jobs. Each job calls the same agent.astream() path used by
/chat/stream, so results are identical SSE format.
"""

import asyncio
import inspect
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import AsyncGenerator, List, Optional

try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False
    Limiter = None
    get_remote_address = None

_schedule_rate_limit = os.environ.get("RATE_LIMIT_SCHEDULE", "30/minute")
if _SLOWAPI_AVAILABLE:
    _limiter = Limiter(key_func=get_remote_address)
else:
    _limiter = None


def _maybe_limit(rate: str):
    if _limiter is None:
        def _noop(func):
            return func
        return _noop
    return _limiter.limit(rate)

try:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
    from apscheduler.triggers.cron import CronTrigger
    from apscheduler.triggers.date import DateTrigger
    _APSCHEDULER_AVAILABLE = True
except ImportError:
    _APSCHEDULER_AVAILABLE = False
    
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from oai_agent_core.core.base_agent import BaseAgent
from oai_agent_server.utils.response_extractor import ResponseContentExtractor
from oai_agent_server.utils.serialization import make_serializable
from oai_agent_server.utils.database_logger import DatabaseLogger
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


if _APSCHEDULER_AVAILABLE:
    scheduler = AsyncIOScheduler()
else:
    scheduler = None


# ---- helper: stream agent response as SSE (same format as /chat/stream) ----
async def _stream_agent(agent: BaseAgent, db_logger: DatabaseLogger, message: str, session_id: str,
                        user_id: str, job_id: Optional[str] = None) -> AsyncGenerator[str, None]:
    """Yield SSE events identical to /chat/stream.

    When *job_id* is given the collected chunks are stored in ``result_store``.
    """
    response_extractor = ResponseContentExtractor(agent)
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
    if job_id and db_logger.is_active:
        await db_logger.log_scheduled_job_run(
            job_id=job_id,
            run_id=uuid.uuid4().hex[:8],
            timestamp=datetime.now(timezone.utc),
            session_id=session_id,
            status="completed",
            error_message=None,
            stream_chunks=collected_chunks,
        )

    yield "data: [DONE]\n\n"

# ---- helper: background execution (stores full result for later retrieval) ----
async def _execute_agent_job_bg(agent: BaseAgent, db_logger: DatabaseLogger, job_id: str, prompt: str,
                              session_id: str, user_id: str):
    message = prompt
    run_id = uuid.uuid4().hex[:8]
    logger.info("[Scheduler] job=%s run=%s session=%s running: %s",
                job_id, run_id, session_id, message)
    
    status = "completed"
    error_message = None
    stream_chunks: List[dict] = []

    try:
        response_extractor = ResponseContentExtractor(agent)
        config = {"session_id": session_id, "user_id": user_id}

        # Stream to capture ALL chunks (tool results, events, final text)
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

        logger.info("[Scheduler] job=%s run=%s completed", job_id, run_id)
    except Exception as exc:
        logger.error("[Scheduler] job=%s run=%s failed: %s",
                     job_id, run_id, exc, exc_info=True)
        status = "failed"
        error_message = str(exc)
    finally:
        if db_logger.is_active:
            await db_logger.log_scheduled_job_run(
                job_id=job_id,
                run_id=run_id,
                timestamp=datetime.now(timezone.utc),
                session_id=session_id,
                status=status,
                error_message=error_message,
                stream_chunks=stream_chunks,
            )


# ---- helper: build the right trigger ----
def _build_trigger(req: ScheduleRequest):
    if not _APSCHEDULER_AVAILABLE:
        return None
    if req.cron_expression:
        parts = req.cron_expression.strip().split()
        return CronTrigger(
            minute=parts[0], hour=parts[1],
            day=parts[2], month=parts[3],
            day_of_week=parts[4],
        )
    if req.run_at:
        # APScheduler expects datetime objects for DateTrigger
        return DateTrigger(run_date=datetime.fromisoformat(req.run_at.replace('Z', '+00:00')))
    return None


# ---------------------------------------------------------------------------
# Router factory
# ---------------------------------------------------------------------------

def create_schedule_router(agent: BaseAgent, db_logger: DatabaseLogger, allowed_modes: Optional[List[str]] = None) -> Optional[APIRouter]:
    """Build & return the /schedule router bound to *agent*."""

    if allowed_modes is None:
        allowed_modes = ["schedule"]

    if "schedule" not in allowed_modes:
        return None

    if not _APSCHEDULER_AVAILABLE:
        logger.warning("APScheduler is not installed. Schedule endpoints will not be available. Install it using `pip install apscheduler`.")
        return None

    router = APIRouter(prefix="/schedule", tags=["schedule"])

    # ---- endpoints ----

    @router.post("", response_model=ScheduleResponse)
    @_maybe_limit(_schedule_rate_limit)
    async def create_schedule(request: Request, schedule_request: ScheduleRequest):
        """Create a new scheduled agent job.

        Example UI payloads::

            Recurring:  {"cron_expression": "0 9 * * 1-5", "prompt": "daily report"}
            One-time:   {"run_at": "2026-04-05T09:00:00Z", "prompt": "monthly summary"}
            Immediate:  {"prompt": "quick task", "run_now": true}
        """
        job_id = schedule_request.job_id or uuid.uuid4().hex[:8]
        session_id = schedule_request.session_id or f"schedule-{job_id}"
        trigger = _build_trigger(schedule_request)

        # Log/update job in DB
        if db_logger.is_active:
            await db_logger.log_scheduled_job(
                job_id=job_id,
                agent_name=agent.agent_name,
                cron_expression=schedule_request.cron_expression,
                run_at=datetime.fromisoformat(schedule_request.run_at.replace('Z', '+00:00')) if schedule_request.run_at else None,
                prompt=schedule_request.prompt,
                session_id=session_id,
                user_id=schedule_request.user_id,
                enabled=schedule_request.enabled,
            )

        if trigger:
            scheduler.add_job(
                _execute_agent_job_bg,
                trigger=trigger,
                args=[agent, db_logger, job_id, schedule_request.prompt, session_id, schedule_request.user_id],
                id=job_id,
                replace_existing=True,
            )
            # If job was paused, resume it
            if not schedule_request.enabled:
                scheduler.pause_job(job_id)


        if not scheduler.running:
            scheduler.start()

        if schedule_request.run_now or (not trigger and schedule_request.enabled):
            asyncio.create_task(
                _execute_agent_job_bg(agent, db_logger, job_id, schedule_request.prompt, session_id, schedule_request.user_id)
            )

        next_run = None
        if trigger:
            job = scheduler.get_job(job_id)
            if job and job.next_run_time:
                next_run = job.next_run_time.isoformat()

        return ScheduleResponse(
            job_id=job_id,
            schedule=schedule_request.cron_expression,
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
            _stream_agent(agent, db_logger, message, session_id, request.user_id, job_id=job_id),
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
        if db_logger.is_active:
            db_jobs = await db_logger.get_all_scheduled_jobs()
            for db_job in db_jobs:
                job_id = db_job["job_id"]
                aps_job = scheduler.get_job(job_id)
                
                next_run = aps_job.next_run_time.isoformat() if aps_job and aps_job.next_run_time else None
                active = aps_job is not None and not aps_job.pending # APScheduler jobs are "active" if not removed and not paused

                jobs.append(ScheduleInfo(
                    job_id=job_id,
                    cron_expression=db_job.get("cron_expression"),
                    prompt=db_job["prompt"],
                    session_id=db_job["session_id"],
                    user_id=db_job["user_id"],
                    enabled=db_job["enabled"],
                    next_run=next_run,
                    active=active,
                ).model_dump())
        return {"schedules": jobs}

    @router.get("/results/{job_id}")
    async def get_results(job_id: str, limit: int = 10):
        """Return the last *limit* run results for a job.

        Each run includes:
        - run_id, timestamp, status
        - stream_chunks: raw output from the agent stream
        """
        if not db_logger.is_active:
            raise HTTPException(status_code=500, detail="Database logging not active.")
        
        results = await db_logger.get_scheduled_job_runs(job_id, limit)
        sanitized = []
        for r in results:
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
        if not db_logger.is_active:
            raise HTTPException(status_code=500, detail="Database logging not active.")

        stream_chunks = await db_logger.get_scheduled_job_run_stream_chunks(job_id, run_index)
        
        if not stream_chunks:
            raise HTTPException(status_code=404, detail="No stream chunks found for this run.")

        async def _replay() -> AsyncGenerator[str, None]:
            # Assuming stream_chunks is a list of dicts, each dict is a chunk content
            # And each chunk content already contains 'session_id' if it was logged that way
            for chunk_content in stream_chunks:
                json_data = json.dumps({
                    "content": chunk_content,
                    "session_id": chunk_content.get("session_id", ""), # Ensure session_id is present
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
        if not db_logger.is_active:
            raise HTTPException(status_code=500, detail="Database logging not active.")
        
        job_cfg = await db_logger.get_scheduled_job(job_id)
        if not job_cfg:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        
        try:
            scheduler.pause_job(job_id)
            await db_logger.update_scheduled_job(
                job_id=job_id,
                cron_expression=job_cfg["cron_expression"],
                run_at=job_cfg["run_at"],
                prompt=job_cfg["prompt"],
                session_id=job_cfg["session_id"],
                user_id=job_cfg["user_id"],
                enabled=False,
            )
        except Exception:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found in scheduler")
        return {"job_id": job_id, "status": "paused"}

    @router.put("/{job_id}/resume")
    async def resume_schedule(job_id: str):
        """Resume a paused schedule."""
        if not db_logger.is_active:
            raise HTTPException(status_code=500, detail="Database logging not active.")

        job_cfg = await db_logger.get_scheduled_job(job_id)
        if not job_cfg:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        try:
            scheduler.resume_job(job_id)
            await db_logger.update_scheduled_job(
                job_id=job_id,
                cron_expression=job_cfg["cron_expression"],
                run_at=job_cfg["run_at"],
                prompt=job_cfg["prompt"],
                session_id=job_cfg["session_id"],
                user_id=job_cfg["user_id"],
                enabled=True,
            )
        except Exception:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found in scheduler")
        return {"job_id": job_id, "status": "resumed"}

    @router.delete("/{job_id}")
    async def delete_schedule(job_id: str):
        """Delete a schedule and discard its stored results."""
        if not db_logger.is_active:
            raise HTTPException(status_code=500, detail="Database logging not active.")

        try:
            scheduler.remove_job(job_id)
        except Exception:
            pass  # already gone from scheduler

        await db_logger.delete_scheduled_job(job_id) # This also cascades to delete runs

        return {"job_id": job_id, "status": "deleted"}

    @router.get("/results")
    async def list_result_job_ids():
        """List all job_ids currently in result_store (debug endpoint)."""
        if not db_logger.is_active:
            raise HTTPException(status_code=500, detail="Database logging not active.")
        
        # For simplicity, return all job_ids that exist in scheduled_jobs table
        # A more precise implementation might query scheduled_job_runs for distinct job_ids
        all_jobs = await db_logger.get_all_scheduled_jobs()
        job_ids_with_results = [job["job_id"] for job in all_jobs]
        return {"job_ids": job_ids_with_results}

    return router
