"""Admin endpoints for inspecting internal A2A task store state.

All endpoints require strict API key authentication (no localhost / CIDR
bypass) and are intended for operator use only.

Endpoints:

* ``GET /admin/tasks``            — paginated list of task summaries.
* ``GET /admin/tasks/stats``      — aggregate counts.
* ``GET /admin/tasks/{task_id}``  — full task row including ``task_data``.
* ``DELETE /admin/tasks/{task_id}`` — hard-delete a task row.

The router introspects the ``DatabaseTaskStore`` attached to
``app.state.task_store`` and reuses its underlying
:class:`~oai_agent_server.utils.database_logger.DatabaseBackend` for read
queries. If the configured store has no DB backend (e.g. the in-memory
fallback) every endpoint returns ``503``.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from oai_agent_server.security.dependencies import verify_api_key_strict

logger = logging.getLogger(__name__)


_SUMMARY_COLUMNS = (
    "task_id",
    "owner",
    "context_id",
    "status",
    "status_ts",
    "created_at",
    "updated_at",
    "expires_at",
)


def _isoformat(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def _normalize_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for col in _SUMMARY_COLUMNS:
        val = row.get(col)
        if col in {"status_ts", "created_at", "updated_at", "expires_at"}:
            out[col] = _isoformat(val)
        else:
            out[col] = val
    return out


def _parse_task_data(value: Any) -> Any:
    """Return ``task_data`` as a dict regardless of backend storage type."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def _resolve_backend_and_store(request: Request) -> Tuple[Any, Any]:
    """Return ``(task_store, backend)`` or raise ``503`` if unavailable."""
    task_store = getattr(request.app.state, "task_store", None)
    if task_store is None:
        raise HTTPException(
            status_code=503,
            detail="Task store not available on this server.",
        )
    backend = getattr(task_store, "_backend", None)
    if backend is None:
        raise HTTPException(
            status_code=503,
            detail="Admin queries require database task store.",
        )
    return task_store, backend


def _is_postgres(backend: Any) -> bool:
    return getattr(backend, "PLACEHOLDER", "?") == "$"


def _ph(backend: Any, n: int) -> str:
    return f"${n}" if _is_postgres(backend) else "?"


def _now_sql(backend: Any) -> Tuple[str, Tuple[Any, ...]]:
    """Return SQL fragment + params for the current timestamp.

    Postgres uses ``NOW()`` inline (no params); SQLite gets a parameter so
    the value matches the ISO strings the task store writes.
    """
    if _is_postgres(backend):
        return "NOW()", ()
    from datetime import timezone

    return "?", (datetime.now(timezone.utc).isoformat(),)


def create_admin_router() -> APIRouter:
    """Build the admin router. Authentication is enforced on every route."""

    router = APIRouter(
        prefix="/admin",
        tags=["admin"],
        dependencies=[Depends(verify_api_key_strict)],
    )

    @router.get("/tasks")
    async def list_tasks(
        request: Request,
        owner: Optional[str] = Query(default=None),
        status: Optional[str] = Query(default=None),
        context_id: Optional[str] = Query(default=None),
        include_expired: bool = Query(default=False),
        limit: int = Query(default=50, ge=1, le=500),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        """List task summaries (no ``task_data``)."""
        _, backend = _resolve_backend_and_store(request)

        conditions: List[str] = []
        params: List[Any] = []

        def add(cond_template: str, value: Any) -> None:
            conditions.append(cond_template.format(ph=_ph(backend, len(params) + 1)))
            params.append(value)

        if owner is not None:
            add("owner = {ph}", owner)
        if status is not None:
            add("status = {ph}", status)
        if context_id is not None:
            add("context_id = {ph}", context_id)
        if not include_expired:
            now_expr, now_params = _now_sql(backend)
            if now_params:
                # Need to inject placeholder using current index.
                ph = _ph(backend, len(params) + 1)
                conditions.append(f"(expires_at IS NULL OR expires_at > {ph})")
                params.extend(now_params)
            else:
                conditions.append(f"(expires_at IS NULL OR expires_at > {now_expr})")

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""

        cols = ", ".join(_SUMMARY_COLUMNS)
        limit_ph = _ph(backend, len(params) + 1)
        offset_ph = _ph(backend, len(params) + 2)
        order = "ORDER BY updated_at DESC NULLS LAST, task_id DESC" \
            if _is_postgres(backend) else "ORDER BY updated_at DESC, task_id DESC"
        query = (
            f"SELECT {cols} FROM a2a_tasks{where} {order} "
            f"LIMIT {limit_ph} OFFSET {offset_ph}"
        )
        params.extend([limit, offset])

        rows = await backend.fetch(query, tuple(params))
        tasks = [_normalize_summary(r) for r in rows]
        return {
            "tasks": tasks,
            "count": len(tasks),
            "limit": limit,
            "offset": offset,
        }

    @router.get("/tasks/stats")
    async def task_stats(request: Request) -> Dict[str, Any]:
        """Return aggregate counts for the task store."""
        task_store, backend = _resolve_backend_and_store(request)

        now_expr, now_params = _now_sql(backend)
        # Non-expired filter (matches what /tasks shows by default).
        if now_params:
            non_expired = "(expires_at IS NULL OR expires_at > ?)"
        else:
            non_expired = f"(expires_at IS NULL OR expires_at > {now_expr})"

        total_q = f"SELECT COUNT(*) AS c FROM a2a_tasks WHERE {non_expired}"
        by_status_q = (
            f"SELECT status, COUNT(*) AS c FROM a2a_tasks "
            f"WHERE {non_expired} GROUP BY status"
        )
        by_owner_q = (
            f"SELECT owner, COUNT(*) AS c FROM a2a_tasks "
            f"WHERE {non_expired} GROUP BY owner"
        )
        if now_params:
            expired_q = "SELECT COUNT(*) AS c FROM a2a_tasks WHERE expires_at IS NOT NULL AND expires_at < ?"
        else:
            expired_q = (
                f"SELECT COUNT(*) AS c FROM a2a_tasks "
                f"WHERE expires_at IS NOT NULL AND expires_at < {now_expr}"
            )

        total_row = await backend.fetch_one(total_q, now_params)
        status_rows = await backend.fetch(by_status_q, now_params)
        owner_rows = await backend.fetch(by_owner_q, now_params)
        expired_row = await backend.fetch_one(expired_q, now_params)

        return {
            "total": int(total_row["c"]) if total_row else 0,
            "by_status": {
                (r.get("status") or ""): int(r["c"]) for r in status_rows
            },
            "by_owner": {
                (r.get("owner") or ""): int(r["c"]) for r in owner_rows
            },
            "expired_pending_cleanup": int(expired_row["c"]) if expired_row else 0,
            "ttl_seconds": getattr(task_store, "_ttl", None),
        }

    @router.get("/tasks/{task_id}")
    async def get_task(request: Request, task_id: str) -> Dict[str, Any]:
        """Return the full row for *task_id* including ``task_data``."""
        _, backend = _resolve_backend_and_store(request)
        ph = _ph(backend, 1)
        query = (
            "SELECT task_id, owner, context_id, status, status_ts, "
            "task_data, created_at, updated_at, expires_at "
            f"FROM a2a_tasks WHERE task_id = {ph}"
        )
        row = await backend.fetch_one(query, (task_id,))
        if not row:
            raise HTTPException(status_code=404, detail="Task not found")
        out = _normalize_summary(row)
        out["task_data"] = _parse_task_data(row.get("task_data"))
        return out

    @router.delete("/tasks/{task_id}", status_code=204)
    async def delete_task(request: Request, task_id: str) -> Response:
        """Hard-delete *task_id*. Returns 204 on success, 404 if missing."""
        _, backend = _resolve_backend_and_store(request)
        ph = _ph(backend, 1)
        existing = await backend.fetch_one(
            f"SELECT task_id FROM a2a_tasks WHERE task_id = {ph}",
            (task_id,),
        )
        if not existing:
            raise HTTPException(status_code=404, detail="Task not found")
        await backend.execute(
            f"DELETE FROM a2a_tasks WHERE task_id = {ph}", (task_id,)
        )
        return Response(status_code=204)

    return router
