"""Tests for routers/admin.py — admin endpoints over the A2A task store."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from oai_agent_server.routers.admin import create_admin_router
from oai_agent_server.security.dependencies import verify_api_key_strict


def _iso(dt):
    return dt.isoformat()


class FakeBackend:
    """Minimal in-memory backend with the subset of fetch/execute needed by admin."""
    PLACEHOLDER = "?"

    def __init__(self, rows=None):
        self.rows = rows or []
        self.executed = []

    async def fetch(self, query, params):
        q = query.upper()
        # Stats: GROUP BY status / owner
        if "GROUP BY STATUS" in q:
            counts = {}
            for r in self.rows:
                counts[r.get("status") or ""] = counts.get(r.get("status") or "", 0) + 1
            return [{"status": k, "c": v} for k, v in counts.items()]
        if "GROUP BY OWNER" in q:
            counts = {}
            for r in self.rows:
                counts[r.get("owner") or ""] = counts.get(r.get("owner") or "", 0) + 1
            return [{"owner": k, "c": v} for k, v in counts.items()]

        rows = list(self.rows)
        # Owner filter
        if "OWNER = ?" in q:
            owner = params[0]
            rows = [r for r in rows if r.get("owner") == owner]
        if "STATUS = ?" in q:
            # Find status param position
            for p in params:
                if isinstance(p, str) and p in {"working", "completed", "failed", "submitted"}:
                    rows = [r for r in rows if r.get("status") == p]
                    break
        # Limit/offset
        if "LIMIT" in q and "OFFSET" in q:
            limit = params[-2]
            offset = params[-1]
            rows = rows[offset:offset + limit]
        return rows

    async def fetch_one(self, query, params):
        q = query.upper()
        if "COUNT(*)" in q:
            count = len(self.rows)
            if "EXPIRES_AT IS NOT NULL AND EXPIRES_AT <" in q:
                # expired count
                now_iso = params[0] if params else None
                if now_iso:
                    count = sum(1 for r in self.rows
                                if r.get("expires_at") and r["expires_at"] < now_iso)
                else:
                    count = 0
            return {"c": count}
        if "FROM A2A_TASKS WHERE TASK_ID" in q:
            tid = params[0]
            for r in self.rows:
                if r.get("task_id") == tid:
                    return r
            return None
        return None

    async def execute(self, query, params):
        self.executed.append((query, params))
        if query.strip().upper().startswith("DELETE FROM A2A_TASKS WHERE TASK_ID"):
            tid = params[0]
            self.rows = [r for r in self.rows if r.get("task_id") != tid]


def _row(task_id="t1", owner="alice", status="working", expires_at=None, task_data=None):
    if expires_at is None:
        expires_at = _iso(datetime.now(timezone.utc) + timedelta(hours=1))
    return {
        "task_id": task_id,
        "owner": owner,
        "context_id": f"ctx-{task_id}",
        "status": status,
        "status_ts": _iso(datetime.now(timezone.utc)),
        "task_data": task_data or json.dumps({"id": task_id}),
        "created_at": _iso(datetime.now(timezone.utc)),
        "updated_at": _iso(datetime.now(timezone.utc)),
        "expires_at": expires_at,
    }


@pytest.fixture
def app_with_admin():
    backend = FakeBackend(rows=[
        _row("t1", "alice", "working"),
        _row("t2", "bob", "completed"),
        _row("t3", "alice", "failed",
             expires_at=_iso(datetime.now(timezone.utc) - timedelta(hours=1))),
    ])
    task_store = MagicMock()
    task_store._backend = backend
    task_store._ttl = 86400

    app = FastAPI()
    app.state.task_store = task_store
    app.include_router(create_admin_router())
    app.dependency_overrides[verify_api_key_strict] = lambda: True
    return app, backend, task_store


@pytest.fixture
def app_no_backend():
    app = FastAPI()
    task_store = MagicMock()
    task_store._backend = None
    app.state.task_store = task_store
    app.include_router(create_admin_router())
    app.dependency_overrides[verify_api_key_strict] = lambda: True
    return app


class TestAuth:
    def test_list_tasks_requires_auth(self):
        # No dependency override — should 401/403
        app = FastAPI()
        backend = FakeBackend()
        task_store = MagicMock()
        task_store._backend = backend
        app.state.task_store = task_store
        app.include_router(create_admin_router())
        client = TestClient(app)
        r = client.get("/admin/tasks")
        assert r.status_code in (401, 403)


class TestListTasks:
    def test_returns_list(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.get("/admin/tasks")
        assert r.status_code == 200
        body = r.json()
        assert "tasks" in body
        assert isinstance(body["tasks"], list)

    def test_filter_by_owner(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.get("/admin/tasks?owner=alice")
        assert r.status_code == 200
        # alice has 2 rows but t3 is expired (excluded by default)
        tasks = r.json()["tasks"]
        assert all(t["owner"] == "alice" for t in tasks)

    def test_filter_by_status(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.get("/admin/tasks?status=working")
        assert r.status_code == 200
        tasks = r.json()["tasks"]
        assert all(t["status"] == "working" for t in tasks)

    def test_pagination(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.get("/admin/tasks?limit=1&offset=0")
        assert r.status_code == 200
        assert r.json()["limit"] == 1
        assert r.json()["offset"] == 0


class TestGetTask:
    def test_get_task_returns_full_row(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.get("/admin/tasks/t1")
        assert r.status_code == 200
        body = r.json()
        assert body["task_id"] == "t1"
        # task_data should be parsed as dict
        assert isinstance(body["task_data"], dict)
        assert body["task_data"]["id"] == "t1"

    def test_get_task_404(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.get("/admin/tasks/nope")
        assert r.status_code == 404


class TestStats:
    def test_stats_returns_aggregate(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.get("/admin/tasks/stats")
        assert r.status_code == 200
        body = r.json()
        assert "total" in body
        assert "by_status" in body
        assert "by_owner" in body
        assert body["ttl_seconds"] == 86400


class TestDelete:
    def test_delete_returns_204(self, app_with_admin):
        app, backend, _ = app_with_admin
        client = TestClient(app)
        r = client.delete("/admin/tasks/t1")
        assert r.status_code == 204
        assert all(row["task_id"] != "t1" for row in backend.rows)

    def test_delete_missing_404(self, app_with_admin):
        app, _, _ = app_with_admin
        client = TestClient(app)
        r = client.delete("/admin/tasks/nope")
        assert r.status_code == 404


class TestNoBackend:
    def test_list_returns_503(self, app_no_backend):
        client = TestClient(app_no_backend)
        r = client.get("/admin/tasks")
        assert r.status_code == 503

    def test_get_returns_503(self, app_no_backend):
        client = TestClient(app_no_backend)
        r = client.get("/admin/tasks/x")
        assert r.status_code == 503

    def test_stats_returns_503(self, app_no_backend):
        client = TestClient(app_no_backend)
        r = client.get("/admin/tasks/stats")
        assert r.status_code == 503

    def test_delete_returns_503(self, app_no_backend):
        client = TestClient(app_no_backend)
        r = client.delete("/admin/tasks/x")
        assert r.status_code == 503
