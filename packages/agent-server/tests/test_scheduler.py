import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI, Request
from datetime import datetime, timezone, timedelta
from pydantic import ValidationError

from oai_agent_server.routers.scheduler import create_schedule_router, ScheduleRequest, ScheduleRunRequest
from oai_agent_server.security.dependencies import verify_api_key

# Make sure APScheduler is mocked/available
@pytest.fixture
def mock_scheduler():
    scheduler = MagicMock()
    job = MagicMock()
    job.next_run_time = datetime.now(timezone.utc) + timedelta(hours=1)
    job.pending = False
    scheduler.get_job.return_value = job
    return scheduler

@pytest.fixture
def scheduler_db_logger():
    logger = MagicMock()
    logger.is_active = True
    
    # Mock return values for async methods
    logger.log_scheduled_job = AsyncMock()
    logger.get_all_scheduled_jobs = AsyncMock(return_value=[
        {
            "job_id": "job1",
            "cron_expression": "0 9 * * *",
            "prompt": "test prompt",
            "session_id": "session1",
            "user_id": "user1",
            "enabled": True,
        }
    ])
    logger.get_scheduled_job_runs = AsyncMock(return_value=[
        {
            "run_id": "run1",
            "timestamp": "2026-06-29T12:00:00",
            "session_id": "session1",
            "status": "completed",
            "error": None,
        }
    ])
    logger.get_scheduled_job_run_stream_chunks = AsyncMock(return_value=[
        {"content": "chunk1", "session_id": "session1"},
        {"content": "chunk2", "session_id": "session1"},
    ])
    logger.get_scheduled_job = AsyncMock(return_value={
        "job_id": "job1",
        "cron_expression": "0 9 * * *",
        "run_at": None,
        "prompt": "test prompt",
        "session_id": "session1",
        "user_id": "user1",
        "enabled": True,
    })
    logger.update_scheduled_job = AsyncMock()
    logger.delete_scheduled_job = AsyncMock()
    logger.log_scheduled_job_run = AsyncMock()
    return logger

@pytest.fixture
def scheduler_app(mock_agent, scheduler_db_logger, mock_scheduler):
    with patch("oai_agent_server.routers.scheduler._APSCHEDULER_AVAILABLE", True), \
         patch("oai_agent_server.routers.scheduler.CronTrigger", MagicMock(), create=True), \
         patch("oai_agent_server.routers.scheduler.DateTrigger", MagicMock(), create=True):
        router = create_schedule_router(mock_agent, scheduler_db_logger)
        app = FastAPI()
        app.include_router(router)
        app.state.scheduler = mock_scheduler
        
        # Bypass auth dependency
        app.dependency_overrides[verify_api_key] = lambda: True
        
        yield app

def test_create_schedule_cron(scheduler_app, scheduler_db_logger, mock_scheduler):
    client = TestClient(scheduler_app)
    payload = {
        "job_id": "job1",
        "cron_expression": "0 9 * * *",
        "prompt": "run daily report",
        "session_id": "session1",
        "user_id": "user1",
        "run_now": False,
        "enabled": True
    }
    response = client.post("/schedule", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job1"
    assert data["schedule"] == "0 9 * * *"
    
    scheduler_db_logger.log_scheduled_job.assert_called_once()
    mock_scheduler.add_job.assert_called_once()

def test_create_schedule_run_at(scheduler_app, scheduler_db_logger, mock_scheduler):
    client = TestClient(scheduler_app)
    payload = {
        "job_id": "job2",
        "run_at": "2026-06-30T09:00:00Z",
        "prompt": "one shot run",
        "enabled": True
    }
    response = client.post("/schedule", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job2"
    
    scheduler_db_logger.log_scheduled_job.assert_called_once()
    mock_scheduler.add_job.assert_called_once()

def test_create_schedule_run_now_only(scheduler_app, scheduler_db_logger, mock_scheduler):
    client = TestClient(scheduler_app)
    payload = {
        "job_id": "job3",
        "prompt": "run now task",
        "run_now": True,
        "enabled": True
    }
    # For immediate run with no trigger, APScheduler is not scheduled with add_job
    response = client.post("/schedule", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job3"

def test_invalid_cron_expression():
    with pytest.raises(ValidationError):
        ScheduleRequest(cron_expression="invalid cron", prompt="test")

def test_list_schedules(scheduler_app, scheduler_db_logger):
    client = TestClient(scheduler_app)
    response = client.get("/schedule")
    assert response.status_code == 200
    data = response.json()
    assert "schedules" in data
    assert len(data["schedules"]) == 1
    assert data["schedules"][0]["job_id"] == "job1"

def test_get_results(scheduler_app, scheduler_db_logger):
    client = TestClient(scheduler_app)
    response = client.get("/schedule/results/job1?limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "job1"
    assert len(data["runs"]) == 1
    assert data["runs"][0]["run_id"] == "run1"

def test_get_results_db_inactive(mock_agent, mock_scheduler):
    inactive_logger = MagicMock()
    inactive_logger.is_active = False
    with patch("oai_agent_server.routers.scheduler._APSCHEDULER_AVAILABLE", True):
        router = create_schedule_router(mock_agent, inactive_logger)
        app = FastAPI()
        app.include_router(router)
        app.state.scheduler = mock_scheduler
        app.dependency_overrides[verify_api_key] = lambda: True
        
        client = TestClient(app)
        response = client.get("/schedule/results/job1")
        assert response.status_code == 500

def test_get_results_stream(scheduler_app, scheduler_db_logger):
    client = TestClient(scheduler_app)
    response = client.get("/schedule/results/job1/stream")
    assert response.status_code == 200
    
    # Read stream chunks
    lines = response.content.decode().split("\n\n")
    assert 'data: {"content": {"content": "chunk1", "session_id": "session1"}, "session_id": "session1"}' in lines
    assert "data: [DONE]" in lines

def test_get_results_stream_not_found(scheduler_app, scheduler_db_logger):
    scheduler_db_logger.get_scheduled_job_run_stream_chunks.return_value = []
    client = TestClient(scheduler_app)
    response = client.get("/schedule/results/job1/stream")
    assert response.status_code == 404

def test_pause_schedule(scheduler_app, scheduler_db_logger, mock_scheduler):
    client = TestClient(scheduler_app)
    response = client.put("/schedule/job1/pause")
    assert response.status_code == 200
    assert response.json()["status"] == "paused"
    mock_scheduler.pause_job.assert_called_once_with("job1")

def test_pause_schedule_not_found(scheduler_app, scheduler_db_logger):
    scheduler_db_logger.get_scheduled_job.return_value = None
    client = TestClient(scheduler_app)
    response = client.put("/schedule/job1/pause")
    assert response.status_code == 404

def test_resume_schedule(scheduler_app, scheduler_db_logger, mock_scheduler):
    client = TestClient(scheduler_app)
    response = client.put("/schedule/job1/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "resumed"
    mock_scheduler.resume_job.assert_called_once_with("job1")

def test_delete_schedule(scheduler_app, scheduler_db_logger, mock_scheduler):
    client = TestClient(scheduler_app)
    response = client.delete("/schedule/job1")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"
    mock_scheduler.remove_job.assert_called_once_with("job1")
    scheduler_db_logger.delete_scheduled_job.assert_called_once_with("job1")

def test_list_result_job_ids(scheduler_app, scheduler_db_logger):
    client = TestClient(scheduler_app)
    response = client.get("/schedule/results")
    assert response.status_code == 200
    assert response.json()["job_ids"] == ["job1"]

def test_run_now_stream(scheduler_app):
    client = TestClient(scheduler_app)
    payload = {
        "job_id": "job1",
        "prompt": "test run immediate",
        "session_id": "session1",
        "user_id": "user1"
    }
    response = client.post("/schedule/run", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    lines = response.content.decode().split("\n\n")
    assert "data: [DONE]" in lines

@pytest.mark.asyncio
async def test_execute_agent_job_bg(mock_agent, scheduler_db_logger):
    from oai_agent_server.routers.scheduler import _execute_agent_job_bg
    await _execute_agent_job_bg(mock_agent, scheduler_db_logger, "job1", "test prompt", "session1", "user1")
    scheduler_db_logger.log_scheduled_job_run.assert_called_once()
