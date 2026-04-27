from typing import Optional, List

from fastapi import APIRouter, Depends, Query

from oai_agent_server.security.dependencies import verify_api_key


def create_logs_router(logging_service, allowed_modes: Optional[List[str]] = None):
    """Create the logs router with configured endpoints."""
    router = APIRouter(tags=["logs"], dependencies=[Depends(verify_api_key)])

    if allowed_modes is None:
        allowed_modes = ["logs"]

    if "logs" in allowed_modes:
        @router.get("/logs")
        async def get_logs(
                session_id: Optional[str] = Query(None),
                user_id: Optional[str] = Query(None),
                endpoint: Optional[str] = Query(None),
                status: Optional[str] = Query(None),
                limit: int = Query(100, ge=1, le=1000),
                offset: int = Query(0, ge=0)
        ):
            """Retrieve logs with optional filtering."""
            return await logging_service.get_logs(session_id, user_id, endpoint, status, limit, offset)

        @router.get("/logs/interaction/{interaction_id}")
        async def get_chat_log_by_interaction_id(interaction_id: str):
            """Retrieve a chat log for a specific interaction."""
            return await logging_service.get_chat_log_by_interaction_id(interaction_id)

        @router.get("/logs/activity/interaction/{interaction_id}")
        async def get_activity_logs_by_interaction_id(interaction_id: str):
            """Retrieve activity logs for a specific interaction."""
            return await logging_service.get_activity_logs_by_interaction_id(interaction_id)

        @router.get("/logs/sessions/{session_id}")
        async def get_session_logs(session_id: str):
            """Retrieve logs for a specific session."""
            return await logging_service.get_session_logs(session_id)

        @router.get("/logs/stats")
        async def get_log_stats(user_id: Optional[str] = Query(None)):
            """Retrieve log statistics."""
            return await logging_service.get_log_stats(user_id=user_id)

        @router.get("/logs/stats/users")
        async def get_user_stats():
            """Retrieve user statistics."""
            return await logging_service.get_user_stats()

    if "monitoring" in allowed_modes:
        @router.get("/evaluations/agent")
        async def get_evaluations_by_agent_name():
            """Retrieve all evaluations for this agent."""
            return await logging_service.get_evaluations_by_agent_name()

        @router.get("/evaluations/session/{session_id}")
        async def get_evaluations_by_session_id(session_id: str):
            """Retrieve all evaluations for a specific session."""
            return await logging_service.get_evaluations_by_session_id(session_id)

        @router.get("/evaluations/{interaction_id}")
        async def get_evaluation(interaction_id: str):
            """Retrieve an evaluation for a specific interaction."""
            return await logging_service.get_evaluation(interaction_id)

    return router
