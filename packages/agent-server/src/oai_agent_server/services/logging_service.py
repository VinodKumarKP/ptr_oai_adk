from typing import Optional

from fastapi.responses import JSONResponse
from fastapi import HTTPException

from oai_agent_server.exceptions import DatabaseNotAvailableException


class LoggingService:
    """Service for retrieving logs and statistics."""

    def __init__(self, db_logger, agent_name):
        self.db_logger = db_logger
        self.agent_name = agent_name

    async def get_logs(
            self,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            endpoint: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 100,
            offset: int = 0
    ):
        """Retrieve logs with optional filtering."""
        if not self.db_logger.is_active:
            await self.db_logger.initialize()
            if not self.db_logger.is_active:
                raise DatabaseNotAvailableException(operation="get_logs")

        logs = await self.db_logger.get_logs(
            agent_name=self.agent_name,
            session_id=session_id,
            user_id=user_id,
            endpoint=endpoint,
            status=status,
            limit=limit,
            offset=offset
        )

        return JSONResponse(content={
            "total": len(logs),
            "limit": limit,
            "offset": offset,
            "logs": logs
        })

    async def get_chat_log_by_interaction_id(self, interaction_id: str):
        """Retrieve a chat log for a specific interaction."""
        if not self.db_logger.is_active:
            await self.db_logger.initialize()
            if not self.db_logger.is_active:
                raise DatabaseNotAvailableException(operation="get_chat_log_by_interaction_id")

        log = await self.db_logger.get_chat_log_by_interaction_id(interaction_id)

        if log is None:
            raise HTTPException(status_code=404, detail="Chat log not found for the given interaction ID.")

        return JSONResponse(content=log)

    async def get_activity_logs_by_interaction_id(self, interaction_id: str):
        """Retrieve activity logs for a specific interaction."""
        if not self.db_logger.is_active:
            await self.db_logger.initialize()
            if not self.db_logger.is_active:
                raise DatabaseNotAvailableException(operation="get_activity_logs_by_interaction_id")

        logs = await self.db_logger.get_activity_logs(interaction_id=interaction_id)

        return JSONResponse(content={
            "interaction_id": interaction_id,
            "total": len(logs),
            "logs": logs
        })

    async def get_evaluation(self, interaction_id: str):
        """Retrieve evaluation for a specific interaction."""
        if not self.db_logger.is_active:
            await self.db_logger.initialize()
            if not self.db_logger.is_active:
                raise DatabaseNotAvailableException(operation="get_evaluation")

        evaluation = await self.db_logger.get_evaluation_by_interaction_id(interaction_id)

        if evaluation is None:
            raise HTTPException(status_code=404, detail="Evaluation not found for the given interaction ID.")

        return JSONResponse(content=evaluation)

    async def get_session_logs(self, session_id: str):
        """Retrieve logs for a specific session."""
        if not self.db_logger.is_active:
            await self.db_logger.initialize()
            if not self.db_logger.is_active:
                raise DatabaseNotAvailableException(operation="get_session_logs")

        logs = await self.db_logger.get_activity_logs(
            agent_name=self.agent_name,
            session_id=session_id
        )

        return JSONResponse(content={
            "session_id": session_id,
            "total": len(logs),
            "logs": logs
        })

    async def get_log_stats(self, user_id: Optional[str] = None):
        """Retrieve log statistics."""
        if not self.db_logger.is_active:
            await self.db_logger.initialize()
            if not self.db_logger.is_active:
                raise DatabaseNotAvailableException(operation="get_log_stats")

        stats = await self.db_logger.get_stats(agent_name=self.agent_name, user_id=user_id)

        return JSONResponse(content={
            "agent_name": self.agent_name,
            "database_active": self.db_logger.is_active,
            "statistics": stats
        })

    async def get_user_stats(self):
        """Retrieve user statistics."""
        if not self.db_logger.is_active:
            await self.db_logger.initialize()
            if not self.db_logger.is_active:
                raise DatabaseNotAvailableException(operation="get_user_stats")
        
        if hasattr(self.db_logger, 'get_user_stats'):
             stats = await self.db_logger.get_user_stats(agent_name=self.agent_name)
        else:
             stats = {}

        return JSONResponse(content={
            "agent_name": self.agent_name,
            "database_active": self.db_logger.is_active,
            "user_statistics": stats
        })
