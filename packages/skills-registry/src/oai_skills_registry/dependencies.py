"""
FastAPI dependency injection for Skills Registry.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.services.skills_registry import SkillsRegistry

# Global instances
_skills_registry: Optional[SkillsRegistry] = None
_db_logger: Optional[SkillsDatabaseLogger] = None
_logger: Optional[logging.Logger] = None

security = HTTPBearer()


async def initialize_registry(logger: Optional[logging.Logger] = None) -> SkillsRegistry:
    """Initialize the skills registry."""
    global _skills_registry, _db_logger, _logger

    _logger = logger or logging.getLogger(__name__)

    # Initialize database logger
    _db_logger = SkillsDatabaseLogger(logger=_logger)
    initialized = await _db_logger.initialize()

    if not initialized:
        _logger.warning("Database logger not initialized - running in no-persistence mode")

    # Initialize skills registry
    _skills_registry = SkillsRegistry(db_logger=_db_logger, logger=_logger)

    return _skills_registry


def get_registry() -> SkillsRegistry:
    """Get the skills registry instance."""
    if _skills_registry is None:
        raise RuntimeError("Registry not initialized. Call initialize_registry() first.")
    return _skills_registry


async def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> HTTPAuthorizationCredentials:
    """Verify bearer token from Authorization header."""
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials


async def get_auth_user(credentials: HTTPAuthorizationCredentials = Depends(verify_bearer_token)) -> str:
    """Extract username/identifier from token (simplified)."""
    # In production, this would validate the JWT and extract user info
    # For now, we'll just return a placeholder
    return "authenticated_user"


async def close_registry() -> None:
    """Close the skills registry."""
    global _skills_registry
    if _skills_registry:
        await _skills_registry.close()
