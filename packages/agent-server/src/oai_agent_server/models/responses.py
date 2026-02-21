from pydantic import BaseModel


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    session_id: str
