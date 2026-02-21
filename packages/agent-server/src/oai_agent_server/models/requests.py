import json
from typing import Optional, Union

from pydantic import BaseModel, field_validator


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: Union[str, dict]
    session_id: Optional[str] = None
    user_id: Optional[str] = "user"

    @field_validator("message", mode="before")
    @classmethod
    def parse_message(cls, v):
        """Parse message field if it's a JSON string."""
        # If input is a string that looks like a dict, try to parse it
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return v
        return v


class StreamChatRequest(BaseModel):
    """Request model for streaming chat endpoint."""
    message: Union[str, dict]
    session_id: Optional[str] = None
    user_id: Optional[str] = "user"
    verbose: Optional[bool] = False

    @field_validator("message", mode="before")
    @classmethod
    def parse_message(cls, v):
        """Parse message field if it's a JSON string."""
        # If input is a string that looks like a dict, try to parse it
        if isinstance(v, str):
            try:
                return json.loads(v)
            except Exception:
                return v
        return v
