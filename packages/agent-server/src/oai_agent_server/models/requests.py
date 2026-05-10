import json
import os
from typing import Optional, Union

from pydantic import BaseModel, field_validator


def _get_max_message_size() -> int:
    """Read max message size from env each call so tests/operators can override."""
    try:
        return int(os.environ.get("MAX_MESSAGE_SIZE_BYTES", "32768"))
    except (TypeError, ValueError):
        return 32768


def _validate_message_size(v):
    """Reject payloads above MAX_MESSAGE_SIZE_BYTES (default 32768)."""
    limit = _get_max_message_size()
    if isinstance(v, str):
        size = len(v.encode("utf-8"))
    elif isinstance(v, dict):
        try:
            size = len(json.dumps(v).encode("utf-8"))
        except (TypeError, ValueError):
            size = len(str(v).encode("utf-8"))
    else:
        return v
    if size > limit:
        raise ValueError(
            f"message exceeds maximum allowed size ({size} > {limit} bytes)"
        )
    return v


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
                v = json.loads(v)
            except Exception:
                pass
        return v

    @field_validator("message")
    @classmethod
    def check_message_size(cls, v):
        return _validate_message_size(v)


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
                v = json.loads(v)
            except Exception:
                pass
        return v

    @field_validator("message")
    @classmethod
    def check_message_size(cls, v):
        return _validate_message_size(v)
