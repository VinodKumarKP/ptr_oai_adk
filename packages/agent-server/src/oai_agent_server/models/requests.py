import json
import os
import re
from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator


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


def _validate_identifier(v: Optional[str], name: str) -> Optional[str]:
    """
    Validate identifier format (UUID or alphanumeric with underscores/hyphens).
    
    Args:
        v: Value to validate
        name: Field name for error messages
        
    Returns:
        Validated value
        
    Raises:
        ValueError: If format is invalid
    """
    if v is None:
        return v
    
    # UUID format: 8-4-4-4-12 hex digits
    uuid_pattern = r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
    # Custom ID format: alphanumeric, underscore, hyphen, 1-100 chars
    custom_pattern = r"^[a-zA-Z0-9_-]{1,100}$"
    
    if not (re.match(uuid_pattern, v, re.IGNORECASE) or re.match(custom_pattern, v)):
        raise ValueError(
            f"{name} must be a valid UUID or alphanumeric identifier (1-100 chars, "
            f"allowing only letters, numbers, underscores, and hyphens)"
        )
    return v


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: Union[str, dict]
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for grouping conversations"
    )
    user_id: Optional[str] = Field(
        "user",
        min_length=1,
        max_length=100,
        description="User identifier, defaults to 'user'"
    )

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
    
    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        """Ensure message is not empty or whitespace-only."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Message cannot be empty or whitespace-only")
        elif isinstance(v, dict):
            if not v:
                raise ValueError("Message dict cannot be empty")
        return v
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        return _validate_identifier(v, "session_id")
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id format."""
        if v is None:
            return v
        # Allow alphanumeric, underscore, hyphen, dot; 1-100 chars
        if not re.match(r"^[a-zA-Z0-9._-]{1,100}$", v):
            raise ValueError(
                "user_id must be 1-100 characters with only letters, numbers, "
                "underscores, hyphens, or dots"
            )
        return v


class StreamChatRequest(BaseModel):
    """Request model for streaming chat endpoint."""
    message: Union[str, dict]
    session_id: Optional[str] = Field(
        None,
        description="Optional session ID for grouping conversations"
    )
    user_id: Optional[str] = Field(
        "user",
        min_length=1,
        max_length=100,
        description="User identifier, defaults to 'user'"
    )
    verbose: Optional[bool] = Field(
        False,
        description="Include verbose debugging information in response"
    )

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
    
    @field_validator("message")
    @classmethod
    def message_not_empty(cls, v):
        """Ensure message is not empty or whitespace-only."""
        if isinstance(v, str):
            if not v.strip():
                raise ValueError("Message cannot be empty or whitespace-only")
        elif isinstance(v, dict):
            if not v:
                raise ValueError("Message dict cannot be empty")
        return v
    
    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, v):
        return _validate_identifier(v, "session_id")
    
    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v):
        """Validate user_id format."""
        if v is None:
            return v
        # Allow alphanumeric, underscore, hyphen, dot; 1-100 chars
        if not re.match(r"^[a-zA-Z0-9._-]{1,100}$", v):
            raise ValueError(
                "user_id must be 1-100 characters with only letters, numbers, "
                "underscores, hyphens, or dots"
            )
        return v
