"""Streaming JSON encoder for efficient chunked response serialization."""

import json
from typing import Any, Iterator, Union


class StreamingJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that supports streaming/chunked encoding.
    
    Allows encoding large JSON objects without keeping the entire 
    serialized string in memory at once.
    """
    
    def iterencode(self, o: Any, _one_shot: bool = False) -> Iterator[str]:
        """Override iterencode to ensure lazy iteration.
        
        Args:
            o: Object to encode
            _one_shot: Internal parameter used by json module
            
        Yields:
            JSON string chunks
        """
        # Force streaming mode (don't use one_shot)
        return super().iterencode(o, False)
    
    def default(self, o: Any) -> Any:
        """Handle non-standard types with sensible defaults.
        
        Args:
            o: Object to serialize
            
        Returns:
            JSON-serializable representation
        """
        # Try common patterns first
        if hasattr(o, 'model_dump'):
            # Pydantic v2 models
            return o.model_dump()
        if hasattr(o, 'dict'):
            # Pydantic v1 models or other dict-like objects
            return o.dict()
        if hasattr(o, '__dict__'):
            # Regular Python objects
            return o.__dict__
        if hasattr(o, 'isoformat'):
            # Datetime objects
            return o.isoformat()
        
        # Fall back to string representation
        return str(o)


def serialize_chunk_streaming(data: Any) -> Iterator[str]:
    """Serialize data as JSON in streaming chunks.
    
    Useful for server-sent events (SSE) format where each message is
    prefixed with 'data: '. This encoder yields partial JSON instead of
    buffering the entire object in memory.
    
    Args:
        data: Data to serialize
        
    Yields:
        JSON string chunks (does NOT include newline suffixes)
        
    Example:
        for chunk in serialize_chunk_streaming(large_object):
            yield f"data: {chunk}\\n\\n"
    """
    encoder = StreamingJSONEncoder()
    for chunk in encoder.iterencode(data):
        if chunk:  # Skip empty chunks
            yield chunk


def serialize_chunk_safe(data: Any, fallback_to_string: bool = True) -> str:
    """Serialize data to JSON with optional string fallback.
    
    Args:
        data: Data to serialize
        fallback_to_string: If True, use str() as final fallback
        
    Returns:
        JSON string or stringified data
    """
    try:
        return json.dumps(data, cls=StreamingJSONEncoder)
    except (TypeError, ValueError) as e:
        if fallback_to_string:
            return json.dumps({"error": str(data), "original_type": type(data).__name__})
        raise
