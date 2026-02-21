from typing import Any


def make_serializable(obj: Any) -> Any:
    """
    Convert non-serializable objects to JSON-serializable format.

    Args:
        obj: Object to make serializable

    Returns:
        JSON-serializable version of the object
    """
    # Handle None
    if obj is None:
        return None

    # Handle primitives (already serializable)
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle lists
    if isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]

    # Handle dictionaries
    if isinstance(obj, dict):
        return {str(key): make_serializable(value) for key, value in obj.items()}

    # Handle objects with __dict__
    if hasattr(obj, '__dict__'):
        return {str(key): make_serializable(value) for key, value in obj.__dict__.items()}

    # Handle objects with dict() method
    if hasattr(obj, 'dict') and callable(obj.dict):
        try:
            return obj.dict()
        except Exception:
            pass

    # Handle objects with model_dump() method (Pydantic v2)
    if hasattr(obj, 'model_dump') and callable(obj.model_dump):
        try:
            return obj.model_dump()
        except Exception:
            pass

    # Handle datetime objects
    if hasattr(obj, 'isoformat') and callable(obj.isoformat):
        try:
            return obj.isoformat()
        except Exception:
            pass

    # Handle bytes
    if isinstance(obj, bytes):
        try:
            return obj.decode('utf-8')
        except Exception:
            return str(obj)

    # Last resort: convert to string
    return str(obj)
