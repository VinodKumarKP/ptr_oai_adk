"""Output serialization and session management for agent execution tracking."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List


class OutputSerializer:
    """Handles serialization and storage of agent execution outputs.

    Manages:
    - Output serialization from various object types
    - Session-based output storage
    - Type-safe serialization
    - Output retrieval and formatting

    Attributes:
        session_outputs: Dictionary mapping session IDs to output lists
        logger: Logger instance
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the output serializer.

        Args:
            logger: Optional logger instance
        """
        self.session_outputs: Dict[str, List[Dict[str, Any]]] = {}
        self.logger = logger or logging.getLogger(__name__)

    def serialize(self, obj: Any) -> Dict[str, Any]:
        """Safely serialize an object to a dictionary.

        Handles different object types:
        - Dictionaries (pass through)
        - Objects with __dict__ attribute
        - Other types (converted to string)

        Args:
            obj: Object to serialize

        Returns:
            Dictionary representation of the object

        Example:
            >>> serializer = OutputSerializer()
            >>> result = serializer.serialize({'key': 'value'})
            >>> print(result)
            {'key': 'value'}
        """
        if isinstance(obj, dict):
            return obj

        if hasattr(obj, '__dict__'):
            try:
                return obj.__dict__
            except Exception as e:
                self.logger.warning(f"Failed to serialize __dict__: {e}")
                return {'content': str(obj), '_serialization_fallback': True}

        return {
            'content': str(obj),
            'type': type(obj).__name__,
            '_serialized_from': 'string_conversion'
        }

    def write_to_session(
            self,
            content: Any,
            session_id: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Write content to a session's output storage.

        Args:
            content: Content to store (will be serialized)
            session_id: Session identifier
            metadata: Optional metadata to merge with content
        """
        # Initialize session storage if needed
        if session_id not in self.session_outputs:
            self.session_outputs[session_id] = []

        try:
            # Serialize the content
            serialized = self.serialize(content)

            # Add metadata if provided
            if metadata:
                serialized.update(metadata)

            # Add timestamp if not present
            if 'timestamp' not in serialized:
                serialized['timestamp'] = datetime.utcnow().isoformat()

            # Store the output
            self.session_outputs[session_id].append(serialized)

            self.logger.debug(
                f"Wrote output to session '{session_id}' "
                f"(total: {len(self.session_outputs[session_id])})"
            )

        except Exception as e:
            self.logger.error(
                f"Failed to write output to session '{session_id}': {e}",
                exc_info=True
            )

    def get_session_outputs(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all outputs for a session.

        Args:
            session_id: Session identifier

        Returns:
            List of output dictionaries (empty list if session not found)
        """
        return self.session_outputs.get(session_id, [])

    def clear_session(self, session_id: str) -> bool:
        """Clear all outputs for a session.

        Args:
            session_id: Session identifier

        Returns:
            True if session existed and was cleared, False otherwise
        """
        if session_id in self.session_outputs:
            del self.session_outputs[session_id]
            self.logger.info(f"Cleared session '{session_id}'")
            return True

        self.logger.warning(f"Session '{session_id}' not found")
        return False

    def get_session_count(self, session_id: str) -> int:
        """Get the number of outputs in a session.

        Args:
            session_id: Session identifier

        Returns:
            Number of outputs in the session
        """
        return len(self.session_outputs.get(session_id, []))

    def list_sessions(self) -> List[str]:
        """Get list of all active session IDs.

        Returns:
            List of session ID strings
        """
        return list(self.session_outputs.keys())

    def get_latest_output(
            self,
            session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent output from a session.

        Args:
            session_id: Session identifier

        Returns:
            Latest output dictionary or None if session is empty
        """
        outputs = self.get_session_outputs(session_id)
        return outputs[-1] if outputs else None

    def filter_outputs_by_type(
            self,
            session_id: str,
            output_type: str
    ) -> List[Dict[str, Any]]:
        """Filter session outputs by type.

        Args:
            session_id: Session identifier
            output_type: Type to filter by (e.g., 'text_delta', 'tool_use')

        Returns:
            List of outputs matching the type
        """
        outputs = self.get_session_outputs(session_id)
        return [
            output for output in outputs
            if output.get('type') == output_type
        ]

    def get_text_content(self, session_id: str) -> str:
        """Extract and concatenate all text content from a session.

        Args:
            session_id: Session identifier

        Returns:
            Combined text content from all text_delta outputs
        """
        text_outputs = self.filter_outputs_by_type(session_id, 'text_delta')
        return ''.join(
            output.get('content', '')
            for output in text_outputs
        )

    def get_tool_uses(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all tool use events from a session.

        Args:
            session_id: Session identifier

        Returns:
            List of tool use event dictionaries
        """
        return self.filter_outputs_by_type(session_id, 'tool_use')

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get a summary of session outputs.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics and metadata
        """
        outputs = self.get_session_outputs(session_id)

        if not outputs:
            return {
                'session_id': session_id,
                'exists': False,
                'total_outputs': 0
            }

        # Count by type
        type_counts = {}
        for output in outputs:
            output_type = output.get('type', 'unknown')
            type_counts[output_type] = type_counts.get(output_type, 0) + 1

        # Get time range
        timestamps = [
            output.get('timestamp')
            for output in outputs
            if 'timestamp' in output
        ]

        return {
            'session_id': session_id,
            'exists': True,
            'total_outputs': len(outputs),
            'output_types': type_counts,
            'first_timestamp': min(timestamps) if timestamps else None,
            'last_timestamp': max(timestamps) if timestamps else None,
            'text_length': len(self.get_text_content(session_id)),
            'tool_use_count': len(self.get_tool_uses(session_id))
        }

    def export_session(
            self,
            session_id: str,
            format: str = 'json'
    ) -> Any:
        """Export session outputs in specified format.

        Args:
            session_id: Session identifier
            format: Export format ('json', 'text', 'summary')

        Returns:
            Formatted session data

        Raises:
            ValueError: If format is not supported
        """
        if format == 'json':
            return self.get_session_outputs(session_id)
        elif format == 'text':
            return self.get_text_content(session_id)
        elif format == 'summary':
            return self.get_session_summary(session_id)
        else:
            raise ValueError(
                f"Unsupported export format: {format}. "
                f"Use 'json', 'text', or 'summary'"
            )

    def clear_all_sessions(self) -> int:
        """Clear all session data.

        Returns:
            Number of sessions that were cleared
        """
        count = len(self.session_outputs)
        self.session_outputs.clear()
        self.logger.info(f"Cleared all {count} sessions")
        return count

    def __len__(self) -> int:
        """Get total number of sessions."""
        return len(self.session_outputs)

    def __contains__(self, session_id: str) -> bool:
        """Check if session exists using 'in' operator."""
        return session_id in self.session_outputs

    def __repr__(self) -> str:
        """String representation of the serializer."""
        return (
            f"OutputSerializer("
            f"sessions={len(self.session_outputs)}, "
            f"total_outputs={sum(len(outputs) for outputs in self.session_outputs.values())})"
        )
