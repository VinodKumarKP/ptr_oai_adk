"""Result extraction and response formatting for claude-agent-sdk responses."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional


class ResultExtractor:
    """Formats claude-agent-sdk responses into the standard ADK response dict."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    def extract_text(self, result: Any) -> str:
        """Pull plain text from an orchestration result dict or raw message."""
        if result is None:
            return ""
        if isinstance(result, str):
            return result
        # Our orchestration builder returns {"text": ..., "raw": ...}
        if isinstance(result, dict):
            return result.get("text", "") or self._from_raw(result.get("raw"))
        return self._from_raw(result)

    def _from_raw(self, message: Any) -> str:
        """Extract text from a raw claude-agent-sdk message object."""
        if message is None:
            return ""
        if hasattr(message, "result") and message.result:
            return str(message.result)
        if hasattr(message, "content"):
            content = message.content
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for block in content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    elif isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                return "\n".join(parts)
        return str(message) if message else ""

    def format_response(
        self,
        result: Any,
        session_id: str,
        model_id: str,
        model_provider: str = "anthropic",
        include_raw: bool = False,
        input_message: Optional[str] = None,
        original_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the standard ADK response dict."""
        text = self.extract_text(result)

        response: Dict[str, Any] = {
            "session_id": session_id,
            "content": {"text": text, "type": "AIMessage"},
            "final": True,
            "model": model_id,
            "provider": model_provider,
        }

        if include_raw:
            response["raw"] = result

        if input_message is not None:
            response["input_message"] = input_message

        if original_message is not None:
            response["original_message"] = original_message

        return response

    def format_streaming_chunk(
        self,
        content: str,
        chunk_type: str = "text_delta",
        agent: Optional[str] = None,
        final: bool = False,
    ) -> Dict[str, Any]:
        chunk: Dict[str, Any] = {
            "content": {"text": content, "type": "AIMessage"},
            "chunk_type": chunk_type,
            "final": final,
        }
        if agent:
            chunk["agent"] = agent
        return chunk
