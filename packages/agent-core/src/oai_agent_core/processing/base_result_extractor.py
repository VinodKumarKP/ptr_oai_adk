"""Base result extractor for agent core."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class BaseResultExtractor(ABC):
    """Base class for extracting and formatting results from agent executions.

    Provides a standardized output format across different agent frameworks
    (OpenAI, LangGraph, AWS Strands).
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the result extractor.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    @abstractmethod
    def extract_text(self, result: Any) -> str:
        """Extract text content from any result type.

        Args:
            result: Result object from agent execution

        Returns:
            Extracted text content
        """
        pass

    @abstractmethod
    def extract_token_usage(self, result: Any) -> Dict[str, Any]:
        """Extract token usage metrics from result.

        Args:
            result: Result object from agent execution

        Returns:
            Dictionary containing token usage (input_tokens, output_tokens, total_tokens)
        """
        pass

    def extract_execution_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract execution metadata from result.

        Args:
            result: Result object from agent execution

        Returns:
            Dictionary containing metadata
        """
        return {
            'result_type': type(result).__name__
        }

    def format_response(
            self,
            result: Any,
            session_id: str,
            model_id: str,
            model_provider: str,
            include_raw: bool = False,
            input_message: Optional[str] = None,
            original_message: Optional[str] = None,
            final: bool = True
    ) -> Dict[str, Any]:
        """Format result into standardized response structure.

        Args:
            result: Raw execution result
            session_id: Session identifier
            model_id: Model identifier
            model_provider: Provider name (openai, bedrock, etc.)
            include_raw: Whether to include raw result object
            input_message: Original input message text
            original_message: Full original message object/string
            final: Whether this is the final response

        Returns:
            Standardized response dictionary
        """
        text_content = self.extract_text(result)
        token_usage = self.extract_token_usage(result)

        response = {
            'content': {
                'text': text_content,
                'type': result.get('type', 'text') if isinstance(result, dict) else 'text',
                'final': final,
                'session_id': str(session_id)
            },
            'model': {
                'model_id': model_id,
                'model_provider': model_provider
            }
        }

        # Add token usage if available
        if token_usage:
            response['token_usage'] = token_usage

        # Add raw result if requested
        if include_raw:
            response['raw_result'] = self._serialize_result(result)

        if input_message:
            response['input_message'] = input_message

        if original_message:
            response['original_message'] = original_message

        if 'tool_calls' in result:
            response['tool_calls'] = result['tool_calls']

        # Add execution metadata
        response['metadata'] = self.extract_execution_metadata(result)

        return response

    def format_streaming_chunk(
            self,
            content: Any,
            chunk_type: str = 'text',
            agent: Optional[str] = None,
            final: bool = False,
            **kwargs
    ) -> Dict[str, Any]:
        """Format a streaming chunk for consistent output."""
        chunk = {
            'content': content,
            'type': chunk_type,
            'final': final
        }

        if agent:
            chunk['agent'] = agent

        # Add any additional fields
        chunk.update(kwargs)

        return chunk

    def _extract_from_content(self, content: Any) -> str:
        """Extract text from content attribute (list or dict or str)."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # Extract text from all blocks that have a 'text' attribute
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    if 'text' in block:
                        text_parts.append(block['text'])
                elif hasattr(block, 'text'):
                    text_parts.append(block.text)
                elif isinstance(block, str):
                    text_parts.append(block)

            return ''.join(text_parts) if text_parts else str(content)

        if isinstance(content, dict) and 'text' in content:
            return content['text']

        return str(content)

    def _serialize_result(self, result: Any) -> Any:
        """Recursively serialize result object for storage."""
        if result is None:
            return None

        if isinstance(result, (str, int, float, bool)):
            return result

        if isinstance(result, dict):
            return {k: self._serialize_result(v) for k, v in result.items()}

        if isinstance(result, (list, tuple)):
            return [self._serialize_result(item) for item in result]

        if hasattr(result, '__dict__'):
            return {
                k: self._serialize_result(v)
                for k, v in result.__dict__.items()
                if not k.startswith('_')  # Skip private attributes
            }

        return str(result)

    def __repr__(self) -> str:
        """String representation of the extractor."""
        return f"{self.__class__.__name__}()"