"""Result extraction and formatting from OpenAI agent execution."""

import logging
from typing import Dict, Any, Optional, List, Union

from oai_agent_core.processing.base_result_extractor import BaseResultExtractor


class ResultExtractor(BaseResultExtractor):
    """Extracts and formats results from OpenAI agent executions.

    Handles different result types:
    - RunResult (from Runner)
    - Dict results
    - String results
    - Message objects

    Attributes:
        logger: Logger instance
    """

    def extract_text(self, result: Any) -> str:
        """Extract text content from any result type.

        Args:
            result: Result object from agent execution

        Returns:
            Extracted text content
        """
        try:
            # Handle RunResult (common in OpenAI agent)
            if hasattr(result, 'final_output'):
                return str(result.final_output)

            # Handle dict results
            if isinstance(result, dict):
                if 'output' in result:
                    return str(result['output'])
                if 'content' in result:
                    return self._extract_from_content(result['content'])
                if 'result' in result:
                    return self.extract_text(result['result'])

            # Handle message objects with content attribute
            if hasattr(result, 'content'):
                return self._extract_from_content(result.content)

            # Fallback to string conversion
            return str(result)
        except Exception as e:
            self.logger.error(f"Error extracting text from result: {e}", exc_info=True)
            return str(result)

    def extract_token_usage(self, result: Any) -> Dict[str, Any]:
        """Extract token usage metrics from result."""
        usage = {}

        # Try usage attribute
        if hasattr(result, 'context_wrapper'):
            usage = result.context_wrapper.usage if hasattr(result.context_wrapper, 'usage') else {}
            usage = {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
                "total_tokens": usage.total_tokens
            }
        elif hasattr(result, 'token_usage'):
            usage = result.token_usage
        elif isinstance(result, dict) and 'usage' in result:
            usage = result['usage']

        return usage if isinstance(usage, dict) else {}

    def format_response(
            self,
            result: Any,
            session_id: str,
            model_id: str,
            model_provider: str = 'openai',
            include_raw: bool = False,
            input_message: str = None,
            original_message: str = None,
            final: bool = True
    ) -> Dict[str, Any]:
        """Format result into standardized response structure."""
        return super().format_response(
            result=result,
            session_id=session_id,
            model_id=model_id,
            model_provider=model_provider,
            include_raw=include_raw,
            input_message=input_message,
            original_message=original_message,
            final=final
        )

    def extract_execution_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract execution metadata from result."""
        metadata = super().extract_execution_metadata(result)
        if isinstance(result, dict) and 'event_type' in result:
            metadata['event_type'] = result['event_type']
        return metadata
