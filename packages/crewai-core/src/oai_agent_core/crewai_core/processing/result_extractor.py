"""Result Extractor for CrewAI execution outputs.

This module handles extraction, formatting, and streaming of CrewAI
execution results.
"""

from typing import Dict, Any, AsyncGenerator, List

from oai_agent_core.processing.base_result_extractor import BaseResultExtractor


class ResultExtractor(BaseResultExtractor):
    """Extracts and formats results from CrewAI executions.

    This class is responsible for:
    - Formatting execution results into standardized responses
    - Extracting session outputs
    - Streaming results incrementally
    - Handling token usage and model metadata

    Attributes:
        output_serializer: Serializer for session outputs
        llm: Language model instance
        logger: Logger instance
    """

    def __init__(self,
                 output_serializer: Any,
                 llm: Any,
                 logger: Any):
        """Initialize the result extractor.

        Args:
            output_serializer: Output serializer instance
            llm: Language model instance
            logger: Logger instance
        """
        super().__init__(logger=logger)
        self.output_serializer = output_serializer
        self.llm = llm

    def get_response(self,
                     session_id: str,
                     final_result: Any = None) -> Dict[str, Any]:
        """Extract and format the final response from session output.

        Args:
            session_id: Session identifier
            final_result: CrewAI execution result with token usage

        Returns:
            Formatted response dictionary containing:
            - content: List with final output
            - token_usage: Token consumption metrics
            - model: Model identifier and provider
        """
        session_output = self.output_serializer.get_session_outputs(session_id)
        response = {}

        # Process session output
        for idx, item in enumerate(session_output):
            is_last = idx == len(session_output) - 1

            if is_last:
                response = {
                    'result': item.get('text') or item.get('result'),
                    'raw': final_result
                }
            else:
                continue

        if len(session_output) == 0 and final_result:
            response = {'result': final_result.raw,
                        'raw': final_result}

        return response

    def get_streaming_response(self,
                               session_id: str,
                               final_result: Any = None) -> List[Dict[str, Any]]:
        """Get all session outputs for streaming.

        Args:
            session_id: Session identifier
            final_result: CrewAI execution result

        Returns:
            List of formatted output chunks
        """
        session_output = self.output_serializer.get_session_outputs(session_id)
        chunks = []

        for item in session_output:
            chunks.append({
                'result': item.get('text') or item.get('result'),
                'raw': final_result
            })

        return chunks

    async def stream_response(self,
                              session_id: str,
                              final_result: Any = None) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream session outputs asynchronously.

        Args:
            session_id: Session identifier
            final_result: CrewAI execution result

        Yields:
            Formatted output chunks
        """
        chunks = self.get_streaming_response(session_id, final_result)

        for chunk in chunks:
            yield chunk

        # Clean up session after streaming
        self.output_serializer.clear_session(session_id)

    def extract_token_usage(self, result: Any) -> Dict[str, Any]:
        """Extract token usage from execution result.

        Args:
            result: CrewAI execution result

        Returns:
            Dictionary with token usage metrics
        """
        # Handle case where result is a dict with 'raw' key (from format_response)
        if isinstance(result, dict) and 'raw' in result:
            final_result = result['raw']
        else:
            final_result = result

        if final_result and hasattr(final_result, 'token_usage'):
            return final_result.token_usage.__dict__
        return {}

    def _extract_model_info(self) -> Dict[str, str]:
        """Extract model information.

        Returns:
            Dictionary with model ID and provider
        """
        return {
            'model_id': self.llm.model,
            'model_provider': getattr(self.llm, 'provider', 'unknown')
        }

    def format_execution_result(self,
                                session_id: str,
                                result: Any,
                                final: bool = True,
                                input_message: str = None) -> Dict[str, Any]:
        """Format an execution result into standardized response.

        Args:
            session_id: Session identifier
            result: CrewAI execution result
            final: Whether this is the final result

        Returns:
            Formatted response dictionary
        """
        return {
            "session_id": session_id,
            "result": result,
            "final": final,
            "input_message": input_message
        }

    def extract_error_details(self,
                              error: Exception,
                              inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and format error details.

        Args:
            error: Exception that occurred
            inputs: Input data that caused the error

        Returns:
            Dictionary with error information
        """
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'inputs_provided': inputs
        }

        # Add specific guidance for common errors
        if "No task outputs available" in str(error):
            error_info['suggestions'] = [
                "Tasks not completing successfully",
                "Missing required inputs in task descriptions",
                "Invalid task context references"
            ]

        return error_info

    def get_summary(self, session_id: str) -> Dict[str, Any]:
        """Get summary of session execution.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with execution summary
        """
        session_output = self.output_serializer.get_session_outputs(session_id)

        return {
            'session_id': str(session_id),
            'step_count': len(session_output),
            'has_output': len(session_output) > 0
        }

    def clear_session(self, session_id: str):
        """Clear session data.

        Args:
            session_id: Session identifier to clear
        """
        self.output_serializer.clear_session(session_id)

    def format_response(
            self,
            result: Any,
            session_id: str,
            model_id: str,
            model_provider: str = 'crewai',
            include_raw: bool = False,
            input_message: str = None,
            original_message: str = None,
            final: bool = True
    ) -> Dict[str, Any]:
        """Format result into standardized response structure.

        Overrides base to supply a sensible default for ``model_provider``.
        """
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

    def extract_text(self, result: Any) -> str:
        """Extract text from result."""
        if isinstance(result, dict):
            if 'result' in result:
                return str(result['result'])
            if 'text' in result:
                return str(result['text'])

        if hasattr(result, 'raw'):
            return str(result.raw)

        return str(result)
