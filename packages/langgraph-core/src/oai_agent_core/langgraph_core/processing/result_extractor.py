"""Result extraction and formatting from LangGraph agent execution."""

import logging
from typing import Dict, Any, Optional, Union

from langchain_core.messages import BaseMessage
from oai_agent_core.processing.base_result_extractor import BaseResultExtractor


class ResultExtractor(BaseResultExtractor):
    """Extracts and formats results from LangGraph agent executions.

    Attributes:
        logger: Logger instance
    """

    def extract_last_message(self, result: Any) -> Optional[Union[BaseMessage, Dict]]:
        """Extract the last message from the result."""
        if hasattr(result, 'get') and 'messages' in result:
            messages = result['messages']
            if messages:
                return messages[-1]
        return None

    def extract_text(self, result: Any) -> Union[str, list]:
        """Extract text content from result."""
        last_message = self.extract_last_message(result)
        if last_message:
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                return last_message['content']
        
        return ""

    def extract_token_usage(self, result: Any) -> Dict[str, Any]:
        """Extract token usage metrics from result."""
        last_message = self.extract_last_message(result)
        if last_message:
            if hasattr(last_message, 'usage_metadata'):
                return last_message.usage_metadata
            elif isinstance(last_message, dict) and 'usage_metadata' in last_message:
                return last_message['usage_metadata']
        return {}

    def format_response(
            self,
            result: Any,
            session_id: str,
            model_id: str,
            model_provider: str = 'langchain',
            include_raw: bool = True,
            input_message: str = None,
            original_message: str = None,
            final: bool = True
    ) -> Dict[str, Any]:
        """Format response to match AWS Strands agent output format."""
        formatted = {}

        content = self.extract_text(result)

        formatted['content'] = {
            "text": content,
            "type": "text",
            "final": final,
            "session_id": session_id
        }

        formatted["token_usage"] = self.extract_token_usage(result)

        formatted["model"] = {
            "model_id": model_id,
            "model_provider": model_provider
        }

        if input_message:
            formatted["input_message"] = input_message

        if include_raw:
            formatted["raw_response"] = result

        if original_message:
            formatted["original_message"] = original_message

        return formatted

    def format_stream_chunk(
            self,
            chunk: Any,
            session_id: str,
            model_id: str,
            model_provider: str = 'langchain',
            is_final: bool = False,
            include_raw: bool = False,
            input_message: str = None,
            original_message: str = None
    ) -> Dict[str, Any]:
        """Format streaming chunk to match AWS Strands format."""
        formatted = {}

        last_message = self.extract_last_message(chunk)
        if last_message:
            content=''
            if hasattr(last_message, 'content'):
                content = last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                content = last_message['content']

            formatted['content'] = {
                "text": content,
                "type": "text",
                "final": is_final,
                "session_id": session_id
            }
            
            # Optimization: Extract token usage directly from last_message if available
            # This avoids calling extract_last_message again inside extract_token_usage
            if hasattr(last_message, 'usage_metadata'):
                formatted["token_usage"] = last_message.usage_metadata
            elif isinstance(last_message, dict) and 'usage_metadata' in last_message:
                formatted["token_usage"] = last_message['usage_metadata']
            else:
                formatted["token_usage"] = {}
        else:
            formatted["content"] = {
                "text": str(chunk),
                "type": "text",
                "final": is_final,
                "session_id": session_id
            }
            formatted["token_usage"] = {}

        formatted["model"] = {
            "model_id": model_id,
            "model_provider": model_provider
        }

        if input_message:
            formatted["input_message"] = input_message

        if include_raw:
            formatted["raw_response"] = chunk

        if original_message:
            formatted["original_message"] = original_message

        return formatted
