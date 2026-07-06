"""Result extraction and formatting from LangGraph agent execution."""

import logging
from typing import Dict, Any, Optional, Union, List

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from pydantic import BaseModel

from oai_agent_core.processing.base_result_extractor import BaseResultExtractor


class ResultExtractor(BaseResultExtractor):
    """Extracts and formats results from LangGraph agent executions.

    Supports multiple stream modes and subgraph configurations:
    - stream_mode="values": Full state at each step
    - stream_mode="messages": New messages at each step
    - subgraphs=True/False: Include messages from subgraphs

    Attributes:
        logger: Logger instance
    """

    def extract_messages(
        self,
        result: Any,
        stream_mode: str = "values",
        subgraphs: bool = False
    ) -> Optional[List[Union[BaseMessage, Dict]]]:
        """Extract all messages from result based on stream mode and subgraph settings.

        Handles various result formats from LangGraph streaming:

        stream_mode="values":
        - Dict: {"messages": [...]}
        - Tuple (with subgraphs): ((), {"messages": [...]})

        stream_mode="messages":
        - Single message: AIMessage(...)
        - Tuple (no subgraphs): (AIMessage(...), metadata_dict)
        - Nested tuple (with subgraphs): (("node_name",), (AIMessage(...), metadata_dict))

        Args:
            result: The chunk/result from LangGraph streaming
            stream_mode: Stream mode ("values" or "messages")
            subgraphs: Whether subgraph messages are included

        Returns:
            List of messages or None if no messages found
        """
        messages = []

        if stream_mode == "values":
            # In values mode, result is typically a dict with state keys including 'messages'
            if hasattr(result, 'get') and 'messages' in result:
                messages = result['messages']
            # Handle tuple format (with subgraphs=True): ((), {"messages": [...]})
            elif isinstance(result, tuple) and len(result) == 2:
                first_elem, second_elem = result
                # First element is usually () for values mode
                if isinstance(second_elem, dict) and 'messages' in second_elem:
                    messages = second_elem['messages']

        elif stream_mode == "messages":
            # Single message object (BaseMessage or message-like object with content)
            if isinstance(result, BaseMessage) or (hasattr(result, 'content') and not isinstance(result, (dict, tuple))):
                messages = [result]
            # Tuple format: (AIMessage, metadata_dict) or (("node",), (AIMessage, metadata)) or ((), (AIMessage, metadata))
            elif isinstance(result, tuple):
                if len(result) == 2:
                    first_elem, second_elem = result
                    # Check if first element is a tuple (indicates nested format)
                    if isinstance(first_elem, tuple):
                        # Nested format variants:
                        # Non-deep: (("node_name",), (AIMessage, metadata))
                        # Deep: ((), (AIMessage, metadata))
                        if isinstance(second_elem, tuple) and len(second_elem) >= 1:
                            msg = second_elem[0]
                            if isinstance(msg, BaseMessage) or hasattr(msg, 'content'):
                                messages = [msg]
                    else:
                        # Simple format: (AIMessage, metadata)
                        if isinstance(first_elem, BaseMessage) or hasattr(first_elem, 'content'):
                            messages = [first_elem]
                        elif isinstance(second_elem, BaseMessage) or hasattr(second_elem, 'content'):
                            messages = [second_elem]
            # Dict containing messages
            elif hasattr(result, 'get'):
                if 'messages' in result:
                    messages = result['messages']
                elif 'message' in result:
                    msg = result['message']
                    messages = [msg] if msg else []
                # Try to extract from node updates
                else:
                    for key, value in result.items():
                        if isinstance(value, dict) and 'messages' in value:
                            messages.extend(value['messages'])

        return messages if messages else None

    def extract_last_message(
        self,
        result: Any,
        stream_mode: str = "values",
        subgraphs: bool = False
    ) -> Optional[Union[BaseMessage, Dict]]:
        """Extract the last message from the result.

        Args:
            result: The chunk/result from LangGraph streaming
            stream_mode: Stream mode ("values" or "messages")
            subgraphs: Whether subgraph messages are included

        Returns:
            The last message or None
        """
        messages = self.extract_messages(result, stream_mode, subgraphs)
        if messages:
            return messages[-1]

        # Fallback to original behavior for compatibility
        if hasattr(result, 'get') and 'messages' in result:
            messages_list = result['messages']
            if messages_list:
                return messages_list[-1]
        if isinstance(result, tuple):
            if len(result) == 1:
                return result[0]
            elif len(result) == 2:
                if isinstance(result[0], AIMessage):
                    return result[0]
                elif isinstance(result[0], HumanMessage):
                    return result[0]
                elif len(result[1]) == 2:
                    return result[1][0]
                else:
                    return result[1]

        return None

    def extract_text(
        self,
        result: Any,
        stream_mode: str = "values",
        subgraphs: bool = False
    ) -> Union[str, list]:
        """Extract text content from result.

        Args:
            result: The chunk/result from LangGraph streaming
            stream_mode: Stream mode ("values" or "messages")
            subgraphs: Whether subgraph messages are included

        Returns:
            Text content as string or list
        """
        last_message = self.extract_last_message(result, stream_mode, subgraphs)
        if last_message:
            if hasattr(last_message, 'content'):
                return last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                return last_message['content']

        return ""

    def extract_token_usage(
        self,
        result: Any,
        stream_mode: str = "values",
        subgraphs: bool = False
    ) -> Dict[str, Any]:
        """Extract token usage metrics from result.

        Args:
            result: The chunk/result from LangGraph streaming
            stream_mode: Stream mode ("values" or "messages")
            subgraphs: Whether subgraph messages are included

        Returns:
            Dictionary with token usage metrics
        """
        last_message = self.extract_last_message(result, stream_mode, subgraphs)
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
            include_raw: bool = False,
            input_message: str = None,
            original_message: str = None,
            final: bool = True,
            stream_mode: str = "values",
            subgraphs: bool = False
    ) -> Dict[str, Any]:
        """Format response into standardized response structure.

        Provides a sensible default for ``model_provider`` ('langchain') so
        callers do not need to supply it explicitly.

        Args:
            result: The result from agent execution
            session_id: Session identifier
            model_id: Model identifier
            model_provider: Model provider name
            include_raw: Whether to include raw response
            input_message: Original input message
            original_message: Original message before processing
            final: Whether this is the final response
            stream_mode: Stream mode ("values" or "messages")
            subgraphs: Whether subgraph messages are included
        """
        formatted = {}

        if isinstance(result, dict) and 'structured_response' in result:
            if isinstance(result['structured_response'], BaseModel):
                content = result['structured_response'].model_dump()
            else:
                content = str(result['structured_response'])
        else:
            content = self.extract_text(result, stream_mode, subgraphs)

        formatted['content'] = {
            "text": content,
            "type": "text",
            "final": final,
            "session_id": session_id
        }

        formatted["token_usage"] = self.extract_token_usage(result, stream_mode, subgraphs)

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
            original_message: str = None,
            stream_mode: str = "values",
            subgraphs: bool = False
    ) -> Dict[str, Any]:
        """Format a streaming chunk into standardized LangGraph/LangChain format.

        Supports different stream modes and subgraph configurations:
        - stream_mode="values": Full state chunks
        - stream_mode="messages": Message-only chunks
        - subgraphs=True: Includes messages from subagents

        Args:
            chunk: The chunk from LangGraph streaming
            session_id: Session identifier
            model_id: Model identifier
            model_provider: Model provider name
            is_final: Whether this is the final chunk
            include_raw: Whether to include raw chunk
            input_message: Original input message
            original_message: Original message before processing
            stream_mode: Stream mode ("values" or "messages")
            subgraphs: Whether subgraph messages are included
        """
        formatted = {}

        last_message = self.extract_last_message(chunk, stream_mode, subgraphs)
        if last_message:
            content = ''
            if hasattr(last_message, 'content'):
                content = last_message.content
            elif isinstance(last_message, dict) and 'content' in last_message:
                content = last_message['content']

            if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
                formatted['tool_calls'] = last_message.tool_calls
                if isinstance(content, str) and content.strip() == '':
                    content = 'Executing tools'

            if hasattr(last_message, 'response_metadata') and last_message.response_metadata.get('finish_reason', None) == 'stop':
                is_final = True

            formatted['content'] = {
                "text": content,
                "type": type(last_message).__name__,
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
