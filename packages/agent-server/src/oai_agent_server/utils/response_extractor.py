from typing import Any, Dict, Optional, Tuple


class ResponseContentExtractor:
    """Extracts content from various response types"""

    def __init__(self, agent):
        self.agent = agent

    def extract_content(self, response: Any) -> Any:
        """Extract content from various response types"""
        default_model = self._get_default_model()

        # Handle LangChain AIMessage objects
        if hasattr(response, 'content'):
            return self._build_result(
                content=response.content,
                token_usage=getattr(response, 'usage_metadata', None),
                model=default_model
            )

        # Handle dictionary responses
        if isinstance(response, dict):
            content, token_usage_from_msg = self._extract_dict_content(response)
            # Priority: token_usage from last_message > response dict
            token_usage = token_usage_from_msg or response.get('usage_metadata') or response.get('token_usage')
            model = response.get('model', default_model)

            if content:
                return self._build_result(content, token_usage, model)

        # Handle list responses
        if isinstance(response, list) and response:
            last_item = response[-1]
            content = self._get_content_from_item(last_item)
            token_usage = self._get_token_usage_from_item(last_item)
            model = self._get_model_from_item(last_item, default_model)

            if content:
                return self._build_result(content, token_usage, model)

        # Fallback to string representation
        return str(response)

    def _get_default_model(self) -> Dict[str, str]:
        """Get default model configuration"""
        model_id = 'unknown'
        if hasattr(self.agent, 'llm') and hasattr(self.agent.llm, 'model_id'):
            model_id = self.agent.llm.model_id
        return {
            'model_id': model_id,
            'provider': self.agent.agent_config.get('cloud_provider', 'unknown')
        }

    def _extract_dict_content(self, response: Dict) -> Tuple[Optional[Any], Optional[Any]]:
        """Extract content and token_usage from dictionary response

        Returns:
            tuple: (content, token_usage) where token_usage may be None
        """
        if 'messages' in response and response['messages']:
            last_message = response['messages'][-1]
            if isinstance(last_message, dict) and 'content' in last_message:
                return last_message['content'], None
            elif hasattr(last_message, 'content'):
                # Extract token_usage from last_message object (LangChain case)
                token_usage = getattr(last_message, 'usage_metadata', None)
                return last_message.content, token_usage

        # Check other content fields
        for key in ('content', 'text', 'output'):
            if key in response:
                return response[key], None

        return None, None

    def _get_content_from_item(self, item: Any) -> Optional[Any]:
        """Get content from an item (dict or object)"""
        if hasattr(item, 'content'):
            return item.content
        elif isinstance(item, dict) and 'content' in item:
            return item['content']
        return None

    def _get_token_usage_from_item(self, item: Any) -> Optional[Any]:
        """Get token usage from an item (dict or object)"""
        if hasattr(item, 'usage_metadata'):
            return item.usage_metadata
        elif isinstance(item, dict):
            return item.get('usage_metadata') or item.get('token_usage')
        return None

    def _get_model_from_item(self, item: Any, default_model: Dict) -> Dict:
        """Get model from an item or return default"""
        if isinstance(item, dict) and 'model' in item:
            return item['model']
        return default_model

    def _build_result(self, content: Any, token_usage: Any, model: Any) -> Any:
        """Build result dict or return content string based on metadata presence"""
        result = {'content': content}

        if token_usage is not None:
            result['token_usage'] = token_usage

        # Always add model (matches original behavior)
        result['model'] = model

        # Return dict if we have metadata, otherwise just the content
        return result if ('token_usage' in result or 'model' in result) else content


def extract_output_text(content: Any) -> str:
    """Extract text representation from content object"""

    output_response = ""
    if isinstance(content, dict):
        if content.get('content'):
            content_data = content['content']
            if isinstance(content_data, list) and content_data and isinstance(content_data[-1], dict):
                output_response = content_data[-1].get('text', '')
            elif isinstance(content_data, dict) and content_data:
                output_response = content_data.get('text', '')
            elif isinstance(content_data, str):
                output_response = content_data
        elif content.get('text'):
            output_response = content['text']
    elif isinstance(content, str):
        output_response = content
    return output_response


def extract_chunk_text(content: Any) -> Optional[str]:
    """Extract text from a streaming chunk content"""
    if isinstance(content, dict) and 'content' in content:
        if isinstance(content['content'], list):
            for item in content['content']:
                if isinstance(item, dict) and 'text' in item:
                    return item['text']
        elif isinstance(content['content'], str):
            return content['content']
    elif isinstance(content, str):
        return content
    return None
