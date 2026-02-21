"""Result extraction and formatting from Strands multi-agent execution."""

from typing import Dict, Any, List

from oai_agent_core.processing.base_result_extractor import BaseResultExtractor

try:
    from strands.multiagent.graph import GraphResult
    from strands.multiagent.swarm import SwarmResult
    from strands.agent.agent_result import AgentResult
except ImportError:
    # Define placeholder types for when strands is not installed
    GraphResult = type('GraphResult', (), {})
    SwarmResult = type('SwarmResult', (), {})
    AgentResult = type('AgentResult', (), {})


class ResultExtractor(BaseResultExtractor):
    """Extracts and formats results from Strands agent executions.

    Handles different result types:
    - GraphResult (from Graph multi-agent systems)
    - SwarmResult (from Swarm multi-agent systems)
    - Single agent results
    - Generic message objects

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
            # Handle GraphResult
            if isinstance(result, GraphResult):
                return self._extract_from_graph_result(result)

            # Handle SwarmResult
            if isinstance(result, SwarmResult):
                return self._extract_from_swarm_result(result)

            if isinstance(result, AgentResult):
                return self._extract_from_agent_result(result)

            # Handle message objects with content attribute
            if hasattr(result, 'message'):
                return self._extract_from_message(result.message)

            # Handle direct message objects
            if hasattr(result, 'content'):
                return self._extract_from_content(result.content)
            
            # Handle dict results (common in some flows)
            if isinstance(result, dict):
                if 'content' in result:
                    return self._extract_from_content(result['content'])
                if 'output' in result:
                    return str(result['output'])
                if 'result' in result:
                    return self.extract_text(result['result'])

            # Fallback to string conversion
            return str(result)
        except Exception as e:
            self.logger.error(f"Error extracting text from result: {e}", exc_info=True)
            return str(result)

    def _safe_get(self, obj: Any, path: str, default: Any = None) -> Any:
        """Safely get nested attribute or dict key.
        
        Args:
            obj: Object to traverse
            path: Dot-separated path (e.g., 'result.message.content')
            default: Value to return if path doesn't exist
            
        Returns:
            Value at path or default
        """
        current = obj
        for part in path.split('.'):
            if current is None:
                return default
                
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return default
        
        return current if current is not None else default

    def _extract_from_agent_result(self, result: AgentResult) -> str:
        """Extract text from AgentResult."""
        try:
            # Try standard path
            content = self._safe_get(result, 'message.content')
            if content:
                return self._extract_from_content(content)
                
            return str(result)
        except Exception as e:
            self.logger.warning(f"Failed to extract from AgentResult: {e}")
            return str(result)

    def _extract_from_graph_result(self, result: GraphResult) -> str:
        """Extract text from GraphResult."""
        try:
            if hasattr(result, 'execution_order') and result.execution_order:
                last_execution = result.execution_order[-1]

                # Try multiple potential paths for the message
                paths = [
                    'result.result.message', # Standard nested result
                    'result.message',        # Direct result
                    'message'                # Direct message
                ]
                
                for path in paths:
                    message = self._safe_get(last_execution, path)
                    if message:
                        return self._extract_from_message(message)

                # Fallback: check if result itself is text-like
                res = self._safe_get(last_execution, 'result')
                if isinstance(res, str):
                    return res

                return str(last_execution)

            return str(result)

        except Exception as e:
            self.logger.warning(f"Failed to extract from GraphResult: {e}")
            return str(result)

    def _extract_from_swarm_result(self, result: SwarmResult) -> str:
        """Extract text from SwarmResult."""
        try:
            if hasattr(result, 'node_history') and result.node_history:
                last_node = result.node_history[-1]

                # Navigate to the executor's messages
                messages = self._safe_get(last_node, 'executor.messages')
                if messages and isinstance(messages, list):
                    last_message = messages[-1]
                    return self._extract_from_content(self._safe_get(last_message, 'content', []))

                return str(last_node)

            return str(result)

        except Exception as e:
            self.logger.warning(f"Failed to extract from SwarmResult: {e}")
            return str(result)

    def _extract_from_message(self, message: Any) -> str:
        """Extract text from a message object."""
        if hasattr(message, 'content'):
            return self._extract_from_content(message.content)
        if isinstance(message, dict) and 'content' in message:
            return self._extract_from_content(message['content'])

        return str(message)

    def extract_token_usage(self, result: Any) -> Dict[str, Any]:
        """Extract token usage metrics from result."""
        usage = {}

        # Try accumulated_usage attribute (common in multi-agent results)
        if hasattr(result, 'accumulated_usage'):
            usage = result.accumulated_usage

        # Try metrics attribute
        elif hasattr(result, 'metrics'):
            if hasattr(result.metrics, 'accumulated_usage'):
                usage = result.metrics.accumulated_usage

        # Try usage attribute (common in single agent results)
        elif hasattr(result, 'usage'):
            usage = result.usage
            
        # Try nested usage in message metadata
        elif hasattr(result, 'message') and hasattr(result.message, 'usage'):
             usage = result.message.usage

        return usage if isinstance(usage, dict) else {}

    def format_response(
            self,
            result: Any,
            session_id: str,
            model_id: str,
            model_provider: str = 'aws-strands',
            include_raw: bool = False,
            input_message: str = None,
            original_message: str = None,
            final: bool = True
    ) -> Dict[str, Any]:
        """Format result into standardized response structure.
        
        Overrides base method to provide default model_provider and enforce final=True.
        """
        return super().format_response(
            result=result,
            session_id=session_id,
            model_id=model_id,
            model_provider=model_provider,
            include_raw=include_raw,
            input_message=input_message,
            original_message=original_message,
            final=True
        )

    def extract_execution_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract execution metadata from result."""
        metadata = super().extract_execution_metadata(result)

        # GraphResult metadata
        if isinstance(result, GraphResult):
            if hasattr(result, 'execution_order'):
                metadata['execution_steps'] = len(result.execution_order)
                metadata['execution_path'] = [
                    getattr(step, 'node_id', 'unknown')
                    for step in result.execution_order
                ]

        # SwarmResult metadata
        if isinstance(result, SwarmResult):
            if hasattr(result, 'node_history'):
                metadata['node_count'] = len(result.node_history)
                metadata['agents_involved'] = [
                    getattr(node, 'node_id', 'unknown')
                    for node in result.node_history
                ]

            if hasattr(result, 'results'):
                metadata['result_count'] = len(result.results)

        return metadata

    def extract_results_list(self, result: Any) -> List[Any]:
        """Extract list of individual results from multi-agent result."""
        results_list = []

        # GraphResult
        if isinstance(result, GraphResult) and hasattr(result, 'execution_order'):
            results_list = [step.result for step in result.execution_order if hasattr(step, 'result')]

        # SwarmResult
        elif isinstance(result, SwarmResult) and hasattr(result, 'results'):
            results_list = result.results

        return results_list

    def __repr__(self) -> str:
        """String representation of the extractor."""
        return "ResultExtractor()"
