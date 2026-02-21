import hashlib
import logging
import os
from abc import ABC
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

from oai_agent_core.components.vector_store.vector_store_factory import VectorStoreFactory


class BaseMemoryStore(ABC):

    def __init__(self, memory_config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None,
                 project_root: Optional[str] = None, llm: Any = None,
                 vector_store: Any = None):
        """Initialize the factory.

        Args:
            memory_config: List of knowledge base configurations.
            logger: Optional logger instance.
            project_root: Optional project root path for resolving relative paths.
            llm: Optional LLM instance for query analysis.
            vector_store: Optional vector store instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.project_root = project_root
        self.vector_store = vector_store
        self._initialize_vector_store(memory_config)
        
        settings = memory_config.get('settings', {})
        self.max_recent_turns: int = settings.get('max_recent_turns', 5)
        self.max_relevant_turns: int = settings.get('max_relevant_turns', 3)
        self.similarity_threshold: float = settings.get('similarity_threshold', 0.6)

    def _initialize_vector_store(self, memory_config):
        """Initialize the vector store from configuration.

        Args:
            memory_config: Memory configuration dictionary.
        """
        # Vector Store Configuration
        vector_store_config = memory_config.get('vector_store', {})
        vector_store_type = vector_store_config.get('type', 'chroma')
        vector_store_settings = vector_store_config.get('settings', {})
        
        # Embedding Configuration
        embedding_config = memory_config.get('embedding', {})
        embedding_model_id = embedding_config.get('model_id', 'amazon.titan-embed-text-v1')
        region_name = embedding_config.get('region_name', 'us-west-2')
        
        # Resolve persist directory if needed
        persist_directory = vector_store_settings.get('persist_directory')
        if persist_directory:
            if not os.path.isabs(persist_directory) and self.project_root:
                persist_directory = os.path.join(self.project_root, persist_directory)
            else:
                persist_directory = os.path.abspath(persist_directory)
            vector_store_settings['persist_directory'] = persist_directory

        # Create Embeddings
        embeddings = self._create_embeddings(embedding_model_id, region_name)
        if not embeddings:
            self.logger.error("Could not create embeddings for memory store")
            return

        # Create Vector Store if not provided
        if self.vector_store is None:
            try:
                # Ensure collection name is set for memory
                if 'collection_name' not in vector_store_settings:
                    vector_store_settings['collection_name'] = 'chat_memory'
                    
                self.vector_store = VectorStoreFactory.create_vector_store(
                    vector_store_type,
                    embedding_function=embeddings,
                    **vector_store_settings
                )
            except Exception as e:
                self.logger.error(f"Failed to create vector store for memory: {e}")
                raise

    def _create_embeddings(self, model_id: str, region_name: str = "us-west-2"):
        """Create embeddings model using LiteLLM.

        Args:
            model_id: Model ID for embeddings.
            region_name: AWS region (if applicable).

        Returns:
            Embeddings instance or None if creation fails.
        """
        try:
            from litellm import embedding
            from langchain_core.embeddings import Embeddings

            class LiteLLMEmbeddings(Embeddings):
                def __init__(self, model_id):
                    self.model_id = model_id

                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    response = embedding(model=self.model_id, input=texts)
                    return [r['embedding'] for r in response['data']]

                def embed_query(self, text: str) -> List[float]:
                    response = embedding(model=self.model_id, input=[text])
                    return response['data'][0]['embedding']

            # Map common model names to LiteLLM format if needed
            # For Bedrock, LiteLLM usually expects "bedrock/model-id"
            if not model_id.startswith("bedrock/") and "amazon" in model_id:
                litellm_model_id = f"bedrock/{model_id}"
            else:
                litellm_model_id = model_id

            return LiteLLMEmbeddings(model_id=litellm_model_id)

        except ImportError:
            self.logger.warning("Could not import litellm. Please install litellm.")
            return None
        except Exception as e:
            self.logger.error(f"Error creating LiteLLM embeddings: {e}")
            return None

    def _create_turn_id(self, session_id: str, user_id: str, timestamp: str) -> str:
        """Create unique ID for a conversation turn."""
        unique_str = f"{session_id}:{user_id}:{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()

    def add_turn(
            self,
            session_id: str,
            user_id: str,
            user_message: str,
            agent_response: str,
            metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a conversation turn to the history.

        Args:
            session_id: Session identifier
            user_id: User identifier
            user_message: User's input message
            agent_response: Agent's response
            metadata: Optional metadata (agent_name, tools_used, etc.)

        Returns:
            Turn ID
        """
        timestamp = datetime.utcnow().isoformat()
        turn_id = self._create_turn_id(session_id, user_id, timestamp)

        # Create document content combining user and agent messages
        # This allows semantic search across both
        # Extract clean text from response (removes JSON metadata noise)
        clean_agent_response = self._extract_text_from_response(agent_response)
        document_text = f"User: {user_message}\nAgent: {clean_agent_response}"

        # Prepare metadata
        turn_metadata = {
            "turn_id": turn_id,
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": timestamp,
            "user_message": user_message,
            "agent_response": clean_agent_response,
            "turn_type": "conversation",
            **(metadata or {})
        }

        try:
            # Store in vector database
            self.vector_store.add_texts(
                texts=[document_text],
                metadatas=[turn_metadata],
                ids=[turn_id]
            )

            if self.logger:
                self.logger.debug(f"Added conversation turn {turn_id} to history")

            return turn_id

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to add conversation turn: {e}")
            raise

    def get_relevant_context(
            self,
            current_message: str,
            session_id: str,
            user_id: str,
            include_recent: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Retrieve relevant conversation context.

        Args:
            current_message: Current user message
            session_id: Session identifier
            user_id: User identifier
            include_recent: Whether to include recent turns

        Returns:
            Tuple of (recent_turns, relevant_turns)
        """
        recent_turns = []
        relevant_turns = []

        try:
            # 1. Get recent turns from this session (sliding window)
            if include_recent:
                recent_turns = self._get_recent_turns(session_id, user_id)

            # 2. Get semantically relevant turns from history
            relevant_turns = self._get_semantic_matches(
                current_message,
                session_id,
                user_id,
                exclude_ids=[t['turn_id'] for t in recent_turns]
            )

            return recent_turns, relevant_turns

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to retrieve conversation context: {e}")
            return [], []

    def _get_recent_turns(
            self,
            session_id: str,
            user_id: str
    ) -> List[Dict[str, Any]]:
        """Get most recent conversation turns from the session."""
        try:
            # Query by session_id and sort by timestamp
            if hasattr(self.vector_store, 'query'):
                results = self.vector_store.query(
                    filter_metadata={
                        "session_id": session_id,
                        "user_id": user_id
                    },
                    n_results=self.max_recent_turns,
                    order_by="timestamp",
                    order="desc"
                )

                # Extract metadata
                turns = []
                for result in results:
                    if hasattr(result, 'metadata'):
                        turns.append(result.metadata)

                # Reverse to chronological order
                return list(reversed(turns))

            return []

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not retrieve recent turns: {e}")
            return []

    def _get_semantic_matches(
            self,
            query: str,
            session_id: str,
            user_id: str,
            exclude_ids: List[str] = None
    ) -> List[Dict[str, Any]]:
        """Get semantically similar conversation turns."""
        try:
            # Perform semantic search
            if hasattr(self.vector_store, 'similarity_search_with_score'):
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=self.max_relevant_turns,  # Get more to filter
                    filter={"user_id": user_id}  # User-specific history
                )

                # Filter and process results
                relevant = []
                exclude_set = set(exclude_ids or [])

                distance_type = None

                for item in results:
                    # Handle different return formats
                    if isinstance(item, tuple) and len(item) == 2:
                        doc, score = item
                        metadata = doc.metadata if hasattr(doc, 'metadata') else doc
                    else:
                        doc = item
                        score = getattr(doc, 'score', 0.0)
                        metadata = doc.metadata if hasattr(doc, 'metadata') else doc

                    # Skip if already in recent turns
                    if metadata.get('turn_id') in exclude_set:
                        continue
                    if distance_type is None:
                        distance_type = self._detect_distance_type([score])
                    normalized_score = self._normalize_score(score, distance_type)

                    # Check similarity threshold
                    if normalized_score >= self.similarity_threshold:
                        metadata['similarity_score'] = normalized_score
                        metadata['raw_score'] = score
                        metadata['distance_type'] = distance_type
                        relevant.append(metadata)

                    # Stop when we have enough
                    if len(relevant) >= self.max_relevant_turns:
                        break

                return relevant

            return []

        except Exception as e:
            if self.logger:
                self.logger.warning(f"Could not perform semantic search: {e}")
            return []

    def _extract_text_from_response(self, agent_response: Any) -> str:
        """Extract clean text from agent response (handles various formats).

        Args:
            agent_response: Agent response in various formats

        Returns:
            Clean text string for embedding
        """
        # If it's already a string, check if it's JSON
        if isinstance(agent_response, str):
            # Try to parse as JSON if it looks like a dict
            if agent_response.strip().startswith('{'):
                try:
                    import json
                    response_dict = json.loads(agent_response)
                    return self._extract_text_from_response(response_dict)
                except:
                    # Not valid JSON, return as-is
                    return agent_response
            return agent_response

        # If it's a dict, extract the actual text content
        if isinstance(agent_response, dict):
            # Check for common response structures
            if 'content' in agent_response:
                content = agent_response['content']

                # Handle list of content blocks
                if isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            text_parts.append(item['text'])
                        elif isinstance(item, str):
                            text_parts.append(item)
                    return '\n'.join(text_parts)

                # Handle direct string content
                elif isinstance(content, str):
                    return content

            # Fallback: look for 'text', 'message', or 'output' keys
            for key in ['text', 'message', 'output', 'response']:
                if key in agent_response:
                    return str(agent_response[key])

        # Last resort: convert to string
        return str(agent_response)

    def _normalize_score(self, score: float, distance_type: str = 'euclidean') -> float:
        """Normalize distance/score to 0-1 similarity range.

        Args:
            score: Raw score/distance from vector store
            distance_type: Type of distance metric ('euclidean', 'cosine', 'dot')

        Returns:
            Normalized similarity score (0-1, higher is better)
        """
        if distance_type == 'cosine':
            # Cosine distance is typically 0-2, convert to similarity
            # Cosine similarity = 1 - (cosine_distance / 2)
            return max(0.0, min(1.0, 1 - (score / 2)))

        elif distance_type == 'dot':
            # Dot product can be negative, normalize to 0-1
            # This is approximate - adjust based on your embedding model
            return max(0.0, min(1.0, (score + 1) / 2))

        elif distance_type == 'euclidean':
            # Euclidean distance - convert to similarity
            # Similarity = 1 / (1 + distance)
            # Scores like 144, 148 suggest unnormalized L2 distance
            return 1 / (1 + score)

        else:
            # Unknown metric, return as-is
            return score

    def _detect_distance_type(self, scores: List[float]) -> str:
        """Auto-detect distance metric type from score ranges.

        Args:
            scores: List of scores from vector store

        Returns:
            Detected distance type
        """
        if not scores:
            return 'cosine'

        max_score = max(scores)
        min_score = min(scores)

        # Cosine: typically 0-2
        if 0 <= min_score <= 2 and 0 <= max_score <= 2:
            return 'cosine'

        # Dot product: can be negative
        elif min_score < 0:
            return 'dot'

        # Large positive values: likely euclidean distance
        elif max_score > 10:
            return 'euclidean'

        # Default to cosine if unclear
        return 'cosine'

    def format_context_for_prompt(
            self,
            recent_turns: List[Dict[str, Any]],
            relevant_turns: List[Dict[str, Any]]
    ) -> str:
        """Format conversation context for inclusion in prompt.

        Args:
            recent_turns: Recent conversation turns
            relevant_turns: Semantically relevant turns

        Returns:
            Formatted context string
        """
        context_parts = []

        # Add recent conversation
        if recent_turns:
            context_parts.append("=== Recent Conversation ===")
            for turn in recent_turns:
                context_parts.append(f"User: {turn['user_message']}")
                context_parts.append(f"Agent: {turn['agent_response']}")
                context_parts.append("")

        # Add relevant historical context
        if relevant_turns:
            context_parts.append("=== Relevant Context from History ===")
            for turn in relevant_turns:
                score = turn.get('similarity_score', 0)
                timestamp = turn.get('timestamp', 'unknown')
                context_parts.append(f"[Relevance: {score:.2f}, Time: {timestamp}]")
                context_parts.append(f"User: {turn['user_message']}")
                context_parts.append(f"Agent: {turn['agent_response']}")
                context_parts.append("")

        return "\n".join(context_parts)
