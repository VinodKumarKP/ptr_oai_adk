"""Base Factory for creating knowledge base tools."""

import logging
import os
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable

from oai_agent_core.components.vector_store.vector_store_factory import VectorStoreFactory
from oai_agent_core.utils.prompt_analyzer import PromptAnalyzer
from oai_agent_core.utils.env_resolver import ConfigResolver


class BaseKnowledgeBaseFactory(ABC):
    """Base Factory for creating knowledge base tools.

    This class handles the initialization of vector stores and creation of
    search tools for knowledge bases. It supports custom document loading
    and embedding generation.
    """

    def __init__(self, knowledge_base_config: List[Dict[str, Any]],
                 logger: Optional[logging.Logger] = None,
                 project_root: Optional[str] = None, llm: Any = None,
                 document_loader: Callable = None,
                 vector_store: Any = None):
        """Initialize the factory.

        Args:
            knowledge_base_config: List of knowledge base configurations.
            logger: Optional logger instance.
            project_root: Optional project root path for resolving relative paths.
            llm: Optional LLM instance for query analysis.
            document_loader: Optional callable to create a document loader instance.
            vector_store: Optional vector store instance.
        """
        self.logger = logger or logging.getLogger(__name__)
        self.project_root = project_root
        self.query_analyzer = PromptAnalyzer(llm, self.logger) if llm else None
        self.vector_store = vector_store
        self.document_loader = document_loader
        self.env_resolver = ConfigResolver(logger=self.logger, allow_eval=os.environ.get('ALLOW_EVAL', False))
        self.loader = None
        self.similarity_threshold = 0.6  # Default threshold
        self.knowledge_base_tools = {}
        self._initialize_knowledge_bases(knowledge_base_config)

    def _initialize_knowledge_bases(self, knowledge_bases: List[Dict[str, Any]]):
        """Initialize knowledge bases from configuration.

        Args:
            knowledge_bases: List of knowledge base configuration dictionaries.
        """
        for kb_config in knowledge_bases:
            self._create_single_knowledge_base(kb_config)

    def _process_data_sources(self, data_sources: List[Dict[str, Any]],
                              text_splitter_settings: Dict[str, Any],
                              loader_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Process data sources and return a docs_dict for the document loader.

        Handles three source types:
        - ``file`` (default): local file, directory, or glob path.
        - ``s3``: S3 bucket URI constructed from ``bucket`` + ``key``/``prefix``.
        - ``dynamic``: any LangChain community loader specified by a dotted
          ``loader`` class path and a ``settings`` block.

        Args:
            data_sources: List of data source configurations from YAML.
            text_splitter_settings: Default chunk_size / chunk_overlap settings.
            loader_settings: File-level loader kwargs forwarded as ``loader`` key.

        Returns:
            docs_dict ready to pass directly to :meth:`loader.load_db`.
        """
        docs_paths = {}

        for i, source in enumerate(data_sources):
            # Infer type: explicit 'type' key > presence of 'loader' key > 'path' key
            if 'type' in source:
                source_type = source['type']
            elif 'loader' in source and 'path' not in source:
                # 'loader' with no 'path' means a dynamic community loader
                source_type = 'dynamic'
            else:
                source_type = 'file'

            # Base settings: text splitter defaults merged with any per-source overrides
            source_settings = text_splitter_settings.copy()
            source_settings['loader'] = loader_settings

            if source_type == 'dynamic':
                loader_class = source.get('loader')
                if len(loader_class.split('.')) == 1:
                    loader_class = "langchain_community.document_loaders." + loader_class
                if not loader_class:
                    self.logger.warning(f"Dynamic data source at index {i} is missing 'loader'. Skipping.")
                    continue

                key = f"__dynamic_{i}__"
                docs_paths[key] = {
                    **source_settings,
                    'type': 'dynamic',
                    'loader_class': loader_class,
                    'settings': self.env_resolver.resolve_dict(source.get('settings', {})),
                    'ttl_seconds': source.get('ttl_seconds', 3600),
                }

            elif source_type == 'file':
                path = source.get('path')
                if not path:
                    self.logger.warning(f"File data source at index {i} is missing 'path'. Skipping.")
                    continue

                if not os.path.isabs(path) and self.project_root:
                    full_path = os.path.join(self.project_root, path)
                else:
                    full_path = path

                docs_paths[full_path] = {
                    **source_settings,
                    'type': 'file',
                }

            elif source_type == 's3':
                bucket = source.get('bucket')
                if not bucket:
                    self.logger.warning(f"S3 data source at index {i} is missing 'bucket'. Skipping.")
                    continue

                key = source.get('key', '')
                prefix = source.get('prefix', '')
                region = source.get('region', 'us-east-1')
                uri = f"s3://{bucket}/{key if key else prefix}"

                docs_paths[uri] = {
                    **source_settings,
                    'type': 's3',
                    'region': region,
                }

            else:
                self.logger.warning(f"Unknown source type '{source_type}' at index {i}. Skipping.")

        return docs_paths

    def _create_single_knowledge_base(self, kb_config: Dict[str, Any]):
        """Create a single knowledge base and register it as a tool.

        Args:
            kb_config: Configuration for a single knowledge base.
        """
        name = kb_config.get('name', 'default_knowledge_base')
        description = kb_config.get('description', 'Search the knowledge base.')

        # Vector Store Configuration
        vector_store_config = kb_config.get('vector_store', {})
        vector_store_type = vector_store_config.get('type', 'chroma')
        vector_store_settings = vector_store_config.get('settings', {})

        # Embedding Configuration
        embedding_config = kb_config.get('embedding', {})
        embedding_model_id = embedding_config.get('model_id', 'amazon.titan-embed-text-v1')
        region_name = embedding_config.get('region_name', 'us-west-2')

        # Retrieval Settings
        retrieval_settings = kb_config.get('retrieval_settings', {})
        self.similarity_threshold = retrieval_settings.get('score_threshold', self.similarity_threshold)

        # Resolve persist directory if needed
        persist_directory = vector_store_settings.get('persist_directory')
        if persist_directory:
            if not os.path.isabs(persist_directory) and self.project_root:
                persist_directory = os.path.join(self.project_root, persist_directory)
            else:
                persist_directory = os.path.abspath(persist_directory)
            os.makedirs(persist_directory, exist_ok=True)
            vector_store_settings['persist_directory'] = persist_directory

        # Create Embeddings
        embeddings = self._create_embeddings(embedding_model_id, region_name)
        if not embeddings:
            self.logger.error(f"Could not create embeddings for knowledge base: {name}")
            return

        # Create Vector Store
        try:
            vector_store = VectorStoreFactory.create_vector_store(
                vector_store_type,
                embedding_function=embeddings,
                **vector_store_settings
            )
        except Exception as e:
            self.logger.error(f"Failed to create vector store for {name}: {e}")
            return

        # Load Documents
        data_sources = kb_config.get('data_sources', [])
        text_splitter_settings = kb_config.get('text_splitter', {})
        loader_settings = kb_config.get("loader", {})
        docs_paths = self._process_data_sources(data_sources, text_splitter_settings, loader_settings)

        if self.document_loader:
            self.loader = self.document_loader(
                db_name=vector_store_settings.get('collection_name', 'default'),
                embedding=embeddings,
                persist_directory=persist_directory,
                vector_store=vector_store
            )
        else:
            from oai_agent_core.components.loaders.document_loader import DocumentLoader
            self.loader = DocumentLoader(
                db_name=vector_store_settings.get('collection_name', 'default'),
                embedding=embeddings,
                persist_directory=persist_directory,
                vector_store=vector_store
            )

        vector_load_type = 'custom'
        if docs_paths:
            self.loader.load_db(docs_paths)
            vector_load_type = 'preloaded'

        # Create Tool
        self.knowledge_base_tools[name] = {
            'vector_store': vector_store,
            'description': description,
            'retrieval_settings': retrieval_settings,
            'vector_load_type': vector_load_type
        }

        # If this is the "main" or only KB, we might want to set self.vector_store for backward compatibility
        if not self.vector_store:
            self.vector_store = vector_store

    def load_documents(self, doc_list: List[str], session_id: Optional[str] = None):
        """
        Load documents into the knowledge base.
        :param doc_list: doc list
        :param session_id: session id
        :return:
        """
        doc_paths = {}
        for doc in doc_list:
            doc_paths[doc] = {}
            if session_id:
                doc_paths[doc]['session_id'] = session_id

        self.loader.load_documents(doc_paths)


    def search_knowledge_base(self, query: str,
                              kb_name: str = None,
                              source_list: List[str] = None,
                              session_id: Optional[str] = None) -> str:
        """Search a specific knowledge base or the default one.

        Args:
            query: The search query string.
            kb_name: The name of the knowledge base to search. If None, searches the default/first one.
            source_list: List of sources to filter by.
            session_id: Optional session_id

        Returns:
            A formatted string containing the content and source of relevant documents.
        """
        if kb_name:
            kb_data = self.knowledge_base_tools.get(kb_name)
            if not kb_data:
                return f"Knowledge base '{kb_name}' not found."
            vector_store = kb_data['vector_store']
            retrieval_settings = kb_data.get('retrieval_settings', {})
            vector_load_type = kb_data.get('vector_load_type', 'custom')
        else:
            # Fallback to default behavior (using self.vector_store or first available)
            if self.vector_store:
                vector_store = self.vector_store
                # Try to find settings for this store if possible, otherwise defaults
                retrieval_settings = {}
                vector_load_type = 'custom'
                # If we have tools registered, try to find which one matches self.vector_store
                for k, v in self.knowledge_base_tools.items():
                    if v['vector_store'] == self.vector_store:
                        retrieval_settings = v.get('retrieval_settings', {})
                        break
            elif self.knowledge_base_tools:
                # Use the first one
                first_key = next(iter(self.knowledge_base_tools))
                kb_data = self.knowledge_base_tools[first_key]
                vector_store = kb_data['vector_store']
                retrieval_settings = kb_data.get('retrieval_settings', {})
                vector_load_type = kb_data.get('vector_load_type', 'custom')
            else:
                return "No knowledge base available."

        top_k = retrieval_settings.get('top_k', 5)
        score_threshold = retrieval_settings.get('score_threshold', self.similarity_threshold)

        # Analyze query if analyzer is available
        queries = [query]
        if self.query_analyzer:
            queries = self.query_analyzer.analyze(query)

        all_results = []

        # Prepare filter if source_list is provided
        search_kwargs = {}
        if source_list:
            # Pass the list directly. The vector store implementation will handle it.
            # For Chroma, Postgres, and S3 stores, we've updated them to handle list values as OR conditions.
            search_kwargs['filter'] = {'source': source_list}

        if session_id and vector_load_type == 'custom':
            if 'filter' not in search_kwargs:
                search_kwargs['filter'] = {}
            search_kwargs['filter']['session_id'] = session_id

        for q in queries:
            results = vector_store.similarity_search_with_score(q, k=top_k, **search_kwargs)
            all_results.extend(results)

        # Deduplicate results based on content and source
        unique_results = {}
        distance_type = None

        # Collect scores to detect distance type
        scores = []
        for item in all_results:
            if isinstance(item, tuple) and len(item) == 2:
                _, score = item
                scores.append(score)
            else:
                doc = item
                score = getattr(doc, 'score', 0.0)
                scores.append(score)

        distance_type = self._detect_distance_type(scores)

        for item in all_results:
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
            else:
                doc = item
                score = getattr(doc, 'score', 0.0)

            # Normalize score
            normalized_score = self._normalize_score(score, distance_type)

            # Filter by threshold
            if normalized_score < score_threshold:
                continue

            # Add score to metadata for display
            doc.metadata['similarity_score'] = normalized_score
            doc.metadata['raw_score'] = score

            key = (doc.page_content, doc.metadata.get('source', 'unknown'))

            # Keep the one with the higher score if duplicate
            if key not in unique_results or unique_results[key].metadata.get('similarity_score', 0) < normalized_score:
                unique_results[key] = doc

        # Sort by score descending
        final_results = sorted(
            list(unique_results.values()),
            key=lambda x: x.metadata.get('similarity_score', 0),
            reverse=True
        )

        if not final_results:
            return "No relevant information found in the knowledge base."

        return "\n\n".join(
            [f"Content: {doc.page_content}\nSource: {doc.metadata.get('source', 'unknown')}\nRelevance: {doc.metadata.get('similarity_score', 0):.2f}" for doc in final_results])

    # Keep backward compatibility for search_custom_knowledge_base
    def search_custom_knowledge_base(self, query: str) -> str:
        return self.search_knowledge_base(query)

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

    @abstractmethod
    def create_tool(self, name:str, description:str) -> Any:
        """Create a tool for searching the knowledge base.

        This method must be implemented by subclasses to return the framework-specific tool.

        Returns:
            A tool instance compatible with the target agent framework.
        """
        pass

    @abstractmethod
    def create_load_tool(self, name: str, description: str) -> Any:
        """Create a tool for loading the knowledge base.

        This method must be implemented by subclasses to return the framework-specific tool.

        Returns:
            A tool instance compatible with the target agent framework.
        """
        pass

    def get_tools(self) -> List[Any]:
        """Get all knowledge base tools.

        Returns:
            List of tool instances.
        """
        tools = []
        for name, data in self.knowledge_base_tools.items():
            tools.append(self.create_tool(name=name, description=data['description']))
            tools.append(self.create_load_tool(name=name, description=f"Load the knowledge base {name}"))
        return tools

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