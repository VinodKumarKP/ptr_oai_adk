import glob
import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import List, Any, Dict, Tuple, Optional
from uuid import uuid4

from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


class LoaderError(Exception):
    """Custom exception for Loader class errors."""
    pass


class BaseDocumentLoader(ABC):
    """Base class for document loaders.

    This class provides common functionality for loading, transforming, and
    processing documents from various sources. It handles file type detection,
    metadata extraction, and integration with vector stores.
    """

    def __init__(self,
                 db_name: str = 'default_db',
                 vector_store: Any = None,
                 embedding: Embeddings = None,
                 persist_directory: str = None,
                 collection_name: str = None):
        """Initialize the BaseDocumentLoader.

        Args:
            db_name: Name of the database (default: 'default_db').
            vector_store: Vector store instance.
            embedding: Embedding function instance.
            persist_directory: Directory for persisting data.

        Raises:
            LoaderError: If neither vector_store nor embedding is provided.
        """
        if vector_store is None and embedding is None:
            raise LoaderError("Either vector_store or embedding must be provided")

        persist_directory = persist_directory if persist_directory else tempfile.gettempdir()
        self.persist_directory = persist_directory

        self.vector_store = vector_store
        self.embedding = embedding
        self.logger = logging.getLogger(__name__)
        self.collection_name = collection_name
        # Note: reinitialize_database is abstract, so this call relies on subclass implementation
        self.reinitialize = self.reinitialize_database()
        self.loaded_files_log = os.path.join(self.persist_directory, f"{self.collection_name}_loaded_files.json")

    def transform_documents_with_metadata(self, original_chunks: List[Document]) -> List[Document]:
        """
        Transform documents to include metadata summary in the page content.

        Args:
            original_chunks (List[Document]): List of original document chunks.

        Returns:
            List[Document]: Transformed documents with enhanced content.

        Raises:
            LoaderError: If transformation fails for all documents.
            TypeError: If original_chunks is not a list.
        """
        if not original_chunks:
            self.logger.warning("Empty document list provided for transformation")
            return []

        if not isinstance(original_chunks, list):
            raise TypeError("original_chunks must be a list of Document objects")

        transformed_docs = []
        failed_transformations = 0

        for i, doc in enumerate(original_chunks):
            try:
                # Validate document structure
                if not isinstance(doc, Document):
                    self.logger.warning(f"Skipping item {i}: Not a Document object")
                    failed_transformations += 1
                    continue

                if not hasattr(doc, 'page_content') or not hasattr(doc, 'metadata'):
                    self.logger.warning(f"Skipping document {i}: Missing required attributes")
                    failed_transformations += 1
                    continue

                # Safely extract metadata with defaults
                metadata = doc.metadata if doc.metadata else {}
                title = str(metadata.get('title', 'Unknown Title'))
                summary = str(metadata.get('summary', 'No summary available'))
                source = str(metadata.get('source', 'Unknown Source'))

                # Validate page content
                page_content = doc.page_content if doc.page_content else ""
                if not isinstance(page_content, str):
                    page_content = str(page_content)

                # Create enhanced page content that includes metadata
                content = f"""Title: {title}

Summary: {summary}

Content: {page_content}

Source: {source}"""

                # Create new document with enhanced content
                new_doc = Document(
                    page_content=content,
                    metadata=metadata  # Keep original metadata too
                )
                transformed_docs.append(new_doc)

            except Exception as e:
                self.logger.error(f"Failed to transform document {i}: {str(e)}")
                failed_transformations += 1
                continue

        if failed_transformations > 0:
            self.logger.warning(f"{failed_transformations} documents failed transformation")

        if not transformed_docs and original_chunks:
            raise LoaderError("All document transformations failed")

        self.logger.info(f"Successfully transformed {len(transformed_docs)} documents")
        return transformed_docs

    def _get_loader_for_file(self, file_path: str, loader_settings: Dict[str, Any] = None):
        """
        Get appropriate document loader based on file extension.

        Args:
            file_path (str): Path to the file.
            loader_settings (Dict[str, Any]): Optional settings for the loader.

        Returns:
            Document loader instance.

        Raises:
            LoaderError: If file type is unsupported or loader fails to initialize.
        """
        file_extension = file_path.split('.')[-1].lower() if '.' in file_path else ''

        loader_map = {
            'pdf': ('langchain_community.document_loaders', 'PyPDFLoader'),
            'docx': ('langchain_community.document_loaders', 'Docx2txtLoader'),
            'txt': ('langchain_community.document_loaders', 'TextLoader'),
            'json': ('langchain_community.document_loaders', 'JSONLoader'),
            'csv': ('langchain_community.document_loaders', 'CSVLoader'),
            'yaml': ('langchain_community.document_loaders', 'TextLoader'),
            # Load as text to preserve structure for LLM
            'yml': ('langchain_community.document_loaders', 'TextLoader'),
            'xml': ('langchain_community.document_loaders', 'UnstructuredXMLLoader'),
            'md': ('langchain_community.document_loaders', 'TextLoader')
        }

        # Augment loader map with custom settings
        loader_kwargs = loader_settings.copy() if loader_settings else {}
        custom_loader_map = loader_kwargs.get(file_extension, {})

        if custom_loader_map and isinstance(custom_loader_map, dict):
            class_name = custom_loader_map.get('class')
            if class_name:
                loader_map[file_extension] = class_name
            loader_kwargs = custom_loader_map.pop('settings', {})

        if file_extension not in loader_map:
            supported_types = ', '.join(loader_map.keys())
            raise LoaderError(f"Unsupported file type '{file_extension}'. Supported types: {supported_types}")

        try:
            loader_info = loader_map[file_extension]

            if isinstance(loader_info, tuple):
                module_name, class_name = loader_info
            elif isinstance(loader_info, str):
                module_name, class_name = loader_info.rsplit('.', 1)
            else:
                raise LoaderError(f"Invalid loader configuration for {file_extension}")

            module = __import__(module_name, fromlist=[class_name])
            loader_class = getattr(module, class_name)
        except ImportError as e:
            raise LoaderError(f"Required dependency for {file_extension} files not found: {str(e)}")
        except Exception as e:
            raise LoaderError(f"Failed to initialize loader for {file_extension}: {str(e)}")

        try:
            # Initialize loader with file-specific parameters
            if file_extension in ['txt', 'yaml', 'yml', 'json', 'md']:
                # Add encoding parameter for text files if not present
                if 'encoding' not in loader_kwargs:
                    loader_kwargs['encoding'] = 'utf-8'

            return loader_class(file_path, **loader_kwargs)

        except Exception as e:
            raise LoaderError(f"Failed to initialize loader for {file_path}: {str(e)}")

    def _get_loader(self, object: Any, loader_settings: Dict[str, Any] = None):
        """
        Get appropriate document loader based on object type or path.

        Args:
            object: File path, directory path, or BaseLoader instance.
            loader_settings (Dict[str, Any]): Optional settings for the loader.

        Returns:
            Document loader instance.

        Raises:
            LoaderError: If object type or path is unsupported.
        """
        if isinstance(object, BaseLoader):
            return object
        if isinstance(object, str):
            if '*' in object:
                dir_part = os.path.dirname(object.split('*', 1)[0]) or '.'
                glob_part = os.path.basename(object)
                return DirectoryLoader(dir_part, glob=glob_part, loader_kwargs=loader_settings)
            if 'wiki/' in object:
                from langchain_community.document_loaders import WikipediaLoader
                return WikipediaLoader(query=object.split('wiki/', 1)[1])
            if 'query' in object:
                from langchain_community.document_loaders import WikipediaLoader
                return WikipediaLoader(query=object.split('query: ', 1)[-1].strip())
            if os.path.isdir(object):
                return DirectoryLoader(object, loader_kwargs=loader_settings)
            if os.path.isfile(object):
                return self._get_loader_for_file(object, loader_settings=loader_settings)
        raise LoaderError(f"Unsupported object type or path: {object}")

    def _load_documents(self, knowledge_base_list: List[str], loader_settings: Dict[str, Any] = None) -> List[Document]:
        """
        Load documents from a list of file paths.

        Args:
            knowledge_base_list (List[str]): List of file paths to load.
            loader_settings (Dict[str, Any]): Optional settings for the loader.

        Returns:
            List[Document]: Loaded documents.

        Raises:
            LoaderError: If no files can be loaded successfully.
            ValueError: If knowledge_base_list is empty.
            TypeError: If knowledge_base_list is not a list.
        """
        if not knowledge_base_list:
            raise ValueError("file_list cannot be empty")

        if not isinstance(knowledge_base_list, list):
            raise TypeError("file_list must be a list of file paths")

        all_documents = []
        failed_files = []

        for knowledge_base in knowledge_base_list:
            try:
                # Load document
                loader = self._get_loader(knowledge_base, loader_settings=loader_settings)

                if isinstance(loader, DirectoryLoader):
                    documents = []
                    self.logger.info(f"Processing directory: {knowledge_base}")
                    file_list = glob.glob(knowledge_base)
                    for file in file_list:
                        self.logger.info(f"Processing file: {file}")
                        loader = self._get_loader(file, loader_settings=loader_settings)
                        tmp_documents = loader.load()
                        # Add source metadata if not present
                        for doc in documents:
                            if 'source' not in doc.metadata:
                                doc.metadata['source'] = os.path.basename(file)

                        documents.extend(tmp_documents)

                else:
                    documents = loader.load()

                    # Validate loaded documents
                    if not documents:
                        self.logger.warning(f"No content extracted from file: {knowledge_base}")
                        failed_files.append(knowledge_base)
                        continue

                all_documents.extend(documents)
                self.logger.info(f"Successfully loaded {len(documents)} documents from: {knowledge_base}")

            except Exception as e:
                self.logger.error(f"Failed to load file {knowledge_base}: {str(e)}")
                failed_files.append(knowledge_base)
                continue

        # Report results
        if failed_files:
            self.logger.warning(f"Failed to load {len(failed_files)} files: {failed_files}")

        if not all_documents:
            raise LoaderError("No documents could be loaded from the provided file list")

        self.logger.info(
            f"Successfully loaded {len(all_documents)} documents from {len(knowledge_base_list) - len(failed_files)} files")
        return all_documents

    def _process_s3_source(self, s3_uri: str,
                           loaded_files: Dict[str, int],
                           session_id: Optional[str] = None,
                           additional_configs: Optional[Dict[str, str]] = None,
                           ) -> Tuple[List[str], Dict[str, int]]:
        try:
            import boto3
        except ImportError:
            self.logger.error("boto3 is required for S3 loading")
            return [], {}

        parts = s3_uri.replace("s3://", "").split("/", 1)
        bucket_name = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        s3 = boto3.client('s3', region_name=additional_configs.get('region', 'us-east-1'))

        # Local download directory
        local_dir = os.path.join(self.persist_directory, "s3_bucket", bucket_name)
        os.makedirs(local_dir, exist_ok=True)

        files_to_load = []
        new_loaded_files = {}

        try:
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

            for page in pages:
                if 'Contents' not in page:
                    continue

                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):
                        continue

                    size = obj['Size']
                    s3_full_uri = f"s3://{bucket_name}/{key}"

                    if session_id:
                        loaded_key = f"{session_id}::{s3_full_uri}"
                    else:
                        loaded_key = s3_full_uri

                    if loaded_key in loaded_files and loaded_files[loaded_key] == size:
                        continue

                    local_file_path = os.path.join(local_dir, key)
                    local_file_dir = os.path.dirname(local_file_path)
                    os.makedirs(local_file_dir, exist_ok=True)

                    try:
                        s3.download_file(bucket_name, key, local_file_path)
                        files_to_load.append(local_file_path)
                        new_loaded_files[loaded_key] = size
                    except Exception as e:
                        self.logger.error(f"Failed to download {s3_full_uri}: {e}")

        except Exception as e:
            self.logger.error(f"Error listing S3 objects for {s3_uri}: {e}")
            return [], {}

        return files_to_load, new_loaded_files

    def _process_local_source(self, doc_path: str, loaded_files: Dict[str, int], session_id: Optional[str] = None) -> \
    Tuple[List[str], Dict[str, int]]:
        """Process local file system source."""
        if '*' in doc_path:
            files = glob.glob(doc_path)
        elif os.path.isdir(doc_path):
            try:
                files = [os.path.join(doc_path, f) for f in os.listdir(doc_path) if
                         os.path.isfile(os.path.join(doc_path, f))]
            except Exception as e:
                self.logger.error(f"Failed to list directory {doc_path}: {str(e)}")
                files = []
        else:
            files = [doc_path]

        files_to_load = []
        new_loaded_files = {}

        for file_path in files:
            if not os.path.exists(file_path):
                continue

            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path)

            if session_id:
                key = f"{session_id}::{file_name}"
            else:
                key = file_name

            if key in loaded_files and loaded_files[key] == file_size:
                continue

            files_to_load.append(file_path)
            new_loaded_files[key] = file_size

        return files_to_load, new_loaded_files

    def _update_s3_metadata(self, docs: List[Document]):
        """Update metadata source to reflect S3 URI."""
        for d in docs:
            source = d.metadata.get('source', '')
            if source.startswith(self.persist_directory):
                # Handle path relative to persist directory
                try:
                    rel_path = os.path.relpath(source, os.path.join(self.persist_directory, "s3_bucket"))
                    # If rel_path starts with .., it means it's not inside s3_bucket dir
                    if not rel_path.startswith(".."):
                        d.metadata['source'] = f"s3://{rel_path}"
                except ValueError:
                    pass
            elif os.path.isabs(source) and "s3_bucket" in source:
                # Fallback for absolute paths containing s3_bucket
                try:
                    idx = source.index("s3_bucket")
                    # +1 for the separator
                    rel_path = source[idx + len("s3_bucket") + 1:]
                    d.metadata['source'] = f"s3://{rel_path}"
                except:
                    pass

    def _generate_chunk(self, docs_dict: Dict[str, Dict[str, Any]]) -> Tuple[List[Document], Dict[str, int]]:
        """Generate chunks from documents.

        Args:
            docs: List of document paths.

        Returns:
            Tuple containing:
            - List of valid Document chunks
            - Dictionary of loaded files and their sizes
        """
        # Validate documents
        valid_docs = []
        loaded_files = self._get_loaded_files()

        for doc in docs_dict:
            session_id = docs_dict[doc].get('session_id')
            loader_settings = docs_dict[doc].get('loader', {})

            is_s3 = False
            if doc.startswith('s3://'):
                region = docs_dict[doc]['region']
                files_to_load, file_updates = self._process_s3_source(doc, loaded_files, session_id, {'region': region})
                is_s3 = True
            else:
                files_to_load, file_updates = self._process_local_source(doc, loaded_files, session_id)

            if not files_to_load:
                continue

            try:
                docs = self._load_documents(knowledge_base_list=files_to_load, loader_settings=loader_settings)

                if is_s3:
                    self._update_s3_metadata(docs)

            except Exception as e:
                self.logger.error(f"Failed to load documents for {doc}: {str(e)}")
                continue

            # Apply chunking strategy
            docs = self._split_text(doc, docs, docs_dict)

            # Add session_id to metadata if available
            if session_id:
                for d in docs:
                    d.metadata['session_id'] = session_id

            for i, doc_item in enumerate(docs):
                if not isinstance(doc_item, Document):
                    self.logger.warning(f"Skipping item {i}: Not a Document object")
                    continue
                if not doc_item.page_content or not doc_item.page_content.strip():
                    self.logger.warning(f"Skipping document {i}: Empty content")
                    continue
                valid_docs.append(doc_item)

            loaded_files.update(file_updates)

        if not valid_docs and not loaded_files:
            raise LoaderError("No valid documents found for loading")

        if not valid_docs and loaded_files and len(docs_dict) > 0:
            self.logger.info("All documents were already loaded.")
            return [], loaded_files

        self.logger.info(f"Processing {len(valid_docs)} valid documents")
        return valid_docs, loaded_files

    def _get_loaded_files(self) -> Dict[str, int]:
        """Get the list of loaded files and their sizes."""
        if os.path.exists(self.loaded_files_log):
            try:
                with open(self.loaded_files_log, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load loaded files log: {str(e)}")
        return {}

    def _save_loaded_files(self, loaded_files: Dict[str, int]):
        """Save the list of loaded files and their sizes."""
        try:
            with open(self.loaded_files_log, 'w') as f:
                json.dump(loaded_files, f)
        except Exception as e:
            self.logger.error(f"Failed to save loaded files log: {str(e)}")

    def _split_text(self, doc: str, docs: list[Document], docs_dict: dict[str, dict[str, Any]]) -> list[Document]:
        """Split text into chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=docs_dict[doc].get('chunk_size', 1000),
            chunk_overlap=docs_dict[doc].get('chunk_overlap', 200)
        )
        docs = text_splitter.split_documents(docs)
        return docs

    @abstractmethod
    def reinitialize_database(self) -> bool:
        """
        Check if the database needs to be reinitialized.

        Returns:
            bool: True if the database needs to be reinitialized, False otherwise.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def _get_document_count(self) -> int:
        """Get the current document count from the vector store.

        Returns:
            int: Number of documents in the vector store.
        """
        if hasattr(self.vector_store, 'count'):
            return self.vector_store.count()
        elif hasattr(self.vector_store, '_collection') and hasattr(self.vector_store._collection, 'count'):
            return self.vector_store._collection.count()
        return 0

    def _get_collection_name(self) -> str:
        """Get the collection name from the vector store.

        Returns:
            str: Name of the collection.
        """
        if hasattr(self.vector_store, 'collection_name'):
            return self.vector_store.collection_name
        elif hasattr(self.vector_store, '_collection') and hasattr(self.vector_store._collection, 'name'):
            return self.vector_store._collection.name
        return "unknown"

    def load_db(self, docs: Dict[str, Dict[str, Any]]):
        """Load documents into the vector store with comprehensive validation.

        Args:
            docs: List of file paths or document sources to load.

        Returns:
            Vector Store: Initialized vector store wrapper.

        Raises:
            LoaderError: If Vector database loading fails.
        """
        try:
            current_count = self._get_document_count()

            if self.reinitialize and current_count > 0:
                collection_name = self._get_collection_name()

                if hasattr(self.vector_store, 'reset_collection'):
                    self.vector_store.reset_collection()
                    self.logger.info(f"Deleted existing collection: {collection_name}")

            if not self.reinitialize and current_count > 0:
                self.logger.info(f"Using existing collection with {current_count} documents")

            self.load_documents(docs)

            return self.vector_store

        except Exception as e:
            self.logger.error(f"Vector store loading failed: {str(e)}")
            raise LoaderError(f"Failed to load documents into vector store: {str(e)}")

    def load_documents(self, docs: Dict[str, Dict[str, Any]]):
        """Load documents into the vector store.

        Args:
            docs: List of file paths or document sources to load.
        """

        # Generate chunks from documents
        chunks, loaded_files = self._generate_chunk(docs_dict=docs)

        if not chunks:
            self.logger.info("No new documents to add to Vector database.")
            return

        self.logger.info(f"Adding {len(chunks)} documents to Vector database...")

        # Batch processing to avoid rate limits
        batch_size = 50
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            uuids = [str(uuid4()) for _ in range(len(batch))]
            self.vector_store.add_documents(documents=batch, ids=uuids)
            self.logger.info(f"Added batch {i // batch_size + 1} of {(total_chunks + batch_size - 1) // batch_size}")

        # Save loaded files log after successful addition to vector store
        self._save_loaded_files(loaded_files)

        # Verify collection
        final_count = self._get_document_count()
        self.logger.info(f"Successfully created Vector database with {final_count} embeddings")

    def query(self, query_text: str, n_results: int = 5) -> List[Document]:
        """Query the vector store collection.

        Args:
            query_text: The query text.
            n_results: Number of results to return.

        Returns:
            List of Document objects with results.
        """
        if self.vector_store:
            return self.vector_store.similarity_search(query_text, k=n_results)

        self.logger.warning("Vector store not initialized. Call load_db() first.")
        return []

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection.

        Returns:
            Dictionary with collection statistics.
        """
        stats = {
            'name': self._get_collection_name(),
            'count': self._get_document_count(),
            'metadata': {}
        }

        if hasattr(self.vector_store, 'collection') and hasattr(self.vector_store.collection, 'metadata'):
            stats['metadata'] = self.vector_store.collection.metadata

        return stats

    def reset_collection(self):
        """Reset the collection by deleting and recreating it.
        
        Raises:
            LoaderError: If reset fails.
        """
        try:
            collection_name = self._get_collection_name()

            if hasattr(self.vector_store, 'reset_collection'):
                self.vector_store.reset_collection()
                self.logger.info(f"Reset collection: {collection_name}")
            else:
                self.logger.warning("Vector store does not support reset_collection")

            # Clear loaded files log
            if os.path.exists(self.loaded_files_log):
                os.remove(self.loaded_files_log)

        except Exception as e:
            self.logger.error(f"Failed to reset collection: {str(e)}")
            raise LoaderError(f"Failed to reset collection: {str(e)}")
