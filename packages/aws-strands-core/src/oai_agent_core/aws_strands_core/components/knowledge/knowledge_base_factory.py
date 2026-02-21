"""Factory for creating knowledge base tools."""

import logging
from typing import Dict, Any, List, Optional, Callable

try:
    from strands.tools import tool
except ImportError:
    raise ImportError("Please install strands-agents: pip install strands-agents")

from oai_agent_core.core.base_knowledge_base_factory import BaseKnowledgeBaseFactory


class KnowledgeBaseFactory(BaseKnowledgeBaseFactory):
    """Factory for creating knowledge base tools.

    Handles creation of tools for:
    - Custom Chroma-based Knowledge Base
    """

    def __init__(self, knowledge_base_config: List[Dict[str, Any]],
                 logger: Optional[logging.Logger] = None,
                 project_root: Optional[str] = None,
                 llm: Any = None,
                 vector_store: Any = None,
                 document_loader: Callable = None):
        """Initialize the factory.

        Args:
            knowledge_base_config: List of knowledge base configurations
            logger: Optional logger instance
            project_root: Optional project root path for resolving relative paths
            llm: Optional LLM instance for query analysis
        """
        super().__init__(knowledge_base_config=knowledge_base_config,
                         logger=logger,
                         project_root=project_root,
                         llm=llm,
                         document_loader=document_loader,
                         vector_store=vector_store)

    def create_tool(self, name: str = "search_knowledge_base",
                    description: str = "Search the knowledge base for relevant documents.") -> Any:
        """Create a tool for searching the knowledge base."""

        @tool(name=name)
        def search_knowledge_base_tool(query: str, source_list: List[str] = None, session_id: Optional[str] = None) -> str:
            """Search the knowledge base for relevant documents.

            Args:
                query: The search query string.
                source_list: List of sources to filter by.
                session_id: Session id, if available
            Returns:
                A string containing relevant document snippets or an answer.
            """
            return self.search_knowledge_base(query, kb_name=name, source_list=source_list, session_id=session_id)

        # Update the tool's name and docstring to reflect the provided name and description
        search_knowledge_base_tool.__name__ = name
        search_knowledge_base_tool.__doc__ = description

        return search_knowledge_base_tool

    def create_load_tool(self, name: str = "load_knowledge_base", description: str = "Load the knowledge base.") -> Any:
        """Create a tool for loading the knowledge base.

        Args:
            name: The name of the tool.
            description: The description of the tool.

        Returns:
            A function tool decorated with @function_tool that agents can use
            to search the knowledge base.
        """

        @tool(name=f"load_knowledge_base_{name}")
        def load_knowledge_base_tool(doc_list: List[str], session_id: Optional[str] = None) -> str:
            """Load the knowledge base.""

            Args:
                doc_list: List of document paths to load.
                session_id: Session id, if available
            """
            return self.load_documents(doc_list, session_id)

        # Update the tool's name and doc
        load_knowledge_base_tool.__name__ = f"load_knowledge_base_{name}"
        load_knowledge_base_tool.__doc__ = description

        return load_knowledge_base_tool
