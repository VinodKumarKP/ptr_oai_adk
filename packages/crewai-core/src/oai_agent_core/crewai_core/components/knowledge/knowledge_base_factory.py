"""Factory for creating knowledge base tools for CrewAI."""

import logging
from typing import Dict, Any, List, Optional

try:
    from crewai.tools import tool
except ImportError:
    raise ImportError("Please install crewai: pip install crewai")

from oai_agent_core.core.base_knowledge_base_factory import BaseKnowledgeBaseFactory


class KnowledgeBaseFactory(BaseKnowledgeBaseFactory):
    """Factory for creating knowledge base tools for CrewAI.

    Handles creation of tools for:
    - Custom Chroma-based Knowledge Base
    """

    def __init__(self, knowledge_base_config: List[Dict[str, Any]],
                 logger: Optional[logging.Logger] = None,
                 project_root: Optional[str] = None,
                 llm: Any = None,
                 vector_store: Any = None,
                 document_loader: Any = None):
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

        def search_knowledge_base_tool(query: str, source_list: List[str] = None, session_id: Optional[str] = None) -> str:
            """Search the knowledge base for relevant documents.
            Args:
                query: The search query string.
                source_list: List of sources to filter by in the knowledge base.
                session_id: Session id, if available
            """
            return self.search_knowledge_base(query, kb_name=name, source_list=source_list, session_id=session_id)

        # Set the docstring dynamically before decoration
        search_knowledge_base_tool.__doc__ = """
        Search the knowledge base for relevant documents.
        Args:
            query: The search query string.
            source_list: List of sources to filter by in the knowledge base.
            session_id: Session id, if available
        """

        # Manually apply the @tool decorator
        # tool(name) returns the actual decorator function
        decorated_tool = tool(f"search_knowledge_base_{name}")(search_knowledge_base_tool)

        return decorated_tool

    def create_load_tool(self, name: str = "load_knowledge_base", description: str = "Load the knowledge base.") -> Any:
        """Create a tool for loading documents into the knowledge base.

        Args:
            name: The name of the tool.
            description: The description of the tool.

        Returns:
            A CrewAI @tool decorated callable that agents can invoke to load
            documents into the knowledge base.
        """

        def load_knowledge_base_tool(doc_list: List[str], session_id: Optional[str] = None) -> str:
            """Load the knowledge base.

            Args:
                doc_list: List of document paths to load.
                session_id: Session id, if available
            """
            return self.load_documents(doc_list, session_id)

        # Update the tool's name and doc
        load_knowledge_base_tool.__doc__ = """
        Load the knowledge base
        Args:
            doc_list: List of document paths to load.
            session_id: Session id, if available
        """

        # Manually apply the @tool decorator
        # tool(name) returns the actual decorator function
        decorated_tool = tool(f"Load_knowledge_base_{name}")(load_knowledge_base_tool)

        return decorated_tool
