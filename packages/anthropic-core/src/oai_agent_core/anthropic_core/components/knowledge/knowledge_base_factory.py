"""Knowledge base factory for Anthropic / claude-agent-sdk agents.

Delegates all vector-store and document-loading logic to the platform-core
``BaseKnowledgeBaseFactory``.  This subclass only adds the two abstract
implementations required by the base class:

- ``create_tool(name, description)``  — returns a callable search tool
- ``create_load_tool(name, description)`` — returns a callable load/ingest tool

Both tools are plain Python callables.  When registered in the
``AnthropicToolRegistry`` they are wrapped as a FastMCP server so
claude-agent-sdk can invoke them as MCP tools.
"""

from typing import Any, Dict, List, Optional

from oai_agent_core.core.base_knowledge_base_factory import BaseKnowledgeBaseFactory


class KnowledgeBaseFactory(BaseKnowledgeBaseFactory):
    """Concrete KB factory for claude-agent-sdk / LiteLLM-proxy agents."""

    def __init__(
        self,
        knowledge_base_config: List[Dict[str, Any]],
        logger=None,
        project_root: Optional[str] = None,
        llm=None,
        document_loader=None,
        vector_store=None,
    ):
        super().__init__(
            knowledge_base_config=knowledge_base_config,
            logger=logger,
            project_root=project_root,
            llm=llm,
            document_loader=document_loader,
            vector_store=vector_store,
        )

    # ── BaseKnowledgeBaseFactory abstract implementations ─────────────────────

    def create_tool(self, name: str, description: str) -> Any:
        """Return a callable that searches the named knowledge base.

        The returned function is a plain Python callable compatible with
        FastMCP's ``add_tool`` so claude-agent-sdk can invoke it as an MCP tool.

        Args:
            name:        Knowledge base name (must match a key in self.knowledge_base_tools).
            description: Human-readable description used as the MCP tool description.

        Returns:
            Async callable ``search_<name>(query: str) -> str``.
        """
        kb_name = name

        async def search_knowledge_base(query: str) -> str:
            """Search the knowledge base and return relevant context."""
            try:
                results = self.search_knowledge_base(
                    query=query,
                    kb_name=kb_name,
                )
                if not results:
                    return f"No results found for query: {query}"
                return "\n\n".join(str(r) for r in results)
            except Exception as exc:
                return f"Knowledge base search error: {exc}"

        search_knowledge_base.__name__ = f"search_{kb_name.replace('-', '_')}"
        search_knowledge_base.__doc__ = description
        return search_knowledge_base

    def create_load_tool(self, name: str, description: str) -> Any:
        """Return a callable that loads / re-ingests documents into the named KB.

        Args:
            name:        Knowledge base name.
            description: Human-readable description for the MCP tool.

        Returns:
            Async callable ``load_<name>(doc_paths: str) -> str``.
        """
        kb_name = name

        async def load_knowledge_base(doc_paths: str) -> str:
            """Load documents into the knowledge base.

            Args:
                doc_paths: Comma-separated list of file paths or URLs to ingest.
            """
            try:
                paths = [p.strip() for p in doc_paths.split(",") if p.strip()]
                self.load_documents(doc_list=paths)
                return f"Loaded {len(paths)} document(s) into knowledge base '{kb_name}'."
            except Exception as exc:
                return f"Knowledge base load error: {exc}"

        load_knowledge_base.__name__ = f"load_{kb_name.replace('-', '_')}"
        load_knowledge_base.__doc__ = description
        return load_knowledge_base
