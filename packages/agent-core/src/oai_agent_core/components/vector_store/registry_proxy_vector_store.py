"""
Registry Proxy Vector Store.

A drop-in replacement for any BaseVectorStore that delegates
``similarity_search_with_score`` to the KB Registry HTTP API instead of
connecting to a vector database directly.

Used by BaseKnowledgeBaseFactory when an agent YAML references a KB by name::

    knowledge_base:
      - registry_name: insurance_policies

At query time the agent calls::

    POST {KB_REGISTRY_URL}/api/v1/kb-registry/knowledge-bases/{kb_name}/query

This keeps vector-store credentials entirely inside the KB Registry service —
agent configurations never need connection strings, API keys, or paths.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class RegistryProxyVectorStore:
    """Vector store whose searches are proxied to the KB Registry query API.

    Implements the subset of the BaseVectorStore interface that
    BaseKnowledgeBaseFactory.search_knowledge_base() relies on:

    * ``similarity_search_with_score(query, k, filter, **kwargs)``
        → calls ``POST /api/v1/kb-registry/knowledge-bases/{kb_name}/query``
        → returns ``[(Document, score), ...]``

    Document indexing is intentionally NOT supported — use the KB Registry
    UI or API to manage documents.
    """

    def __init__(
        self,
        kb_name: str,
        registry_url: str,
        auth_token: str,
        retrieval_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.kb_name = kb_name
        self.registry_url = registry_url.rstrip("/")
        self.auth_token = auth_token
        self.retrieval_settings = retrieval_settings or {}
        self._query_url = (
            f"{self.registry_url}/api/v1/kb-registry"
            f"/knowledge-bases/{kb_name}/query"
        )
        logger.debug(
            "RegistryProxyVectorStore created for KB '%s' → %s",
            kb_name, self._query_url,
        )

    # ------------------------------------------------------------------
    # Core search — called synchronously from BaseKnowledgeBaseFactory
    # (which runs inside asyncio.to_thread so blocking requests is fine)
    # ------------------------------------------------------------------

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,  # noqa: A002
        **kwargs,
    ) -> List[Tuple[Any, float]]:
        """Proxy the similarity search to the KB Registry query endpoint.

        Args:
            query:  Natural-language search text.
            k:      Maximum number of results to return.
            filter: Metadata filter dict (forwarded as ``filter_metadata``).

        Returns:
            List of ``(langchain Document, similarity_score)`` tuples.
        """
        import requests  # noqa: PLC0415 (stdlib-like, always available)

        payload: Dict[str, Any] = {"query": query, "k": k}

        # Respect score_threshold if stored in retrieval_settings
        score_threshold = self.retrieval_settings.get("score_threshold")
        if score_threshold is not None:
            payload["score_threshold"] = float(score_threshold)

        if filter:
            payload["filter_metadata"] = filter

        logger.debug(
            "RegistryProxyVectorStore.query: kb=%s k=%d query=%r",
            self.kb_name, k, query[:80],
        )

        try:
            response = requests.post(
                self._query_url,
                json=payload,
                headers={"Authorization": f"Bearer {self.auth_token}"},
                timeout=30,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error(
                "KB registry query failed for '%s': %s", self.kb_name, exc
            )
            return []

        data = response.json()
        results = data.get("results", [])

        # Convert registry QueryResult → (langchain Document, score)
        output: List[Tuple[Any, float]] = []
        for item in results:
            doc = _make_document(item, self.kb_name)
            score = float(item.get("score") or 0.0)
            output.append((doc, score))

        logger.debug(
            "RegistryProxyVectorStore: kb=%s returned %d results",
            self.kb_name, len(output),
        )
        return output

    # ------------------------------------------------------------------
    # Stubs for methods BaseKnowledgeBaseFactory might call
    # ------------------------------------------------------------------

    def add_documents(self, documents: list) -> None:  # noqa: ARG002
        raise NotImplementedError(
            f"RegistryProxyVectorStore for '{self.kb_name}' does not support "
            "document indexing. Use the KB Registry API to manage documents."
        )

    def reset_collection(self) -> None:
        """No-op — collection is managed by the KB Registry."""
        logger.debug(
            "RegistryProxyVectorStore.reset_collection called for '%s' — no-op",
            self.kb_name,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_document(item: Dict[str, Any], kb_name: str) -> Any:
    """Convert a KB Registry QueryResult dict to a LangChain Document.

    Falls back to a plain namespace object if langchain_core is not installed.
    """
    content = item.get("content", "")
    metadata = dict(item.get("metadata") or {})
    # Ensure standard keys are present for BaseKnowledgeBaseFactory formatting
    metadata.setdefault("source", item.get("document_name") or kb_name)
    metadata.setdefault("doc_id", item.get("document_id"))
    metadata.setdefault("kb_name", kb_name)

    try:
        from langchain_core.documents import Document  # noqa: PLC0415
        return Document(page_content=content, metadata=metadata)
    except ImportError:
        # Lightweight fallback — only .page_content and .metadata are accessed
        class _Doc:
            def __init__(self, pc: str, meta: dict) -> None:
                self.page_content = pc
                self.metadata = meta
        return _Doc(content, metadata)
