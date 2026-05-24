"""Pinecone vector store — integrated inference (no local embeddings).

The index must be created in the Pinecone console with an *integrated
embedding model* (e.g. ``multilingual-e5-large``, ``llama-text-embed-v2``).
Pinecone embeds the text server-side, so no local embedding function is
needed or used.

SDK requirement: ``pinecone >= 5.0.0``

Required kwargs
---------------
collection_name : str  – stored as metadata on each record.
api_key         : str  – Pinecone project API key.
index_name      : str  – Name of an *existing* Pinecone index with
                          integrated embedding enabled.

Optional kwargs
---------------
namespace : str  – Pinecone namespace that isolates this KB
                   within the index (defaults to ``collection_name``).
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, List, Optional, Tuple
from uuid import uuid4

from langchain_core.documents import Document

from oai_agent_core.core.base_vector_store import BaseVectorStore


class PineconeVectorStore(BaseVectorStore):
    """Pinecone vector store using server-side (integrated) inference."""

    # Text field name stored on every Pinecone record
    _TEXT_FIELD = "text"

    def __init__(self, **kwargs):
        # Pass embedding_function=None — Pinecone embeds server-side
        kwargs.setdefault("embedding_function", None)
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        if not self.collection_name:
            raise ValueError("collection_name is required")

        self._api_key    = kwargs.get("api_key")
        self._index_name = kwargs.get("index_name")
        self._namespace  = kwargs.get("namespace") or self.collection_name

        if not self._api_key:
            raise ValueError("api_key is required for Pinecone vector store")
        if not self._index_name:
            raise ValueError("index_name is required for Pinecone vector store")

        self._index = self._connect()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _connect(self):
        try:
            from pinecone import Pinecone  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "pinecone package is required. "
                "Install it with: pip install 'pinecone>=5.0.0'"
            ) from exc
        pc = Pinecone(api_key=self._api_key)
        return pc.Index(self._index_name)

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_record(doc: Document, rec_id: str) -> dict:
        """Build a dict suitable for ``index.upsert_records``."""
        record = {
            "_id": rec_id,
            "text": doc.page_content,  # field Pinecone embeds
        }
        # Flatten metadata so Pinecone stores it alongside the vector
        for k, v in (doc.metadata or {}).items():
            if k not in ("_id", "text"):
                record[k] = v
        return record

    # Candidate field names that may hold the main text content
    _TEXT_FIELD_CANDIDATES = ("text", "chunk_text", "content", "page_content")

    @staticmethod
    def _hit_to_doc_score(hit) -> Tuple[Document, float]:
        """Convert a Pinecone search hit (dict or object) to (Document, score)."""
        if isinstance(hit, dict):
            fields = dict(hit.get("fields") or {})
            score  = float(hit.get("_score", 0.0))
        else:
            fields = dict(getattr(hit, "fields", None) or {})
            score  = float(getattr(hit, "_score", 0.0))

        # Extract text from whichever field holds it
        text = ""
        for candidate in PineconeVectorStore._TEXT_FIELD_CANDIDATES:
            if candidate in fields:
                text = fields.pop(candidate)
                break

        return Document(page_content=text, metadata=fields), score

    # ------------------------------------------------------------------
    # BaseVectorStore — write
    # ------------------------------------------------------------------

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> None:
        """Upsert documents; Pinecone embeds the text server-side."""
        if not documents:
            return
        if ids is None:
            ids = [str(uuid4()) for _ in documents]

        records = [self._build_record(doc, vid) for doc, vid in zip(documents, ids)]

        # Pinecone recommends batches of ≤96 for upsert_records
        batch = 96
        for i in range(0, len(records), batch):
            self._index.upsert_records(
                namespace=self._namespace,
                records=records[i : i + batch],
            )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        text_list = list(texts)
        if ids is None:
            ids = [str(uuid4()) for _ in text_list]
        meta_list = metadatas or [{} for _ in text_list]
        docs = [Document(page_content=t, metadata=m) for t, m in zip(text_list, meta_list)]
        self.add_documents(docs, ids=ids)
        return ids

    # ------------------------------------------------------------------
    # BaseVectorStore — read
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search using Pinecone's integrated inference (text-in, results-out)."""
        filter_metadata: Optional[dict] = kwargs.get("filter_metadata")

        search_kwargs: dict = {
            "namespace": self._namespace,
            "inputs": {"text": query},
            "top_k": k,
            # No "fields" filter — return all stored fields so we can find
            # the text regardless of what the field is named in this index.
        }
        if filter_metadata:
            search_kwargs["filter"] = filter_metadata

        response = self._index.search(**search_kwargs)

        # SDK may return a plain dict or a response object depending on version
        if isinstance(response, dict):
            hits = (response.get("result") or {}).get("hits", [])
        else:
            try:
                hits = response.result.hits
            except AttributeError:
                hits = []

        return [self._hit_to_doc_score(h) for h in hits]

    def count(self) -> int:
        try:
            stats = self._index.describe_index_stats()
            # SDK may return an object or a dict
            namespaces = getattr(stats, "namespaces", None) or stats.get("namespaces", {})
            ns = namespaces.get(self._namespace)
            if ns is None:
                return 0
            return int(getattr(ns, "vector_count", None) or ns.get("vector_count", 0))
        except Exception as exc:
            self.logger.warning("count() failed: %s", exc)
            return 0

    # ------------------------------------------------------------------
    # BaseVectorStore — delete
    # ------------------------------------------------------------------

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        if not ids:
            return None
        try:
            self._index.delete(ids=ids, namespace=self._namespace)  # kwargs — OK
            return True
        except Exception as exc:
            self.logger.error("delete() failed: %s", exc)
            return False

    def reset_collection(self) -> None:
        """Remove all records in this KB's namespace."""
        try:
            self._index.delete(delete_all=True, namespace=self._namespace)
        except Exception as exc:
            self.logger.error("reset_collection() failed: %s", exc)

    def delete_collection(self) -> None:
        self.reset_collection()

    # ------------------------------------------------------------------
    # Flexible query interface
    # ------------------------------------------------------------------

    def query(
        self,
        query_text: Optional[str] = None,
        filter_metadata: Optional[dict] = None,
        n_results: int = 4,
        order_by: Optional[str] = None,
        order: str = "desc",
        **kwargs: Any,
    ) -> List[Document]:
        if query_text:
            pairs = self.similarity_search_with_score(
                query_text, k=n_results, filter_metadata=filter_metadata
            )
            return [doc for doc, _ in pairs]
        return []

    # ------------------------------------------------------------------
    # LangChain compat
    # ------------------------------------------------------------------

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding=None,          # ignored — Pinecone embeds server-side
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "PineconeVectorStore":
        store = cls(**kwargs)
        store.add_texts(texts, metadatas=metadatas)
        return store
