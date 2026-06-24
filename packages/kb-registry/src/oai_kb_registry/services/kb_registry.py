"""
Knowledge Base Registry — core service layer.

Orchestrates:
  - Registering / deleting knowledge bases
  - Indexing documents (upload, S3, URL) into the configured vector store
  - Running semantic queries against indexed knowledge bases
  - Reindexing entire knowledge bases
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from oai_kb_registry.services.db.database_logger import KBDatabaseLogger
from oai_kb_registry.services.document_processor import DocumentProcessor
from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory

if TYPE_CHECKING:
    from oai_agent_core.core.base_vector_store import BaseVectorStore


logger = logging.getLogger(__name__)

# Thread pool for CPU-bound embedding / chunking work (keeps FastAPI's event
# loop free during heavy indexing operations)
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="kb-index")


class KBRegistry:
    """Top-level registry service used by all FastAPI route handlers."""

    def __init__(
        self,
        db_logger: KBDatabaseLogger,
        log: Optional[logging.Logger] = None,
        auto_start_infra: bool = False,
        infra_compose_file: Optional[str] = None,
        infra_startup_timeout: int = 60,
    ) -> None:
        self.db = db_logger
        self.logger = log or logging.getLogger(__name__)
        # Cache of live vector store instances keyed by kb_name
        self._vector_stores: Dict[str, "BaseVectorStore"] = {}
        # Infra provisioning settings — forwarded from dependencies.py
        self._auto_start_infra       = auto_start_infra
        self._infra_compose_file     = infra_compose_file
        self._infra_startup_timeout  = infra_startup_timeout
        # Tracks which on-demand vector store services (pgvector, chromadb)
        # THIS process started, so close() knows what to shut down.
        self._vector_services_started: set[str] = set()

    async def close(self) -> None:
        """Shut down the registry in reverse startup order.

        Sequence
        --------
        1. Clear the in-memory vector store cache (no external calls).
        2. Stop any on-demand vector store containers this process started
           (pgvector, chromadb) — Phase 2 infra.
        3. Close the metadata DB connection.

        The core infra (postgres, valkey) is stopped separately by
        ``dependencies.close_registry()`` after this method returns —
        that is Phase 1 infra and should outlive the DB connection.
        """
        # 1. Drop the in-memory cache first so no new queries can be issued
        #    against vector stores we're about to shut down.
        self._vector_stores.clear()
        self.logger.debug("close: vector store cache cleared")

        # 2. Stop on-demand vector store containers (pgvector / chromadb).
        if self._vector_services_started:
            await self._stop_vector_store_infra()

        # 3. Close the metadata DB connection.
        await self.db.close()
        self.logger.debug("close: metadata DB connection closed")

    # ------------------------------------------------------------------ #
    #  Vector store cache helpers                                          #
    # ------------------------------------------------------------------ #

    async def _get_or_create_vector_store(self, kb_name: str) -> "BaseVectorStore":
        """Return a cached vector store for ``kb_name``, creating it if needed."""
        if kb_name in self._vector_stores:
            return self._vector_stores[kb_name]

        kb = await self.db.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_name!r}")

        raw_configs = await self.db.get_kb_configs(kb["id"])
        configs = VectorStoreProviderFactory.decrypt_configs(raw_configs)

        vs = VectorStoreProviderFactory.create_vector_store(
            kb_name=kb_name,
            vector_db_type=kb["vector_db_type"],
            deployment_mode=kb["deployment_mode"],
            embedding_model_id=kb.get("embedding_model_id"),
            embedding_region=kb.get("embedding_region"),
            configs=configs,
        )
        self._vector_stores[kb_name] = vs
        return vs

    def _invalidate_cache(self, kb_name: str) -> None:
        self._vector_stores.pop(kb_name, None)

    # ------------------------------------------------------------------ #
    #  On-demand vector store infrastructure provisioning                 #
    # ------------------------------------------------------------------ #

    async def _ensure_vector_store_infra(
        self,
        vector_db_type: str,
        deployment_mode: str,
    ) -> None:
        """Start the required vector store container for builtin deployments.

        ``deployment_mode="builtin"`` means the registry manages the container
        lifecycle — always start it when needed, regardless of the
        ``auto_start_infra`` flag (which only controls Phase 1 startup of the
        *metadata* postgres + valkey services).

        The call is a no-op when:
          - deployment_mode is "external"  — user manages their own infra
          - vector_db_type is "s3"         — no container needed
        """
        if deployment_mode != "builtin":
            self.logger.debug(
                "_ensure_vector_store_infra: skipped — deployment_mode=%r (external)",
                deployment_mode,
            )
            return

        try:
            from oai_kb_registry.services.infra_manager import (  # noqa: PLC0415
                InfraManager,
                _BUNDLED_COMPOSE,
            )
            from pathlib import Path  # noqa: PLC0415

            # s3 type has no entry in _VECTOR_SERVICE_MAP — no-op
            service = InfraManager._VECTOR_SERVICE_MAP.get(vector_db_type.lower())
            if service is None:
                self.logger.debug(
                    "_ensure_vector_store_infra: no container needed for vector_db_type=%r",
                    vector_db_type,
                )
                return

            compose_file = (
                Path(self._infra_compose_file)
                if self._infra_compose_file
                else None
            )
            await InfraManager.ensure_vector_store_service(
                vector_db_type=vector_db_type,
                compose_file=compose_file,
                timeout=self._infra_startup_timeout,
            )
            # Record the compose service name so close() can stop it later.
            self._vector_services_started.add(service)
            self.logger.debug(
                "_ensure_vector_store_infra: recorded '%s' as started by this process",
                service,
            )
        except ImportError:
            self.logger.warning(
                "_ensure_vector_store_infra: InfraManager not available — "
                "make sure Docker is installed if using builtin deployment"
            )
        except Exception as exc:
            # Non-fatal: log a warning and let the vector store connection
            # attempt proceed — it may still succeed if the service was
            # already started externally.
            self.logger.warning(
                "_ensure_vector_store_infra: failed to start container "
                "for vector_db_type=%r: %s",
                vector_db_type, exc,
            )

    async def _stop_vector_store_infra(self) -> None:
        """Stop vector store containers that this process started on-demand.

        Called from :meth:`close` before the metadata DB is closed.
        Non-fatal: a failure to stop a container is logged as a warning so
        it never prevents the rest of the shutdown sequence from running.
        """
        if not self._vector_services_started:
            return

        services = sorted(self._vector_services_started)
        self.logger.info(
            "auto_stop_infra: stopping on-demand vector store services: %s", services
        )

        try:
            from oai_kb_registry.services.infra_manager import InfraManager  # noqa: PLC0415
            from pathlib import Path  # noqa: PLC0415

            compose_file = (
                Path(self._infra_compose_file)
                if self._infra_compose_file
                else None
            )
            # Run the synchronous docker compose stop in a thread so we don't
            # block the event loop during shutdown.
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: InfraManager.stop_services(
                    compose_file=compose_file,
                    services=services,
                ),
            )
            self._vector_services_started.clear()
            self.logger.info(
                "auto_stop_infra: vector store services stopped successfully: %s", services
            )
        except ImportError:
            self.logger.warning(
                "_stop_vector_store_infra: InfraManager not available — "
                "containers left running"
            )
        except Exception as exc:
            self.logger.warning(
                "_stop_vector_store_infra: failed to stop services %s: %s",
                services, exc,
            )

    # ------------------------------------------------------------------ #
    #  Startup restoration                                                 #
    # ------------------------------------------------------------------ #

    async def restore_from_db(self) -> None:
        """Re-initialise all persisted knowledge bases after a service restart.

        For each KB stored in the database:
          1. Ensure its vector store container is running (builtin KBs only,
             and only when ``auto_start_infra`` is enabled).
          2. Warm up the in-memory vector store cache so the first query after
             restart is not slower than subsequent ones.

        Failures for individual KBs are logged and skipped — a broken KB must
        not prevent healthy KBs from being restored.
        """
        kbs = await self.db.get_all_knowledge_bases()
        if not kbs:
            self.logger.info("restore_from_db: no knowledge bases found in database")
            return

        self.logger.info("restore_from_db: restoring %d knowledge base(s)…", len(kbs))
        restored = 0
        failed = 0

        for kb in kbs:
            kb_name = kb.get("name")
            if not kb_name:
                continue
            try:
                # 1. Ensure vector store container is running (no-op if
                #    auto_start_infra=False or deployment_mode is external/s3)
                await self._ensure_vector_store_infra(
                    kb.get("vector_db_type", "chroma"),
                    kb.get("deployment_mode", "builtin"),
                )

                # 2. Warm up the vector store connection
                await self._get_or_create_vector_store(kb_name)
                self.logger.info("restore_from_db: ✓ %s", kb_name)
                restored += 1
            except Exception as exc:
                self.logger.warning(
                    "restore_from_db: ✗ %s — will retry on first use (%s)",
                    kb_name, exc,
                )
                failed += 1

        self.logger.info(
            "restore_from_db: complete — %d restored, %d failed",
            restored, failed,
        )

    # ------------------------------------------------------------------ #
    #  Knowledge base registration                                         #
    # ------------------------------------------------------------------ #

    async def register_knowledge_base(
        self,
        name: str,
        description: Optional[str],
        tags: List[str],
        vector_db_type: str,
        deployment_mode: str,
        embedding_model_id: str,
        chunk_size: int,
        chunk_overlap: int,
        vector_db_config: Optional[Dict[str, Any]] = None,
        embedding_region: Optional[str] = None,
        retrieval_config: Optional[Dict[str, Any]] = None,
        performed_by: str = "system",
    ) -> Dict[str, Any]:
        """Register a new knowledge base (or update if it already exists)."""

        kb = await self.db.create_knowledge_base(
            name=name,
            description=description,
            tags=tags,
            vector_db_type=vector_db_type,
            deployment_mode=deployment_mode,
            embedding_model_id=embedding_model_id,
            embedding_region=embedding_region,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            retrieval_config=retrieval_config,
        )

        kb_id = kb.get("id")
        if kb_id:
            # Persist encrypted vector DB connection config
            if vector_db_config:
                encrypted = VectorStoreProviderFactory.encrypt_configs(vector_db_config)
                for k, v in encrypted.items():
                    await self.db.upsert_kb_config(kb_id, k, v)

            await self.db.log_action(kb_id, "register", performed_by, f"Registered {name}")

        # Ensure the required vector store container is running (builtin only)
        await self._ensure_vector_store_infra(vector_db_type, deployment_mode)

        # Warm up the vector store connection on registration
        try:
            self._invalidate_cache(name)
            await self._get_or_create_vector_store(name)
            self.logger.info("Vector store initialised for KB: %s", name)
        except Exception as exc:
            self.logger.warning(
                "Vector store warm-up failed for KB %s (will retry on first use): %s",
                name, exc,
            )
            if kb_id:
                await self.db.update_kb_status(name, "error")
                await self.db.log_action(
                    kb_id, "init_error", performed_by,
                    f"Vector store init failed: {exc}",
                )

        return kb

    async def delete_knowledge_base(
        self,
        name: str,
        performed_by: str = "system",
    ) -> None:
        """Delete a knowledge base and all its indexed documents."""
        kb = await self.db.get_knowledge_base(name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {name!r}")

        kb_id = kb["id"]

        # Drop the vector store collection first
        try:
            vs = await self._get_or_create_vector_store(name)
            vs.reset_collection()
        except Exception as exc:
            self.logger.warning("Could not drop vector store for %s: %s", name, exc)

        self._invalidate_cache(name)

        # Cascade delete through DB (FK constraints handle docs + configs)
        await self.db.delete_knowledge_base(name)
        self.logger.info("Deleted KB: %s", name)

    # ------------------------------------------------------------------ #
    #  Document indexing                                                   #
    # ------------------------------------------------------------------ #

    async def _index_documents_in_thread(
        self,
        kb_name: str,
        doc_id: int,
        documents,  # List[Document]
        vs: "BaseVectorStore",
    ) -> int:
        """Run CPU-bound embedding + vector insertion in thread pool."""
        loop = asyncio.get_event_loop()

        def _do_add():
            vs.add_documents(documents)
            return len(documents)

        return await loop.run_in_executor(_executor, _do_add)

    async def _assert_ingestable(self, kb_name: str) -> None:
        """Reject document ingestion for read-only graph KBs (managed externally)."""
        kb = await self.db.get_knowledge_base(kb_name)
        if kb and str(kb.get("vector_db_type", "")).lower() in (
            "neo4j_graph", "neo4j_kg", "neo4j",
        ):
            raise ValueError(
                f"Document ingestion is not supported for Neo4j knowledge graph "
                f"'{kb_name}'. The graph is managed by an external pipeline; the "
                f"registry only queries it."
            )

    async def index_document_from_upload(
        self,
        kb_name: str,
        filename: str,
        data: bytes,
        performed_by: str = "system",
    ) -> Dict[str, Any]:
        """Index a file uploaded via the API."""
        await self._assert_ingestable(kb_name)
        kb = await self.db.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_name!r}")

        kb_id = kb["id"]

        # Create pending DB record
        doc_record = await self.db.create_document(
            kb_id=kb_id,
            name=filename,
            source_type="upload",
            source_uri=None,
            file_size_bytes=len(data),
        )
        doc_id = doc_record.get("id")

        await self.db.update_document_status(doc_id, "indexing")

        try:
            processor = DocumentProcessor(
                chunk_size=kb["chunk_size"],
                chunk_overlap=kb["chunk_overlap"],
            )
            vs = await self._get_or_create_vector_store(kb_name)
            documents = processor.process_upload(data, filename, kb_name, doc_id)
            chunk_count = await self._index_documents_in_thread(kb_name, doc_id, documents, vs)
            await self.db.update_document_status(doc_id, "indexed", chunk_count)
            await self.db.log_action(
                kb_id, "add_document", performed_by,
                f"Indexed {filename} ({chunk_count} chunks)",
            )
            self.logger.info("Indexed %s → %s (%d chunks)", filename, kb_name, chunk_count)
        except Exception as exc:
            self.logger.error("Failed to index %s: %s", filename, exc)
            await self.db.update_document_status(doc_id, "failed", error_message=str(exc))
            await self.db.log_action(kb_id, "index_error", performed_by, str(exc))
            raise RuntimeError(f"Indexing failed: {exc}") from exc

        return await self.db.get_document(doc_id) or {}

    async def index_document_from_s3(
        self,
        kb_name: str,
        bucket: str,
        key: str,
        performed_by: str = "system",
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Index an object already stored in S3."""
        await self._assert_ingestable(kb_name)
        kb = await self.db.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_name!r}")

        kb_id = kb["id"]
        filename = key.split("/")[-1]
        source_uri = f"s3://{bucket}/{key}"

        doc_record = await self.db.create_document(
            kb_id=kb_id,
            name=filename,
            source_type="s3",
            source_uri=source_uri,
            file_size_bytes=None,
        )
        doc_id = doc_record.get("id")

        await self.db.update_document_status(doc_id, "indexing")

        try:
            processor = DocumentProcessor(
                chunk_size=kb["chunk_size"],
                chunk_overlap=kb["chunk_overlap"],
            )
            vs = await self._get_or_create_vector_store(kb_name)
            documents = processor.process_s3_object(
                bucket=bucket,
                key=key,
                kb_name=kb_name,
                doc_id=doc_id,
                region=region,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
            )
            chunk_count = await self._index_documents_in_thread(kb_name, doc_id, documents, vs)
            await self.db.update_document_status(doc_id, "indexed", chunk_count)
            await self.db.log_action(
                kb_id, "add_document", performed_by,
                f"Indexed S3 object {source_uri} ({chunk_count} chunks)",
            )
        except Exception as exc:
            self.logger.error("Failed to index S3 object %s: %s", source_uri, exc)
            await self.db.update_document_status(doc_id, "failed", error_message=str(exc))
            await self.db.log_action(kb_id, "index_error", performed_by, str(exc))
            raise RuntimeError(f"S3 indexing failed: {exc}") from exc

        return await self.db.get_document(doc_id) or {}

    async def index_document_from_text(
        self,
        kb_name: str,
        name: str,
        text: str,
        source_type: str = "external",
        source_uri: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        performed_by: str = "system",
    ) -> Dict[str, Any]:
        """Index pre-extracted text (e.g. from a LangChain loader) into the KB.

        Used by the source sync service — each LangChain Document becomes one
        ``kb_documents`` record that is chunked, embedded, and stored.
        """
        await self._assert_ingestable(kb_name)
        kb = await self.db.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_name!r}")

        kb_id = kb["id"]

        doc_record = await self.db.create_document(
            kb_id=kb_id,
            name=name,
            source_type=source_type,
            source_uri=source_uri,
            file_size_bytes=len(text.encode("utf-8", errors="replace")),
        )
        doc_id = doc_record.get("id")

        await self.db.update_document_status(doc_id, "indexing")

        try:
            processor = DocumentProcessor(
                chunk_size=kb["chunk_size"],
                chunk_overlap=kb["chunk_overlap"],
            )
            vs = await self._get_or_create_vector_store(kb_name)
            documents = processor.process_text(
                text=text,
                name=name,
                kb_name=kb_name,
                doc_id=doc_id,
                source_type=source_type,
                extra_metadata=extra_metadata,
            )
            chunk_count = await self._index_documents_in_thread(kb_name, doc_id, documents, vs)
            await self.db.update_document_status(doc_id, "indexed", chunk_count)
            await self.db.log_action(
                kb_id, "add_document", performed_by,
                f"Indexed {name!r} from {source_type} ({chunk_count} chunks)",
            )
            self.logger.info("Indexed %s → %s (%d chunks)", name, kb_name, chunk_count)
        except Exception as exc:
            self.logger.error("Failed to index text doc %s: %s", name, exc)
            await self.db.update_document_status(doc_id, "failed", error_message=str(exc))
            await self.db.log_action(kb_id, "index_error", performed_by, str(exc))
            raise RuntimeError(f"Indexing failed: {exc}") from exc

        return await self.db.get_document(doc_id) or {}

    async def remove_document(
        self,
        kb_name: str,
        doc_id: int,
        performed_by: str = "system",
    ) -> None:
        """Remove a document record from the DB (vector data is NOT deleted — see note)."""
        kb = await self.db.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_name!r}")

        doc = await self.db.get_document(doc_id)
        if not doc or doc["kb_id"] != kb["id"]:
            raise ValueError(f"Document {doc_id} not found in KB {kb_name!r}")

        # Note: vector stores don't all support fine-grained per-document deletion.
        # We delete the DB record and let the next reindex rebuild the collection.
        await self.db.delete_document(doc_id)
        await self.db.log_action(
            kb["id"], "remove_document", performed_by,
            f"Removed document id={doc_id} ({doc.get('name', '?')})",
        )

    # ------------------------------------------------------------------ #
    #  Query                                                               #
    # ------------------------------------------------------------------ #

    async def query(
        self,
        kb_name: str,
        query_text: str,
        k: int = 5,
        score_threshold: Optional[float] = None,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Run a semantic similarity search and return scored results."""
        loop = asyncio.get_event_loop()
        vs = await self._get_or_create_vector_store(kb_name)

        def _do_search():
            results = vs.similarity_search_with_score(
                query_text, k=k, filter=filter_metadata
            )
            return results

        raw = await loop.run_in_executor(_executor, _do_search)

        output: List[Dict[str, Any]] = []
        for doc, score in raw:
            if score_threshold is not None and score < score_threshold:
                continue
            output.append(
                {
                    "content": doc.page_content,
                    "score": round(float(score), 4),
                    "metadata": doc.metadata,
                    "document_name": doc.metadata.get("doc_name"),
                    "document_id": doc.metadata.get("doc_id"),
                }
            )
        return output

    # ------------------------------------------------------------------ #
    #  Reindex                                                             #
    # ------------------------------------------------------------------ #

    async def reindex_knowledge_base(
        self,
        kb_name: str,
        performed_by: str = "system",
    ) -> int:
        """Drop and rebuild the vector store from all indexed documents.

        Returns the number of documents queued for re-indexing.
        """
        kb = await self.db.get_knowledge_base(kb_name)
        if not kb:
            raise ValueError(f"Knowledge base not found: {kb_name!r}")

        kb_id = kb["id"]
        docs = await self.db.get_kb_documents(kb_id)
        indexed_docs = [d for d in docs if d["status"] == "indexed"]

        if not indexed_docs:
            self.logger.info("No indexed documents to reindex for KB: %s", kb_name)
            return 0

        # Clear existing vector store collection
        try:
            vs = await self._get_or_create_vector_store(kb_name)
            vs.reset_collection()
        except Exception as exc:
            self.logger.warning("Could not reset vector store for %s: %s", kb_name, exc)

        self._invalidate_cache(kb_name)

        await self.db.log_action(
            kb_id, "reindex_start", performed_by,
            f"Starting reindex of {len(indexed_docs)} documents",
        )

        # Mark all docs as pending
        for doc in indexed_docs:
            await self.db.update_document_status(doc["id"], "pending")

        # Fire-and-forget background task
        asyncio.create_task(
            self._reindex_all_documents(kb_name, kb, indexed_docs, performed_by)
        )

        return len(indexed_docs)

    async def _reindex_all_documents(
        self,
        kb_name: str,
        kb: Dict[str, Any],
        docs: List[Dict[str, Any]],
        performed_by: str,
    ) -> None:
        """Background coroutine that reindexes all documents one by one."""
        kb_id = kb["id"]
        processor = DocumentProcessor(
            chunk_size=kb["chunk_size"],
            chunk_overlap=kb["chunk_overlap"],
        )
        success = 0
        failed  = 0

        for doc in docs:
            doc_id = doc["id"]
            try:
                vs = await self._get_or_create_vector_store(kb_name)
                await self.db.update_document_status(doc_id, "indexing")

                if doc["source_type"] == "upload":
                    self.logger.warning(
                        "Reindex of upload doc id=%s skipped — bytes no longer available. "
                        "Re-upload the file to reindex.",
                        doc_id,
                    )
                    await self.db.update_document_status(
                        doc_id, "failed",
                        error_message="Reindex of upload: original file bytes no longer stored.",
                    )
                    failed += 1
                    continue

                if doc["source_type"] == "s3":
                    uri = doc.get("source_uri", "")
                    # Parse s3://bucket/key
                    parts = uri.replace("s3://", "").split("/", 1)
                    bucket = parts[0]
                    key    = parts[1] if len(parts) > 1 else ""
                    raw_configs = await self.db.get_kb_configs(kb_id)
                    configs = VectorStoreProviderFactory.decrypt_configs(raw_configs)
                    documents = processor.process_s3_object(
                        bucket=bucket, key=key, kb_name=kb_name, doc_id=doc_id,
                        aws_access_key_id=configs.get("aws_access_key_id"),
                        aws_secret_access_key=configs.get("aws_secret_access_key"),
                    )
                else:
                    self.logger.warning("Unsupported source_type for reindex: %s", doc["source_type"])
                    continue

                chunk_count = await self._index_documents_in_thread(kb_name, doc_id, documents, vs)
                await self.db.update_document_status(doc_id, "indexed", chunk_count)
                success += 1

            except Exception as exc:
                self.logger.error("Reindex failed for doc id=%s: %s", doc_id, exc)
                await self.db.update_document_status(doc_id, "failed", error_message=str(exc))
                failed += 1

        await self.db.log_action(
            kb_id, "reindex_complete", performed_by,
            f"Reindex complete: {success} succeeded, {failed} failed",
        )
        self.logger.info(
            "Reindex finished for KB %s: %d succeeded, %d failed",
            kb_name, success, failed,
        )
