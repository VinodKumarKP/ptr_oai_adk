"""Neo4j knowledge-graph store (GraphRAG) — read-only.

Phase 1 scope
-------------
* **traversal** (default): find entry nodes via a Neo4j full-text index (or a
  property-CONTAINS fallback), then expand an N-hop neighbourhood and serialise
  the resulting sub-graph to text.  No embeddings required.
* **text2cypher** (opt-in): an LLM turns the natural-language question into
  Cypher using the *introspected* graph schema; the query is safety-checked and
  executed read-only.  Uses ``langchain-neo4j`` (``GraphCypherQAChain``).

The graph is assumed to be **already loaded** by a separate, governed pipeline —
this store never writes.  Hybrid (vector + graph) retrieval is planned for
phase 2.

This store satisfies the :class:`BaseVectorStore` seam
(``similarity_search_with_score`` → ``[(Document, score)]``) so it drops into the
existing ``VectorStoreFactory`` / ``BaseKnowledgeBaseFactory`` pipeline unchanged.
Scores are returned already normalised to ``0..1``; the store sets
``returns_normalized_scores = True`` so the factory skips its distance-to-score
conversion.

SDK requirements
----------------
neo4j >= 5.x            – raw driver: introspection, full-text entry, traversal
langchain-neo4j         – schema + Text2Cypher retriever (text2cypher mode only)

Required settings
-----------------
url       : str  – bolt/neo4j URI, e.g. ``bolt://localhost:7687``
username  : str  – read-only DB user is strongly recommended
password  : str  – supports ``${ENV_VAR}`` expansion

Optional settings
-----------------
database          : str   – Neo4j database name (default ``neo4j``)
retrieval_mode    : str   – ``traversal`` (default) | ``text2cypher``
entry_strategy    : str   – ``fulltext`` (default) | ``entity_linking``
fulltext_index    : str   – name of an existing full-text index (entry nodes)
max_hops          : int   – neighbourhood radius for traversal (default 2)
rel_limit         : int   – max relationships pulled per entry node (default 50)
node_text_props   : list  – node properties tried for a display name
query_timeout     : float – per-query server timeout in seconds (default 30)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

from oai_agent_core.core.base_vector_store import BaseVectorStore

# Cypher clauses that mutate the graph — rejected on every executed query.
_WRITE_CLAUSE = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|FOREACH|LOAD\s+CSV|"
    r"CALL\s+(?:dbms|apoc\.(?:create|merge|refactor))|"
    r"CREATE\s+(?:INDEX|CONSTRAINT))\b",
    re.IGNORECASE,
)

_DEFAULT_TEXT_PROPS = ("name", "title", "id", "description", "text")

# Hardened Cypher-generation prompt: forces read-only, schema-bound queries and
# case-insensitive partial matching so name mismatches (e.g. "Home Shield" vs
# "Home Shield 360") don't silently return zero rows.
_CYPHER_GENERATION_TEMPLATE = """You are a Neo4j Cypher expert. Generate ONE read-only \
Cypher query that answers the question using ONLY the schema provided.

Schema:
{schema}

Rules:
- Read-only ONLY. Never use CREATE, MERGE, SET, DELETE, REMOVE, DROP, or LOAD CSV.
- Use only the node labels, relationship types and properties present in the schema.
- Match names and text case-insensitively and partially. Use
  `toLower(n.name) CONTAINS toLower("term")` — NEVER exact equality (`=`) on
  name/title/description properties, because the stored value may be longer or
  differently cased than the term in the question.
- Return the specific properties needed to answer (e.g. names, amounts), not whole nodes.
- Do not invent labels, relationships or properties that are not in the schema.
- Output only the Cypher query, with no explanation or code fences.

Question: {question}
Cypher query:"""

# Hardened QA prompt: an empty result must NOT become a confident negative
# ("not covered"). This is the safety-critical guard for the bank use case.
_QA_TEMPLATE = """You answer questions using ONLY the query results given as context.

Context (Cypher query results):
{context}

Rules:
- Base the answer strictly on the context above.
- If the context is empty or does not contain the requested information, say you
  could not find it in the knowledge graph. NEVER state that something does not
  exist, is "not covered", or is "not available" merely because the result is empty.
- Be concise and mention the entities/relationships the answer is based on.

Question: {question}
Answer:"""


class GraphReadOnlyError(RuntimeError):
    """Raised when a write operation is attempted on the read-only graph store."""


class UnsafeCypherError(ValueError):
    """Raised when a Cypher statement contains a mutating clause."""


class Neo4jGraphStore(BaseVectorStore):
    """Read-only Neo4j knowledge-graph retriever (GraphRAG, phase 1)."""

    def __init__(self, **kwargs: Any):
        # Graph retrieval needs no local embeddings in phase 1.
        kwargs.setdefault("embedding_function", None)
        super().__init__(**kwargs)
        self.logger = logging.getLogger(__name__)

        self._url = self._resolve(kwargs.get("url"))
        self._username = self._resolve(kwargs.get("username"))
        self._password = self._resolve(kwargs.get("password"))
        self._database = self._resolve(kwargs.get("database")) or "neo4j"

        if not self._url:
            raise ValueError("url is required for Neo4j graph store")

        self._mode = (kwargs.get("retrieval_mode") or "traversal").lower()
        self._entry_strategy = (kwargs.get("entry_strategy") or "fulltext").lower()
        self._fulltext_index = kwargs.get("fulltext_index")
        self._max_hops = max(1, int(kwargs.get("max_hops", 2)))
        self._rel_limit = int(kwargs.get("rel_limit", 50))
        self._text_props = tuple(kwargs.get("node_text_props") or _DEFAULT_TEXT_PROPS)
        self._timeout = float(kwargs.get("query_timeout", 30))
        self._llm = kwargs.get("llm")
        # Optional region for a bare model-id LLM (e.g. Bedrock); used by LiteLLM.
        self._llm_region = kwargs.get("llm_region")

        # Vector entry (hybrid, node-level): a separate vector index holds one
        # embedding per graph node; its hits give the entry node ids we traverse
        # from. Built lazily from `vector_entry` config using this store's
        # embedding_function. See _vector_entry_nodes.
        self._vector_entry_config = kwargs.get("vector_entry")
        self._entry_id_field = kwargs.get("entry_id_field", "node_id")
        self._entry_id_property = kwargs.get("entry_id_property", "id")
        self._vector_entry_store = None  # lazily created
        if self._entry_strategy == "vector" and not self._vector_entry_config:
            raise ValueError(
                "entry_strategy 'vector' requires a 'vector_entry' store config "
                "(the node-level embedding index)."
            )

        # The factory reads this to skip distance→similarity normalization.
        self.returns_normalized_scores = True

        self._driver = self._connect()
        self._lc_graph = None      # lazily-created langchain-neo4j Neo4jGraph
        self._schema_cache: Optional[str] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(value: Any) -> Any:
        """Expand ``${VAR}`` / ``$VAR`` env references in string settings."""
        if isinstance(value, str) and "$" in value:
            return os.path.expandvars(value)
        return value

    def _connect(self):
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "neo4j package is required. Install it with: "
                "pip install 'oai-agent-core[neo4j]'"
            ) from exc
        auth = (self._username, self._password) if self._username else None
        return GraphDatabase.driver(self._url, auth=auth)

    def _assert_read_only(self, cypher: str) -> None:
        if _WRITE_CLAUSE.search(cypher or ""):
            raise UnsafeCypherError(
                f"Refusing to run a mutating Cypher statement on the read-only "
                f"knowledge graph: {cypher!r}"
            )

    def _read(self, cypher: str, params: Optional[dict] = None) -> List[dict]:
        """Run a guarded, read-only Cypher query and return records as dicts."""
        from neo4j import READ_ACCESS  # noqa: PLC0415

        self._assert_read_only(cypher)
        with self._driver.session(
            database=self._database, default_access_mode=READ_ACCESS
        ) as session:
            tx = session.begin_transaction(timeout=self._timeout)
            try:
                result = tx.run(cypher, params or {})
                records = [r for r in result]
                tx.commit()
            except Exception:
                tx.rollback()
                raise
        return records

    @staticmethod
    def _node_label(node) -> str:
        labels = list(getattr(node, "labels", []) or [])
        return labels[0] if labels else "Node"

    def _node_display(self, node) -> str:
        for prop in self._text_props:
            if prop in node and node[prop] not in (None, ""):
                return str(node[prop])
        return self._node_label(node)

    # ------------------------------------------------------------------
    # Schema introspection (cached) — via langchain-neo4j
    # ------------------------------------------------------------------

    def _graph(self):
        """Lazily create and cache a langchain-neo4j Neo4jGraph."""
        if self._lc_graph is None:
            try:
                from langchain_neo4j import Neo4jGraph  # noqa: PLC0415
            except ImportError as exc:
                raise RuntimeError(
                    "langchain-neo4j is required for schema introspection / "
                    "text2cypher. Install it with: pip install 'oai-agent-core[neo4j]'"
                ) from exc
            self._lc_graph = Neo4jGraph(
                url=self._url,
                username=self._username,
                password=self._password,
                database=self._database,
            )
        return self._lc_graph

    def get_schema(self, refresh: bool = False) -> str:
        """Return the introspected graph schema (labels, rel types, properties)."""
        if self._schema_cache is None or refresh:
            graph = self._graph()
            if refresh:
                graph.refresh_schema()
            self._schema_cache = graph.schema
        return self._schema_cache

    # ------------------------------------------------------------------
    # Entry-node discovery
    # ------------------------------------------------------------------

    def _entry_nodes(self, query: str, k: int) -> List[Tuple[Any, float]]:
        """Find candidate entry nodes for a query → [(node, raw_score)]."""
        if self._entry_strategy == "vector":
            return self._vector_entry_nodes(query, k)

        if self._entry_strategy == "entity_linking":
            terms = self._extract_entities(query) or [query]
        else:
            terms = [query]

        hits: List[Tuple[Any, float]] = []
        seen = set()
        per_term = max(1, k // len(terms)) + 1
        for term in terms:
            for node, score in self._lookup_nodes(term, per_term):
                eid = node.element_id
                if eid not in seen:
                    seen.add(eid)
                    hits.append((node, score))
        hits.sort(key=lambda t: t[1], reverse=True)
        return hits[:k]

    # ------------------------------------------------------------------
    # Vector entry (hybrid, node-level): semantic search → entry node ids
    # ------------------------------------------------------------------

    def _get_vector_entry_store(self):
        """Lazily build the node-level vector index from `vector_entry` config."""
        if self._vector_entry_store is None:
            from oai_agent_core.components.vector_store.vector_store_factory import (  # noqa: PLC0415
                VectorStoreFactory,
            )
            cfg = self._vector_entry_config or {}
            self._vector_entry_store = VectorStoreFactory.create_vector_store(
                cfg.get("type", "postgres"),
                embedding_function=self.embedding_function,
                **(cfg.get("settings", {})),
            )
        return self._vector_entry_store

    def _fetch_node_by_id(self, node_id: Any):
        """Fetch a single graph node by its shared-id property (read-only)."""
        # Dynamic property access via parameter — no string interpolation, so no
        # injection risk. An index on the id property is recommended for scale.
        cypher = "MATCH (n) WHERE n[$prop] = $id RETURN n LIMIT 1"
        rows = self._read(cypher, {"prop": self._entry_id_property, "id": node_id})
        return rows[0]["n"] if rows else None

    def _vector_entry_nodes(self, query: str, k: int) -> List[Tuple[Any, float]]:
        """Semantic entry: vector-search node embeddings → resolve to graph nodes.

        Each vector hit carries the shared node id in ``metadata[entry_id_field]``;
        we resolve it to the live graph node and traverse from there. Scores are
        rank-based (best hit first) so they are metric-agnostic.
        """
        store = self._get_vector_entry_store()
        hits = store.similarity_search_with_score(query, k=k)

        results: List[Tuple[Any, float]] = []
        seen = set()
        total = len(hits)
        for rank, (doc, _score) in enumerate(hits):
            node_id = (doc.metadata or {}).get(self._entry_id_field)
            if node_id is None:
                self.logger.warning(
                    "Vector entry hit missing '%s' in metadata; skipping.",
                    self._entry_id_field,
                )
                continue
            node = self._fetch_node_by_id(node_id)
            if node is None:
                self.logger.warning(
                    "Vector entry id %r not found in graph (property '%s').",
                    node_id, self._entry_id_property,
                )
                continue
            if node.element_id in seen:
                continue
            seen.add(node.element_id)
            results.append((node, float(total - rank)))  # rank-descending score
        return results

    def _lookup_nodes(self, term: str, k: int) -> List[Tuple[Any, float]]:
        """Look up entry nodes for a single term (fulltext index or CONTAINS)."""
        if self._fulltext_index:
            cypher = (
                "CALL db.index.fulltext.queryNodes($index, $q) YIELD node, score "
                f"RETURN node, score LIMIT {int(k)}"
            )
            try:
                rows = self._read(cypher, {"index": self._fulltext_index, "q": term})
                return [(r["node"], float(r["score"])) for r in rows]
            except Exception as exc:  # index missing / query error → fallback
                self.logger.warning(
                    "Full-text lookup on index '%s' failed (%s); "
                    "falling back to property scan.",
                    self._fulltext_index, exc,
                )

        # Fallback: case-insensitive CONTAINS over common display properties.
        props = " OR ".join(
            f"toLower(toString(n.`{p}`)) CONTAINS toLower($q)" for p in self._text_props
        )
        cypher = f"MATCH (n) WHERE {props} RETURN n AS node LIMIT {int(k)}"
        rows = self._read(cypher, {"q": term})
        return [(r["node"], 1.0) for r in rows]

    def _extract_entities(self, query: str) -> List[str]:
        """Use the LLM (if available) to pull entity mentions from the question."""
        if not self._llm:
            return []
        prompt = (
            "Extract the key named entities (people, organisations, products, "
            "identifiers) from the question below. Return ONLY a comma-separated "
            "list, no commentary.\n\nQuestion: " + query
        )
        try:
            content = self._invoke_llm(prompt)
            return [t.strip() for t in content.split(",") if t.strip()]
        except Exception as exc:
            self.logger.warning("Entity extraction failed: %s", exc)
            return []

    @staticmethod
    def _resolve_model_id(llm: Any) -> Optional[str]:
        """Resolve a LiteLLM model id from the various LLM wrappers in use.

        Handles Strands ``LiteLLMModel`` (``get_config()['model_id']``), plain
        config dicts, ``model_id``/``model`` attributes, and bare strings.
        """
        if isinstance(llm, str):
            return llm
        if hasattr(llm, "get_config"):
            cfg = llm.get_config() or {}
            return cfg.get("model_id") or cfg.get("model")
        cfg = getattr(llm, "config", None)
        if isinstance(cfg, dict):
            return cfg.get("model_id") or cfg.get("model")
        return getattr(llm, "model_id", None) or getattr(llm, "model", None)

    def _invoke_llm(self, prompt: str) -> str:
        """Call the configured LLM, supporting LiteLLM/Strands and LangChain.

        Prefers a LiteLLM completion when a model id can be resolved (covers the
        Strands ``LiteLLMModel``, which is *not* a LangChain Runnable and has no
        ``.invoke``/``.bind``); otherwise falls back to a LangChain ``.invoke``.
        """
        llm = self._llm
        model_id = self._resolve_model_id(llm)
        if model_id:
            from litellm import completion  # noqa: PLC0415
            extra = {"aws_region_name": self._llm_region} if self._llm_region else {}
            resp = completion(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                **extra,
            )
            return resp["choices"][0]["message"]["content"]
        if hasattr(llm, "invoke"):
            resp = llm.invoke(prompt)
            return resp.content if hasattr(resp, "content") else str(resp)
        raise ValueError("Could not resolve an LLM model id or .invoke() interface.")

    # ------------------------------------------------------------------
    # Sub-graph traversal + serialisation
    # ------------------------------------------------------------------

    def _neighbourhood_text(self, node) -> Tuple[str, dict]:
        """Expand an entry node's N-hop neighbourhood → (serialised text, meta)."""
        eid = node.element_id
        cypher = (
            "MATCH (n) WHERE elementId(n) = $eid "
            f"MATCH p = (n)-[*1..{self._max_hops}]-(m) "
            f"RETURN p LIMIT {self._rel_limit}"
        )
        rows = self._read(cypher, {"eid": eid})

        header = f"({self._node_label(node)}: \"{self._node_display(node)}\")"
        lines: List[str] = [header]
        seen_rel = set()
        for r in rows:
            path = r["p"]
            for rel in path.relationships:
                if rel.element_id in seen_rel:
                    continue
                seen_rel.add(rel.element_id)
                start, end = rel.start_node, rel.end_node
                lines.append(
                    f"  ({self._node_label(start)}: \"{self._node_display(start)}\") "
                    f"-[{rel.type}]-> "
                    f"({self._node_label(end)}: \"{self._node_display(end)}\")"
                )

        meta = {
            "source": f"neo4j:{self._node_label(node)}",
            "entry_node": self._node_display(node),
            "entry_node_id": eid,
            "relationships": len(seen_rel),
        }
        return "\n".join(lines), meta

    # ------------------------------------------------------------------
    # text2cypher (opt-in)
    # ------------------------------------------------------------------
    #
    # Implemented directly (LLM → Cypher → guarded read → LLM answer) rather than
    # via langchain-neo4j's GraphCypherQAChain, because the agent's LLM is a
    # Strands ``LiteLLMModel`` — not a LangChain Runnable — so the chain's
    # ``llm.bind(...)`` call fails. Doing it ourselves also keeps the full safety
    # guard and the hardened prompts under our control.

    @staticmethod
    def _clean_cypher(text: str) -> str:
        """Strip code fences / prose so only the Cypher statement remains."""
        t = (text or "").strip()
        t = re.sub(r"^```(?:cypher)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
        t = re.sub(r"^\s*cypher(?:\s+query)?\s*:\s*", "", t, flags=re.IGNORECASE)
        return t.strip()

    def _rows_to_text(self, rows: list, limit: int) -> str:
        """Serialise Cypher result rows to a compact text context."""
        out = []
        for r in rows[:limit]:
            data = r.data() if hasattr(r, "data") else dict(r)
            out.append("; ".join(f"{key}: {val}" for key, val in data.items()))
        return "\n".join(out)

    def _text2cypher(self, query: str, k: int) -> List[Tuple[Document, float]]:
        if not self._llm:
            raise ValueError(
                "retrieval_mode 'text2cypher' requires an LLM. Provide one via the "
                "knowledge_base factory, or use retrieval_mode 'traversal'."
            )

        # 1. Generate Cypher from the question + introspected schema (hardened
        #    prompt forces read-only, schema-bound, case-insensitive CONTAINS).
        schema = self.get_schema()
        gen_prompt = _CYPHER_GENERATION_TEMPLATE.format(schema=schema, question=query)
        cypher = self._clean_cypher(self._invoke_llm(gen_prompt))

        # 2. Execute it read-only. Any guard violation or execution error becomes
        #    an empty result rather than an exception, so the agent never falls
        #    back to hallucinating from parametric knowledge.
        rows: list = []
        try:
            self._assert_read_only(cypher)
            rows = self._read(cypher)
        except UnsafeCypherError:
            self.logger.warning("Refused non-read-only generated Cypher: %s", cypher)
        except Exception as exc:
            self.logger.warning("Generated Cypher failed (%s): %s", exc, cypher)

        context = self._rows_to_text(rows, limit=k)
        if not rows:
            self.logger.warning(
                "text2cypher returned 0 rows for query %r (generated Cypher: %s). "
                "The answer must not assert a negative from an empty result.",
                query, cypher,
            )

        # 3. Answer strictly from the rows (hardened QA prompt forbids asserting
        #    negatives when the context is empty).
        qa_prompt = _QA_TEMPLATE.format(
            context=context or "(no results)", question=query
        )
        answer = self._invoke_llm(qa_prompt)

        doc = Document(
            page_content=answer,
            metadata={
                "source": "neo4j:text2cypher",
                "cypher": cypher,
                "result_rows": len(rows),
            },
        )
        return [(doc, 1.0)]

    # ------------------------------------------------------------------
    # BaseVectorStore — read seam
    # ------------------------------------------------------------------

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        if self._mode == "text2cypher":
            return self._text2cypher(query, k)

        entries = self._entry_nodes(query, k)
        if not entries:
            return []

        max_raw = max((s for _, s in entries), default=1.0) or 1.0
        results: List[Tuple[Document, float]] = []
        for node, raw in entries:
            text, meta = self._neighbourhood_text(node)
            meta["raw_score"] = raw
            norm = max(0.0, min(1.0, raw / max_raw))   # 0..1, top entry = 1.0
            results.append((Document(page_content=text, metadata=meta), norm))
        return results

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k=k, **kwargs)]

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
            return self.similarity_search(query_text, k=n_results)
        return []

    def count(self) -> int:
        try:
            rows = self._read("MATCH (n) RETURN count(n) AS c")
            return int(rows[0]["c"]) if rows else 0
        except Exception as exc:
            self.logger.warning("count() failed: %s", exc)
            return 0

    def graph_stats(self, sample_limit: int = 5) -> dict:
        """Read-only snapshot of the graph so a user can judge if this KB fits.

        Returns node count, labels, relationship types, optional per-label counts
        (when APOC is available), full-text index presence (validates the
        configured index), and a few sample nodes. Every query is read-only and
        bounded; failures degrade to empty fields rather than raising.
        """
        stats: dict = {
            "node_count": 0,
            "relationship_count": None,
            "labels": [],
            "label_counts": None,
            "relationship_types": [],
            "fulltext_indexes": [],
            "fulltext_index_configured": self._fulltext_index,
            "fulltext_index_present": None,
            "sample_nodes": [],
            "retrieval_mode": self._mode,
            "entry_strategy": self._entry_strategy,
        }

        try:
            rows = self._read("MATCH (n) RETURN count(n) AS c")
            stats["node_count"] = int(rows[0]["c"]) if rows else 0
        except Exception as exc:
            self.logger.warning("graph_stats: node count failed: %s", exc)

        try:
            rows = self._read("CALL db.labels() YIELD label RETURN label ORDER BY label")
            stats["labels"] = [r["label"] for r in rows]
        except Exception as exc:
            self.logger.warning("graph_stats: labels failed: %s", exc)

        try:
            rows = self._read(
                "CALL db.relationshipTypes() YIELD relationshipType "
                "RETURN relationshipType ORDER BY relationshipType"
            )
            stats["relationship_types"] = [r["relationshipType"] for r in rows]
        except Exception as exc:
            self.logger.warning("graph_stats: relationship types failed: %s", exc)

        # Per-label counts + total rel count — cheap via APOC when present.
        try:
            rows = self._read(
                "CALL apoc.meta.stats() YIELD labels, relCount "
                "RETURN labels, relCount"
            )
            if rows:
                stats["label_counts"] = rows[0].get("labels")
                stats["relationship_count"] = int(rows[0].get("relCount") or 0)
        except Exception:
            pass  # APOC not installed — optional

        # Full-text indexes — validate the configured one actually exists.
        try:
            rows = self._read("SHOW FULLTEXT INDEXES YIELD name RETURN name")
            names = [r["name"] for r in rows]
            stats["fulltext_indexes"] = names
            if self._fulltext_index:
                stats["fulltext_index_present"] = self._fulltext_index in names
        except Exception as exc:
            self.logger.warning("graph_stats: fulltext index listing failed: %s", exc)

        try:
            rows = self._read(f"MATCH (n) RETURN n LIMIT {int(sample_limit)}")
            stats["sample_nodes"] = [
                {
                    "labels": list(getattr(r["n"], "labels", []) or []),
                    "display": self._node_display(r["n"]),
                }
                for r in rows
            ]
        except Exception as exc:
            self.logger.warning("graph_stats: sample nodes failed: %s", exc)

        return stats

    # ------------------------------------------------------------------
    # BaseVectorStore — write seam (read-only: refused)
    # ------------------------------------------------------------------

    def add_documents(self, documents, ids=None) -> None:  # noqa: D401
        raise GraphReadOnlyError(
            "Neo4jGraphStore is read-only; load the graph via your governed ETL "
            "pipeline, not through the agent runtime."
        )

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        *,
        ids: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> List[str]:
        raise GraphReadOnlyError("Neo4jGraphStore is read-only; ingestion is disabled.")

    def delete(self, ids: Optional[List[str]] = None, **kwargs: Any) -> Optional[bool]:
        self.logger.warning("delete() ignored — Neo4jGraphStore is read-only.")
        return None

    def reset_collection(self) -> None:
        self.logger.warning("reset_collection() ignored — Neo4jGraphStore is read-only.")

    def reinitialize_database(self) -> bool:
        # A pre-loaded graph never needs (re)initialisation from the agent side.
        return False

    @classmethod
    def from_texts(cls, texts, embedding=None, metadatas=None, **kwargs):
        raise GraphReadOnlyError(
            "Neo4jGraphStore is read-only and cannot be built from texts."
        )

    def close(self) -> None:
        try:
            if self._driver is not None:
                self._driver.close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass
