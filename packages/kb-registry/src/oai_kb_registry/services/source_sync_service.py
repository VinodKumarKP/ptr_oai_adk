"""
Source sync service — loads documents from external sources via LangChain
loaders and indexes them into a Knowledge Base.

Responsibilities:
  1. Resolve ${ENV_VAR} placeholders in loader config values.
  2. Dispatch to the correct LangChain loader implementation.
  3. Feed each resulting LangChain Document through KBRegistry.index_document_from_text.
  4. Provide a lightweight test_connection helper (fetches a small sample only).

All loader functions are synchronous (LangChain loaders are not async).
They are run inside asyncio.to_thread / run_in_executor so the FastAPI
event loop stays free during potentially slow network I/O.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from oai_kb_registry.services.kb_registry import KBRegistry

logger = logging.getLogger(__name__)

# Matches ${SOME_VAR_NAME} — resolved from os.environ at sync time
_ENV_VAR_RE = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _resolve_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
    """Replace ${VAR_NAME} string values with their os.environ counterparts."""
    resolved: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, str):
            m = _ENV_VAR_RE.match(v)
            resolved[k] = os.environ.get(m.group(1), "") if m else v
        else:
            resolved[k] = v
    return resolved


# ---------------------------------------------------------------------------
# Loader implementations (synchronous — called inside to_thread)
# ---------------------------------------------------------------------------

def _load_confluence(config: Dict[str, Any]) -> List[Any]:
    try:
        from langchain_community.document_loaders import ConfluenceLoader  # noqa: PLC0415
    except ImportError:
        raise ImportError("atlassian-python-api is required: pip install atlassian-python-api")

    space_key_raw = config.get("space_key", "")
    space_keys = [s.strip() for s in space_key_raw.split(",") if s.strip()]

    loader = ConfluenceLoader(
        url=config["url"],
        username=config["username"],
        api_key=config["api_key"],
    )

    limit = int(config.get("limit", 50))
    include_attachments = bool(config.get("include_attachments", False))

    docs: List[Any] = []
    if space_keys:
        for sk in space_keys:
            docs.extend(
                loader.load(
                    space_key=sk,
                    limit=limit,
                    include_attachments=include_attachments,
                )
            )
    else:
        docs.extend(loader.load(limit=limit, include_attachments=include_attachments))
    return docs


def _load_sharepoint(config: Dict[str, Any]) -> List[Any]:
    try:
        from langchain_community.document_loaders import SharePointLoader  # noqa: PLC0415
    except ImportError:
        raise ImportError("O365 is required: pip install O365")

    loader = SharePointLoader(
        client_id=config["client_id"],
        client_secret=config["client_secret"],
        tenant_id=config["tenant_id"],
        site_name=config["site_name"],
        folder_path=config.get("document_library_path") or "/",
    )
    return loader.load()


def _load_s3_directory(config: Dict[str, Any]) -> List[Any]:
    from langchain_community.document_loaders import S3DirectoryLoader  # noqa: PLC0415

    kwargs: Dict[str, Any] = {}
    if config.get("aws_access_key_id") and config.get("aws_secret_access_key"):
        kwargs["aws_access_key_id"] = config["aws_access_key_id"]
        kwargs["aws_secret_access_key"] = config["aws_secret_access_key"]

    loader = S3DirectoryLoader(
        bucket=config["bucket"],
        prefix=config.get("prefix", ""),
        region_name=config.get("region", "us-east-1"),
        **kwargs,
    )
    return loader.load()


def _load_web(config: Dict[str, Any]) -> List[Any]:
    raw_urls = config.get("urls", "")
    urls = [u.strip() for u in raw_urls.splitlines() if u.strip()]
    if not urls:
        return []

    if config.get("use_sitemap"):
        from langchain_community.document_loaders.sitemap import SitemapLoader  # noqa: PLC0415
        loader = SitemapLoader(web_path=urls[0])
        return loader.load()

    from langchain_community.document_loaders import WebBaseLoader  # noqa: PLC0415
    loader = WebBaseLoader(web_paths=urls)
    return loader.load()


def _load_github(config: Dict[str, Any]) -> List[Any]:
    try:
        from langchain_community.document_loaders import GithubFileLoader  # noqa: PLC0415
    except ImportError:
        raise ImportError("PyGithub is required: pip install PyGithub")

    file_filter_pattern = config.get("file_filter", "*.md") or "*.md"

    from fnmatch import fnmatch  # noqa: PLC0415

    def _filter(file_path: str) -> bool:
        return fnmatch(file_path, file_filter_pattern)

    loader = GithubFileLoader(
        repo=config["repo"],
        branch=config.get("branch", "main"),
        access_token=config.get("access_token") or "",
        file_filter=_filter,
    )
    return loader.load()


# ---------------------------------------------------------------------------
# Dispatch table  source_type → loader function
# ---------------------------------------------------------------------------

_LOADER_DISPATCH = {
    "confluence":   _load_confluence,
    "sharepoint":   _load_sharepoint,
    "s3_directory": _load_s3_directory,
    "web":          _load_web,
    "github":       _load_github,
}


def _doc_name(doc: Any, index: int, source_type: str) -> str:
    """Extract a human-readable name from a LangChain Document's metadata."""
    meta = getattr(doc, "metadata", {}) or {}
    return (
        meta.get("title")
        or meta.get("source")
        or meta.get("path")
        or meta.get("file_path")
        or meta.get("url")
        or f"{source_type}_doc_{index}"
    )


def _doc_uri(doc: Any) -> Optional[str]:
    """Extract a URI (source URL / S3 path) from a Document's metadata."""
    meta = getattr(doc, "metadata", {}) or {}
    return (
        meta.get("source")
        or meta.get("url")
        or meta.get("loc")
        or None
    )


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------

class SourceSyncService:
    """Runs LangChain loaders and feeds results into the KB Registry."""

    def __init__(
        self,
        registry: "KBRegistry",
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._registry = registry
        self.logger = log or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Connection tester
    # ------------------------------------------------------------------

    async def test_connection(
        self,
        source_type: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fetch a small sample to verify credentials / connectivity.

        Returns:
            {"status": "ok",    "message": "...", "sample_count": N}
            {"status": "error", "message": "..."}
        """
        loader_fn = _LOADER_DISPATCH.get(source_type)
        if not loader_fn:
            return {"status": "error", "message": f"Unknown source type: {source_type!r}"}

        resolved = _resolve_env_vars(config)
        # For test runs, cap to avoid long-running fetches
        _cap_for_test(resolved)

        try:
            docs = await asyncio.to_thread(loader_fn, resolved)
            return {
                "status": "ok",
                "message": f"Connection successful. Found {len(docs)} document(s).",
                "sample_count": len(docs),
            }
        except ImportError as exc:
            return {"status": "error", "message": str(exc)}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    # ------------------------------------------------------------------
    # Full sync
    # ------------------------------------------------------------------

    async def sync_source(
        self,
        source_id: str,
        kb_name: str,
        source_type: str,
        config: Dict[str, Any],
    ) -> Tuple[int, Optional[str]]:
        """Load all documents from the source and index them into the KB.

        Returns:
            (indexed_count, error_message_or_None)
        """
        loader_fn = _LOADER_DISPATCH.get(source_type)
        if not loader_fn:
            return 0, f"Unknown source type: {source_type!r}"

        resolved = _resolve_env_vars(config)

        # ── Load from source ────────────────────────────────────────────
        try:
            self.logger.info(
                "sync: loading source %s (type=%s, kb=%s)", source_id, source_type, kb_name
            )
            docs = await asyncio.to_thread(loader_fn, resolved)
            self.logger.info("sync: loaded %d document(s) from source %s", len(docs), source_id)
        except Exception as exc:
            self.logger.error("sync: load failed for source %s: %s", source_id, exc)
            return 0, str(exc)

        # ── Index each document ─────────────────────────────────────────
        indexed = 0
        errors: List[str] = []

        for i, doc in enumerate(docs):
            text = getattr(doc, "page_content", "") or ""
            if not text.strip():
                continue

            name = _doc_name(doc, i, source_type)
            uri  = _doc_uri(doc) or source_id
            meta = {**(getattr(doc, "metadata", {}) or {}), "data_source_id": source_id}

            try:
                await self._registry.index_document_from_text(
                    kb_name=kb_name,
                    name=name,
                    text=text,
                    source_type=source_type,
                    source_uri=uri,
                    extra_metadata=meta,
                    performed_by=f"source:{source_id}",
                )
                indexed += 1
            except Exception as exc:
                self.logger.error(
                    "sync: failed to index doc %d (%s) from source %s: %s",
                    i, name, source_id, exc,
                )
                errors.append(str(exc))

        error_msg: Optional[str] = None
        if errors:
            shown = errors[:3]
            if len(errors) > 3:
                shown.append(f"… and {len(errors) - 3} more")
            error_msg = "; ".join(shown)

        self.logger.info(
            "sync: source %s finished — %d indexed, %d errors",
            source_id, indexed, len(errors),
        )
        return indexed, error_msg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cap_for_test(config: Dict[str, Any]) -> None:
    """Cap limits for test-connection calls to avoid long fetches."""
    if "limit" in config:
        config["limit"] = min(int(config.get("limit", 3)), 3)
    else:
        config["limit"] = 3
