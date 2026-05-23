"""
Document processor for KB Registry.

Responsibilities:
  1. Parse uploaded files (PDF, DOCX, TXT, Markdown) or fetch S3/URL content.
  2. Split text into overlapping chunks using LangChain's RecursiveCharacterTextSplitter.
  3. Wrap each chunk in a LangChain Document with metadata (source, kb_name, doc_id …).

The processor is intentionally stateless — it receives raw bytes or a source
reference and returns a list of LangChain Documents ready for embedding.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from langchain_core.documents import Document


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_text_from_pdf(data: bytes) -> str:
    """Extract text from PDF bytes using pypdf."""
    try:
        import pypdf  # noqa: PLC0415

        reader = pypdf.PdfReader(io.BytesIO(data))
        parts: List[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts)
    except ImportError:
        raise RuntimeError(
            "pypdf is required to process PDF files. "
            "Install it with: pip install pypdf"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to parse PDF: {exc}") from exc


def _extract_text_from_docx(data: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        import docx  # noqa: PLC0415

        doc = docx.Document(io.BytesIO(data))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return "\n\n".join(paragraphs)
    except ImportError:
        raise RuntimeError(
            "python-docx is required to process DOCX files. "
            "Install it with: pip install python-docx"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to parse DOCX: {exc}") from exc


def _extract_text_from_bytes(data: bytes, filename: str) -> str:
    """Route to the right extractor based on file extension."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_text_from_pdf(data)
    if suffix in {".docx", ".doc"}:
        return _extract_text_from_docx(data)
    # TXT, MD, CSV, etc. — decode as UTF-8
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("latin-1", errors="replace")


# ---------------------------------------------------------------------------
# S3 fetch helper
# ---------------------------------------------------------------------------

def _fetch_s3_content(
    bucket: str,
    key: str,
    region: str = "us-east-1",
    aws_access_key_id: Optional[str] = None,
    aws_secret_access_key: Optional[str] = None,
) -> bytes:
    try:
        import boto3  # noqa: PLC0415

        kwargs: dict = {"region_name": region}
        if aws_access_key_id and aws_secret_access_key:
            kwargs["aws_access_key_id"] = aws_access_key_id
            kwargs["aws_secret_access_key"] = aws_secret_access_key

        s3 = boto3.client("s3", **kwargs)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except ImportError:
        raise RuntimeError("boto3 is required to fetch S3 objects.")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch s3://{bucket}/{key}: {exc}") from exc


# ---------------------------------------------------------------------------
# URL fetch helper (synchronous, kept simple)
# ---------------------------------------------------------------------------

def _fetch_url_content(url: str) -> bytes:
    try:
        import urllib.request  # noqa: PLC0415

        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            return resp.read()
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch URL {url}: {exc}") from exc


# ---------------------------------------------------------------------------
# Main processor class
# ---------------------------------------------------------------------------

class DocumentProcessor:
    """Parses, splits, and returns LangChain Documents ready for vector indexing."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap

    def _get_splitter(self):  # type: ignore[return]
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        except ImportError:
            raise RuntimeError(
                "langchain-text-splitters is required. "
                "Install it with: pip install langchain-text-splitters"
            )

    def _build_documents(
        self,
        text: str,
        base_metadata: dict,
    ) -> "List[Document]":
        from langchain_core.documents import Document  # noqa: PLC0415

        splitter = self._get_splitter()
        chunks = splitter.split_text(text)
        documents: List[Document] = []
        for i, chunk in enumerate(chunks):
            meta = {**base_metadata, "chunk_index": i, "total_chunks": len(chunks)}
            documents.append(Document(page_content=chunk, metadata=meta))
        logger.debug(
            "Split '%s' into %d chunks (size=%d, overlap=%d)",
            base_metadata.get("doc_name", "?"), len(chunks),
            self.chunk_size, self.chunk_overlap,
        )
        return documents

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_upload(
        self,
        data: bytes,
        filename: str,
        kb_name: str,
        doc_id: Optional[int] = None,
        extra_metadata: Optional[dict] = None,
    ) -> "List[Document]":
        """Parse ``data`` bytes (a file upload) and return Document chunks."""
        text = _extract_text_from_bytes(data, filename)
        meta = {
            "kb_name": kb_name,
            "doc_name": filename,
            "doc_id": doc_id,
            "source_type": "upload",
            **(extra_metadata or {}),
        }
        return self._build_documents(text, meta)

    def process_s3_object(
        self,
        bucket: str,
        key: str,
        kb_name: str,
        doc_id: Optional[int] = None,
        region: str = "us-east-1",
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        extra_metadata: Optional[dict] = None,
    ) -> "List[Document]":
        """Fetch an S3 object and return Document chunks."""
        data = _fetch_s3_content(bucket, key, region, aws_access_key_id, aws_secret_access_key)
        filename = key.split("/")[-1]
        text = _extract_text_from_bytes(data, filename)
        meta = {
            "kb_name": kb_name,
            "doc_name": filename,
            "doc_id": doc_id,
            "source_type": "s3",
            "s3_bucket": bucket,
            "s3_key": key,
            **(extra_metadata or {}),
        }
        return self._build_documents(text, meta)

    def process_url(
        self,
        url: str,
        kb_name: str,
        doc_id: Optional[int] = None,
        extra_metadata: Optional[dict] = None,
    ) -> "List[Document]":
        """Fetch a URL and return Document chunks (plain text / HTML stripped)."""
        data = _fetch_url_content(url)
        filename = url.split("/")[-1] or "url_content.txt"
        text = _extract_text_from_bytes(data, filename)
        meta = {
            "kb_name": kb_name,
            "doc_name": filename,
            "doc_id": doc_id,
            "source_type": "url",
            "source_url": url,
            **(extra_metadata or {}),
        }
        return self._build_documents(text, meta)
