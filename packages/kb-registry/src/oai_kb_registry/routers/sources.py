"""
Data-source management routes.

  GET    /loaders                                              — List all available loader types
  POST   /loaders/test                                        — Test a loader connection
  POST   /knowledge-bases/{kb_name}/sources                   — Add & auto-sync a data source
  GET    /knowledge-bases/{kb_name}/sources                   — List sources for a KB
  DELETE /knowledge-bases/{kb_name}/sources/{source_id}       — Remove a source
  POST   /knowledge-bases/{kb_name}/sources/{source_id}/sync  — Trigger a manual sync
  GET    /knowledge-bases/{kb_name}/sources/{source_id}/status — Get sync status / recent runs
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from oai_kb_registry.dependencies import get_registry, get_sync_service, verify_bearer_token
from oai_kb_registry.loaders.catalog import LOADER_CATALOG, get_public_config
from oai_kb_registry.models import (
    DataSourceCreate,
    DataSourceResponse,
    LoaderCatalogEntry,
    LoaderField,
    LoaderFieldType,
    SyncRunResponse,
    SyncStatus,
    TestConnectionRequest,
    TestConnectionResponse,
)
from oai_kb_registry.services.kb_registry import KBRegistry
from oai_kb_registry.services.source_sync_service import SourceSyncService

router = APIRouter()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _catalog_entry_to_model(entry: Dict[str, Any]) -> LoaderCatalogEntry:
    """Convert a raw catalog dict to a Pydantic LoaderCatalogEntry."""
    fields = [
        LoaderField(
            name=f["name"],
            type=LoaderFieldType(f["type"]),
            label=f["label"],
            required=f.get("required", False),
            default=f.get("default"),
            placeholder=f.get("placeholder"),
            hint=f.get("hint"),
            help=f.get("help"),
            env_var_hint=f.get("env_var_hint"),
            options=f.get("options"),
        )
        for f in entry.get("fields", [])
    ]
    return LoaderCatalogEntry(
        id=entry["id"],
        display_name=entry["display_name"],
        category=entry["category"],
        description=entry["description"],
        pip_extra=entry.get("pip_extra"),
        fields=fields,
    )


def _row_to_source_response(row: Dict[str, Any]) -> DataSourceResponse:
    """Convert a DB row dict to a DataSourceResponse."""
    config_public: Dict[str, Any] = {}
    if row.get("config_public"):
        try:
            config_public = json.loads(row["config_public"])
        except (json.JSONDecodeError, TypeError):
            pass

    return DataSourceResponse(
        id=str(row["id"]),
        kb_name=row.get("kb_name", ""),
        source_type=row["source_type"],
        display_name=row["display_name"],
        config_public=config_public,
        sync_schedule=row.get("sync_schedule"),
        sync_status=SyncStatus(row.get("sync_status", "never")),
        last_sync_at=row.get("last_sync_at"),
        last_sync_error=row.get("last_sync_error"),
        document_count=row.get("document_count", 0),
        created_at=row.get("created_at") or datetime.now(timezone.utc),
    )


def _row_to_run_response(row: Dict[str, Any]) -> SyncRunResponse:
    return SyncRunResponse(
        id=str(row["id"]),
        source_id=str(row["source_id"]),
        started_at=row.get("started_at"),
        completed_at=row.get("completed_at"),
        status=SyncStatus(row.get("status", "running")),
        document_count=row.get("document_count", 0),
        error_message=row.get("error_message"),
    )


# ---------------------------------------------------------------------------
# Background sync task
# ---------------------------------------------------------------------------

async def _run_sync(
    sync_svc: SourceSyncService,
    registry: KBRegistry,
    source_id: str,
    kb_name: str,
    source_type: str,
    config: Dict[str, Any],
    run_id: Optional[str] = None,
) -> None:
    """Background coroutine: execute a full sync and record the run.

    If *run_id* is supplied the caller already created the sync-run record and
    set the source status to ``running``; we skip those DB writes and go
    straight to the actual data fetch.  Otherwise we create them here.
    """
    db = registry.db
    if run_id is None:
        run_id = await db.create_sync_run(source_id)
        await db.update_data_source_sync_status(source_id, "running")

    try:
        indexed, error_msg = await sync_svc.sync_source(
            source_id=source_id,
            kb_name=kb_name,
            source_type=source_type,
            config=config,
        )
        final_status = "failed" if error_msg and indexed == 0 else "success"
        await db.update_sync_run(
            run_id,
            status=final_status,
            document_count=indexed,
            error_message=error_msg,
        )
        await db.update_data_source_sync_status(
            source_id,
            status=final_status,
            last_sync_error=error_msg,
            document_count=indexed,
        )
        logger.info(
            "Sync completed for source %s: %d docs, status=%s",
            source_id, indexed, final_status,
        )
    except Exception as exc:
        err = str(exc)
        logger.error("Sync failed for source %s: %s", source_id, err)
        await db.update_sync_run(run_id, status="failed", error_message=err)
        await db.update_data_source_sync_status(
            source_id, status="failed", last_sync_error=err
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/loaders", response_model=Dict[str, List[LoaderCatalogEntry]])
async def list_loaders(
    _auth: bool = Depends(verify_bearer_token),
) -> Dict[str, List[LoaderCatalogEntry]]:
    """Return all available loader types grouped by category.

    Example categories: ``collaboration``, ``cloud_storage``, ``web``, ``code``.
    """
    grouped: Dict[str, List[LoaderCatalogEntry]] = {}
    for entry in LOADER_CATALOG.values():
        model = _catalog_entry_to_model(entry)
        grouped.setdefault(model.category, []).append(model)
    return grouped


@router.post("/loaders/test", response_model=TestConnectionResponse)
async def test_loader_connection(
    body: TestConnectionRequest,
    _auth: bool = Depends(verify_bearer_token),
    sync_svc: SourceSyncService = Depends(get_sync_service),
) -> TestConnectionResponse:
    """Test credentials / connectivity for a loader without persisting anything.

    Fetches a small sample (≤ 3 docs) to verify the configuration works.
    """
    result = await sync_svc.test_connection(
        source_type=body.source_type,
        config=body.config,
    )
    return TestConnectionResponse(
        status=result["status"],
        message=result.get("message"),
        sample_count=result.get("sample_count"),
    )


@router.post("/knowledge-bases/{kb_name}/sources", response_model=DataSourceResponse, status_code=201)
async def create_source(
    kb_name: str,
    body: DataSourceCreate,
    background_tasks: BackgroundTasks,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
    sync_svc: SourceSyncService = Depends(get_sync_service),
) -> DataSourceResponse:
    """Register a new data source and start an initial sync in the background.

    The source is persisted immediately; the sync runs asynchronously.
    Poll ``GET /sources/{source_id}/status`` to monitor progress.
    """
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_name!r} not found")

    kb_id = kb["id"]

    # Separate the secret-containing config from the safe-to-display version
    config_encrypted = json.dumps(body.config)
    config_public    = json.dumps(get_public_config(body.source_type, body.config))

    source_id = await registry.db.create_data_source(
        kb_id=kb_id,
        source_type=body.source_type,
        display_name=body.display_name,
        config_encrypted=config_encrypted,
        config_public=config_public,
        sync_schedule=body.sync_schedule,
    )

    # Kick off an immediate background sync
    background_tasks.add_task(
        _run_sync,
        sync_svc, registry,
        source_id, kb_name,
        body.source_type, body.config,
    )

    logger.info("Created data source %s for kb=%s; initial sync queued", source_id, kb_name)

    row = await registry.db.get_data_source(source_id) or {}
    row.setdefault("kb_name", kb_name)
    return _row_to_source_response(row)


@router.get("/knowledge-bases/{kb_name}/sources", response_model=List[DataSourceResponse])
async def list_sources(
    kb_name: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
) -> List[DataSourceResponse]:
    """List all data sources registered to a knowledge base."""
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_name!r} not found")

    rows = await registry.db.get_kb_data_sources(kb["id"])
    return [_row_to_source_response({**r, "kb_name": kb_name}) for r in rows]


@router.delete("/knowledge-bases/{kb_name}/sources/{source_id}", status_code=204)
async def delete_source(
    kb_name: str,
    source_id: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
) -> None:
    """Remove a data source.  Does not delete already-indexed documents."""
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_name!r} not found")

    row = await registry.db.get_data_source(source_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Source {source_id!r} not found")
    if row.get("kb_id") != kb["id"]:
        raise HTTPException(status_code=404, detail=f"Source {source_id!r} does not belong to {kb_name!r}")

    await registry.db.delete_data_source(source_id)


@router.post(
    "/knowledge-bases/{kb_name}/sources/{source_id}/sync",
    response_model=SyncRunResponse,
    status_code=202,
)
async def trigger_sync(
    kb_name: str,
    source_id: str,
    background_tasks: BackgroundTasks,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
    sync_svc: SourceSyncService = Depends(get_sync_service),
) -> SyncRunResponse:
    """Trigger a manual sync for a data source.

    Returns a ``202 Accepted`` with the new sync-run record.
    The sync executes asynchronously — poll the status endpoint for progress.
    """
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_name!r} not found")

    row = await registry.db.get_data_source(source_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Source {source_id!r} not found")
    if row.get("kb_id") != kb["id"]:
        raise HTTPException(status_code=404, detail=f"Source {source_id!r} does not belong to {kb_name!r}")

    if row.get("sync_status") == "running":
        raise HTTPException(status_code=409, detail="A sync is already in progress for this source")

    # Decode the stored (full) config for the sync
    config: Dict[str, Any] = {}
    if row.get("config_encrypted"):
        try:
            config = json.loads(row["config_encrypted"])
        except (json.JSONDecodeError, TypeError):
            pass

    # Create the run record synchronously so we can return it (and pass it to
    # the background task to avoid creating a duplicate).
    run_id = await registry.db.create_sync_run(source_id)
    await registry.db.update_data_source_sync_status(source_id, "running")

    background_tasks.add_task(
        _run_sync,
        sync_svc, registry,
        source_id, kb_name,
        row["source_type"], config,
        run_id,  # re-use the record we just created
    )

    return SyncRunResponse(
        id=run_id,
        source_id=source_id,
        started_at=datetime.now(timezone.utc),
        completed_at=None,
        status=SyncStatus.RUNNING,
        document_count=0,
        error_message=None,
    )


@router.get(
    "/knowledge-bases/{kb_name}/sources/{source_id}/status",
    response_model=Dict[str, Any],
)
async def get_source_status(
    kb_name: str,
    source_id: str,
    _auth: bool = Depends(verify_bearer_token),
    registry: KBRegistry = Depends(get_registry),
) -> Dict[str, Any]:
    """Return the current sync status and recent sync runs for a source."""
    kb = await registry.db.get_knowledge_base(kb_name)
    if not kb:
        raise HTTPException(status_code=404, detail=f"Knowledge base {kb_name!r} not found")

    row = await registry.db.get_data_source(source_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"Source {source_id!r} not found")
    if row.get("kb_id") != kb["id"]:
        raise HTTPException(status_code=404, detail=f"Source {source_id!r} does not belong to {kb_name!r}")

    runs = await registry.db.get_source_sync_runs(source_id)

    return {
        "source": _row_to_source_response({**row, "kb_name": kb_name}).model_dump(),
        "recent_runs": [_row_to_run_response(r).model_dump() for r in runs[:10]],
    }
