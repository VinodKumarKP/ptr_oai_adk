"""
oai_platform_core.db — shared async database backend infrastructure.

Re-exports the four public symbols that consuming packages need:
  - DatabaseBackend         abstract interface
  - BasePostgresBackend     connection-pool lifecycle management for PostgreSQL
  - BaseSQLiteBackend       per-request open/use/close pattern for SQLite
  - PersistentSQLiteBackend single persistent connection (WAL-optional) for SQLite
"""
from oai_platform_core.db.base import (
    DatabaseBackend,
    BasePostgresBackend,
    BaseSQLiteBackend,
    PersistentSQLiteBackend,
)

__all__ = [
    "DatabaseBackend",
    "BasePostgresBackend",
    "BaseSQLiteBackend",
    "PersistentSQLiteBackend",
]
