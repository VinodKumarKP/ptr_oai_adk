import os
import pytest
import logging
from unittest.mock import patch
from oai_agent_registry.services.db.database_logger import SQLiteBackend

@pytest.mark.asyncio
async def test_sqlite_backend_initialize(tmp_path):
    db_file = tmp_path / "test_agent_registry.db"
    backend = SQLiteBackend()
    
    logger = logging.getLogger("test")
    
    # Run initialize using patched environment variable SQLITE_DB_PATH
    with patch.dict(os.environ, {"SQLITE_DB_PATH": str(db_file)}):
        success = await backend.initialize(logger)
        
    assert success is True
    assert db_file.exists()
    
    # Call schema creation again with error in migrations (should handle migration error silently)
    backend._db_path = str(db_file)
    await backend._create_schema()
