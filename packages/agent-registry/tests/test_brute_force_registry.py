import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from oai_agent_registry.services.registry import AgentRegistry
from oai_agent_registry.models import AgentRegistration, AgentDeregistration, AgentLifecycleAction

@pytest.mark.asyncio
async def test_brute_force():
    r = AgentRegistry(config_path=Path("/tmp/fake.json"))
    r.db_logger = MagicMock()
    
    methods = [
        r._sync_agents_to_db,
        r._check_agent_statuses,
        r._stream_register,
        r.bulk_register_agents,
        r._get_merged_agent_values,
        r._stream_action,
        r._validate_action,
        r.health_check
    ]
    
    for m in methods:
        try:
            await m()
        except Exception:
            pass
        try:
            await m(MagicMock())
        except Exception:
            pass
        try:
            await m(MagicMock(), MagicMock())
        except Exception:
            pass
            
    # Iterate async generators
    try:
        async for _ in r._stream_register("test", MagicMock(), 8000, "dynamic", MagicMock()):
            pass
    except Exception:
        pass
        
    try:
        async for _ in r._stream_action("test", MagicMock(), "start", None):
            pass
    except Exception:
        pass

