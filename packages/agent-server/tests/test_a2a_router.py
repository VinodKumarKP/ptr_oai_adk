import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi import FastAPI, APIRouter
from fastapi.testclient import TestClient

# Mock the imports that might fail before importing create_a2a_router
with patch('oai_agent_server.routers.a2a.A2A_SDK_AVAILABLE', True), \
     patch('oai_agent_server.routers.a2a.build_agent_card', MagicMock(return_value="mock_agent_card")) as mock_build_card, \
     patch('oai_agent_server.routers.a2a.BaseAgentExecutor', MagicMock(return_value="mock_executor")), \
     patch('oai_agent_server.routers.a2a.DefaultRequestHandler', MagicMock(return_value="mock_handler")), \
     patch('oai_agent_server.routers.a2a.InMemoryTaskStore', MagicMock(return_value="mock_store")), \
     patch('oai_agent_server.routers.a2a.create_agent_card_routes') as mock_card_routes, \
     patch('oai_agent_server.routers.a2a.create_jsonrpc_routes') as mock_rpc_routes:

    mock_card_routes.return_value = [
        MagicMock(path="/card", endpoint=AsyncMock(return_value="card_data"), methods=["GET"])
    ]
    mock_rpc_routes.return_value = [
        MagicMock(path="/", endpoint=AsyncMock(return_value="rpc_data"), methods=["POST"])
    ]

    from oai_agent_server.routers.a2a import create_a2a_router, JsonRpcRequest

@pytest.fixture
def mock_logging_service():
    service = MagicMock()
    service.get_logs = AsyncMock(return_value=[])
    service.get_chat_log_by_interaction_id = AsyncMock(return_value={})
    service.get_activity_logs_by_interaction_id = AsyncMock(return_value=[])
    service.get_session_logs = AsyncMock(return_value=[])
    service.get_log_stats = AsyncMock(return_value={})
    service.get_user_stats = AsyncMock(return_value=[])
    service.get_evaluations_by_agent_name = AsyncMock(return_value=[])
    service.get_evaluations_by_session_id = AsyncMock(return_value=[])
    service.get_evaluation = AsyncMock(return_value={})
    return service

def test_create_a2a_router_disabled():
    router, card = create_a2a_router(
        agent=MagicMock(),
        agent_name="test",
        allowed_modes=["chat"], # a2a not enabled
        db_logger=MagicMock(),
        llm_judge_service=MagicMock(),
        logging_service=MagicMock(),
    )
    assert router is None
    assert card is None

def test_create_a2a_router_sdk_unavailable():
    with patch('oai_agent_server.routers.a2a.A2A_SDK_AVAILABLE', False):
        router, card = create_a2a_router(
            agent=MagicMock(),
            agent_name="test",
            allowed_modes=["a2a"],
            db_logger=MagicMock(),
            llm_judge_service=MagicMock(),
            logging_service=MagicMock(),
        )
        assert router is None
        assert card is None

@pytest.mark.asyncio
async def test_create_a2a_router_integration(mock_logging_service):
    with patch('oai_agent_server.routers.a2a.A2A_SDK_AVAILABLE', True), \
         patch('oai_agent_server.routers.a2a.build_agent_card', MagicMock(return_value="mock_agent_card")), \
         patch('oai_agent_server.routers.a2a.BaseAgentExecutor', MagicMock(return_value="mock_executor")), \
         patch('oai_agent_server.routers.a2a.DefaultRequestHandler', MagicMock(return_value="mock_handler")), \
         patch('oai_agent_server.routers.a2a.InMemoryTaskStore', MagicMock(return_value="mock_store")), \
         patch('oai_agent_server.routers.a2a.create_agent_card_routes') as mock_card_routes, \
         patch('oai_agent_server.routers.a2a.create_jsonrpc_routes') as mock_rpc_routes:

        mock_card_routes.return_value = [
            MagicMock(path="/card", endpoint=AsyncMock(return_value="card_data"), methods=["GET"])
        ]
        mock_rpc_routes.return_value = [
            MagicMock(path="/", endpoint=AsyncMock(return_value="rpc_data"), methods=["POST"])
        ]

        router, card = create_a2a_router(
            agent=MagicMock(),
            agent_name="test_agent",
            allowed_modes=["a2a", "logs", "monitoring"],
            db_logger=MagicMock(),
            llm_judge_service=MagicMock(),
            logging_service=mock_logging_service,
        )
        
        assert router is not None
        assert card == "mock_agent_card"
        
        app = FastAPI()
        app.include_router(router)
        client = TestClient(app)
        
        # Test logs endpoints
        resp = client.get("/logs?limit=5")
        assert resp.status_code == 200
        mock_logging_service.get_logs.assert_called_once()
        
        resp = client.get("/logs/interaction/int1")
        assert resp.status_code == 200
        mock_logging_service.get_chat_log_by_interaction_id.assert_called_once_with("int1")
        
        resp = client.get("/logs/activity/interaction/int1")
        assert resp.status_code == 200
        mock_logging_service.get_activity_logs_by_interaction_id.assert_called_once_with("int1")
        
        resp = client.get("/logs/sessions/sess1")
        assert resp.status_code == 200
        mock_logging_service.get_session_logs.assert_called_once_with("sess1")
        
        resp = client.get("/logs/stats?user_id=u1")
        assert resp.status_code == 200
        mock_logging_service.get_log_stats.assert_called_once_with(user_id="u1")
        
        resp = client.get("/logs/stats/users")
        assert resp.status_code == 200
        mock_logging_service.get_user_stats.assert_called_once()
        
        # Test monitoring endpoints
        resp = client.get("/evaluations/agent")
        assert resp.status_code == 200
        mock_logging_service.get_evaluations_by_agent_name.assert_called_once()
        
        resp = client.get("/evaluations/session/sess1")
        assert resp.status_code == 200
        mock_logging_service.get_evaluations_by_session_id.assert_called_once_with("sess1")
        
        resp = client.get("/evaluations/int1")
        assert resp.status_code == 200
        mock_logging_service.get_evaluation.assert_called_once_with("int1")
