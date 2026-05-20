import pytest
from unittest.mock import MagicMock, patch
from oai_agent_server.services.token_service import TokenService
from oai_agent_server.exceptions import TokenGenerationException

class TestTokenService:
    
    @pytest.fixture
    def token_service(self):
        with patch('oai_agent_server.services.token_service.TokenManager') as MockTM:
            service = TokenService()
            # Ensure the mock is set correctly on the instance
            service.token_manager = MockTM.return_value
            yield service

    def test_generate_token_success(self, token_service):
        token_service.token_manager.generate_token.return_value = "generated_token"
        
        result = token_service.generate_token(
            server_key="test_server",
            user_id="user1",
            role_id="admin",
            ttl_seconds=3600
        )
        
        assert isinstance(result, dict)
        assert result["token"] == "generated_token"
        assert result["user_id"] == "user1"
        assert result["role_id"] == "admin"
        assert result["ttl_seconds"] == 3600
        
        token_service.token_manager.generate_token.assert_called_once_with(
            server_key="test_server",
            user_id="user1",
            role_id="admin",
            ttl_seconds=3600,
            max_tokens=None
        )

    def test_generate_token_defaults(self, token_service):
        token_service.token_manager.generate_token.return_value = "generated_token"
        
        result = token_service.generate_token(server_key="test_server")
        
        assert result["token"] == "generated_token"
        assert result["user_id"] is None
        assert result["role_id"] is None
        assert result["ttl_seconds"] == 3600
        
        token_service.token_manager.generate_token.assert_called_once_with(
            server_key="test_server",
            user_id=None,
            role_id=None,
            ttl_seconds=3600,
            max_tokens=None
        )

    def test_generate_token_exception(self, token_service):
        token_service.token_manager.generate_token.side_effect = Exception("Redis error")
        
        with pytest.raises(TokenGenerationException) as excinfo:
            token_service.generate_token(server_key="test_server")
        
        # TokenGenerationException format: "Token generation error: {reason}"
        assert "Redis error" in str(excinfo.value.detail)
