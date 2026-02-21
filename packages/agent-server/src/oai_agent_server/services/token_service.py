from typing import Optional, Dict, Any

from oai_agent_server.exceptions import TokenGenerationException
from oai_agent_server.utils.token_manager import TokenManager


class TokenService:

    def __init__(self):
        self.token_manager = TokenManager()

    def generate_token(self, server_key: str,
                       user_id: Optional[str] = None,
                       role_id: Optional[str] = None,
                       ttl_seconds: Optional[int] = 3600) -> Dict[str, Any]:
        """
        Generate a token with embedded metadata.

        :param server_key: server key
        :param user_id: user id
        :param role_id: role id
        :param ttl_seconds: ttl_seconds
        :return: Dictionary containing the token and metadata
        """
        try:
            token = self.token_manager.generate_token(
                server_key=server_key,
                user_id=user_id,
                role_id=role_id,
                ttl_seconds=ttl_seconds
            )
            return {
                "token": token,
                "user_id": user_id,
                "role_id": role_id,
                "ttl_seconds": ttl_seconds
            }
        except Exception as e:
            raise TokenGenerationException(reason=str(e))
