from typing import Optional, Dict, Any, List

from oai_agent_server.exceptions import TokenGenerationException
from oai_platform_core.security import TokenManager


class TokenService:

    def __init__(self):
        self.token_manager = TokenManager(db_path_name="agent_server_tokens.db")

    def generate_token(self, server_key: str,
                       user_id: Optional[str] = None,
                       role_id: Optional[str] = None,
                       ttl_seconds: Optional[int] = 3600,
                       max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Generate a token with embedded metadata.

        :param server_key: server key
        :param user_id: user id
        :param role_id: role id
        :param ttl_seconds: ttl_seconds
        :param max_tokens: optional hard cap on active tokens; raises ValueError when reached
        :return: Dictionary containing the token and metadata
        """
        try:
            token = self.token_manager.generate_token(
                server_key=server_key,
                user_id=user_id,
                role_id=role_id,
                ttl_seconds=ttl_seconds,
                max_tokens=max_tokens,
            )
            return {
                "token": token,
                "user_id": user_id,
                "role_id": role_id,
                "ttl_seconds": ttl_seconds
            }
        except Exception as e:
            raise TokenGenerationException(reason=str(e))

    def get_all_tokens(self, server_key: str, include_expired: bool = False) -> List[Dict[str, Any]]:
        """Return all tokens for the given server key.

        :param server_key: server key to list tokens for
        :param include_expired: whether to include already-expired TTL tokens
        :return: List of token metadata dicts
        """
        return self.token_manager.get_all_tokens(server_key, include_expired=include_expired)

    def revoke_token(self, token: str) -> bool:
        """Revoke a specific token.

        :param token: the token string to revoke
        :return: True if revoked, False if not found or already revoked
        """
        return self.token_manager.revoke_token(token)

    def revoke_all_tokens(self, server_key: str) -> int:
        """Revoke all tokens for the given server key.

        :param server_key: server key whose tokens should be revoked
        :return: count of revoked tokens
        """
        return self.token_manager.revoke_all_tokens(server_key)
