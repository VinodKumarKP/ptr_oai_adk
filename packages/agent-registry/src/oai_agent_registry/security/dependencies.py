import base64
import os
from typing import Optional

from fastapi import Header, HTTPException, Request
from fastapi.security import APIKeyHeader

from oai_agent_registry.models import RegistryConfig
from oai_platform_core.security.saml_token_validation import TokenValidator, TokenValidationError
from oai_platform_core.security import TokenManager

# Define the API key security scheme
api_key_header = APIKeyHeader(name="X-API-KEY", auto_error=False)

def is_saml_token(token: str) -> bool:
    """
    Checks if a token is likely a SAML token by checking if it's base64 encoded XML.
    """
    if not token or not isinstance(token, str) or len(token) % 4 != 0:
        return False
    try:
        decoded_token = base64.b64decode(token, validate=True)
        # Check for SAML or SAMLP tags, without requiring the XML declaration
        return b'<saml:' in decoded_token or b'<samlp:' in decoded_token
    except (ValueError, TypeError):
        return False

def _validate_token(request: Request, registry_config: RegistryConfig, agent_name: str):
    """
    Core logic to validate the API token.
    """

    if '/status' in request.url.path:
        return True

    auth_enabled = os.environ.get('AGENT_AUTH_ENABLED', 'true').lower() == 'true'

    if not auth_enabled:
        return True

    if ('localhost' in str(request.url) or
        '0.0.0.0' in str(request.url) or
        '127.0.0.1' in str(request.url) or
        '::1' in str(request.url) or
        'host.docker.internal' in str(request.url)) and os.environ.get('FORCE_AUTH',
                                                                             'false').lower() == 'false':
        return True

    token = request.headers.get("api-token") or \
            request.headers.get("api_token") or \
            request.headers.get("x-api-key")

    authorization = request.headers.get("authorization")
    if not token and authorization:
        if authorization.lower().startswith('bearer '):
            token = authorization[7:]
        else:
            token = authorization

    if not token:
        raise HTTPException(status_code=401, detail="API token required")

    if is_saml_token(token):
        try:
            validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH"))
            validation_result = validator.validate_token_and_get_role(token)
            if validation_result.is_valid:
                request.state.user_role = validation_result.role
                request.state.user_email = validation_result.email
                return
            else:
                raise HTTPException(status_code=401, detail=validation_result.error_message or "Invalid SAML token")
        except TokenValidationError as e:
            raise HTTPException(status_code=401, detail=str(e))
        except Exception:
            raise HTTPException(status_code=500, detail="SAML token validation service unavailable")
    else:
        token_manager = TokenManager(db_path_name="agent_registry_tokens.db")
        user_info = token_manager.validate_token(agent_name, token)

        if not user_info:
            raise HTTPException(status_code=401, detail="Invalid or expired API token")

        request.state.user_id = user_info.get("user_id")
        request.state.user_role = user_info.get("role_id")

async def verify_api_key(request: Request):
    """
    Dependency to validate API tokens.
    """
    if '/status' in request.url.path:
        return True

    from oai_agent_registry.dependencies import registry_instance
    # For requests to the registry itself, we can use a generic agent name
    agent_name = "agent-registry"
    _validate_token(request, registry_instance.registry_config, agent_name)
    return True
