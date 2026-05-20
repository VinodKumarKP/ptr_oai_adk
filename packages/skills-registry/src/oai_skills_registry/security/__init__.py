"""
Skills-registry security package.

Public exports:
- verify_api_key       FastAPI dependency — validates every protected request
- _validate_token      Synchronous core validation helper (SAML + API tokens)
"""

from oai_skills_registry.security.dependencies import _validate_token, verify_api_key

__all__ = ["_validate_token", "verify_api_key"]
