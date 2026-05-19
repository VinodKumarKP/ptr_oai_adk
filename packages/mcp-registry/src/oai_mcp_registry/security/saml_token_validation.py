"""
mcp-registry SAML token validation — thin compatibility shim.

All validation logic lives in ``oai_platform_core.security.saml_token_validation``.
Public key path resolution (in order):
  1. ``SAML_PUBLIC_KEY_PATH`` environment variable  (used by all callers)
  2. ``~/.oai/keys/saml-public-key.pem``            (shared fallback)

Canonical source:
    packages/platform-core/src/oai_platform_core/security/saml_token_validation.py
"""

from oai_platform_core.security.saml_token_validation import (  # noqa: F401
    TokenValidationError,
    ValidationResult,
    TokenValidator,
)

__all__ = ["TokenValidationError", "ValidationResult", "TokenValidator"]
