"""
SAML token validator — canonical implementation for the OAI platform.

This module provides three public symbols that all consuming services import:

    from oai_platform_core.security.saml_token_validation import (
        TokenValidator,
        TokenValidationError,
        ValidationResult,
    )

Key path resolution
-------------------
``TokenValidator`` accepts an explicit ``public_key_path`` constructor argument.
All consuming services already pass ``os.environ.get("SAML_PUBLIC_KEY_PATH")``
so the default path is rarely exercised.  When it *is* needed, resolution order:

1. ``SAML_PUBLIC_KEY_PATH`` environment variable
2. ``~/.oai/keys/saml-public-key.pem`` (consistent fallback across all services)
3. Sub-class override of :py:meth:`_get_default_key_path` — each package-level
   shim overrides this method to point at its own
   ``resources/keys/public-key.pem`` for backward compatibility.
"""

import base64
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Any, Dict, Optional
from xml.etree.ElementTree import ParseError

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

logger = logging.getLogger(__name__)

__all__ = ["TokenValidationError", "ValidationResult", "TokenValidator"]


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class TokenValidationError(Exception):
    """Raised when SAML token validation cannot complete successfully."""


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Structured result returned by :py:meth:`TokenValidator.validate_token_and_get_role`."""

    role: Optional[str] = None
    email: Optional[str] = None
    is_valid: bool = False
    is_tampered: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "role": self.role,
            "isValid": self.is_valid,
            "isTampered": self.is_tampered,
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class TokenValidator:
    """
    Secure SAML token validator with XML parsing and RSA/SHA-256 signature
    verification.

    Usage::

        validator = TokenValidator(os.environ.get("SAML_PUBLIC_KEY_PATH"))
        result = validator.validate_token_and_get_role(token)
        if result.is_valid:
            print(result.role, result.email)
    """

    # SAML namespace mapping
    NAMESPACES = {
        "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
        "saml":  "urn:oasis:names:tc:SAML:2.0:assertion",
        "ds":    "http://www.w3.org/2000/09/xmldsig#",
    }

    def __init__(self, public_key_path: Optional[str] = None) -> None:
        """
        Initialise the validator.

        Args:
            public_key_path: Path to the PEM-encoded RSA public key.  When
                ``None`` (or the env var ``SAML_PUBLIC_KEY_PATH`` resolves to
                ``None``), :py:meth:`_get_default_key_path` is called instead.
        """
        self.public_key_path = public_key_path or self._get_default_key_path()
        self._public_key: Optional[RSAPublicKey] = None

    # ------------------------------------------------------------------
    # Key path resolution — override in subclasses for package-local keys
    # ------------------------------------------------------------------

    def _get_default_key_path(self) -> str:
        """Return the default public key path when none is explicitly provided.

        Resolution order
        ----------------
        1. ``SAML_PUBLIC_KEY_PATH`` environment variable — the standard way
           all services supply the path at runtime.
        2. ``~/.oai/keys/saml-public-key.pem`` — user-level override; place
           a custom key there to replace the bundled one without changing any
           config.
        3. ``<oai_platform_core package>/resources/keys/public-key.pem`` —
           the key bundled with ``oai-platform-core`` itself, used
           automatically when neither of the above is present.
        """
        # 1. Explicit env var
        env_path = os.environ.get("SAML_PUBLIC_KEY_PATH")
        if env_path:
            return env_path

        # 2. User-level override (exists check so we don't shadow the bundled key
        #    with a missing path)
        user_path = os.path.join(
            os.path.expanduser("~"), ".oai", "keys", "saml-public-key.pem"
        )
        if os.path.exists(user_path):
            return user_path

        # 3. Bundled key shipped inside oai-platform-core
        return os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "resources",
            "keys",
            "public-key.pem",
        )

    # ------------------------------------------------------------------
    # Key loading
    # ------------------------------------------------------------------

    def _load_public_key(self) -> RSAPublicKey:
        """Load and cache the RSA public key from disk.

        Returns:
            The loaded :class:`~cryptography.hazmat.primitives.asymmetric.rsa.RSAPublicKey`.

        Raises:
            :exc:`TokenValidationError`: if the file is missing or not a valid
                RSA PEM key.
        """
        if self._public_key is not None:
            return self._public_key

        if not os.path.exists(self.public_key_path):
            raise TokenValidationError(
                f"Public key not found: {self.public_key_path}\n"
                "Set the SAML_PUBLIC_KEY_PATH environment variable or pass "
                "public_key_path to TokenValidator()."
            )

        try:
            with open(self.public_key_path, "rb") as key_file:
                key_data = key_file.read()

            self._public_key = serialization.load_pem_public_key(key_data)

            if not isinstance(self._public_key, RSAPublicKey):
                raise TokenValidationError("Public key must be RSA format")

            logger.info("SAML public key loaded from %s", self.public_key_path)
            return self._public_key

        except TokenValidationError:
            raise
        except Exception as exc:
            raise TokenValidationError(f"Failed to load public key: {exc}") from exc

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _validate_token_format(self, token: str) -> str:
        """Base64-decode *token* and verify it contains SAML XML.

        Returns:
            The decoded XML string.

        Raises:
            :exc:`TokenValidationError`: on encoding or structural problems.
        """
        if not token or not isinstance(token, str):
            raise TokenValidationError("Token must be a non-empty string")

        try:
            decoded_bytes = base64.b64decode(token, validate=True)
            decoded_xml = decoded_bytes.decode("utf-8")

            if not decoded_xml.strip().startswith("<?xml") and (
                "<saml:" not in decoded_xml and "<samlp:" not in decoded_xml
            ):
                raise TokenValidationError("Token does not contain valid SAML XML")

            return decoded_xml

        except (base64.binascii.Error, UnicodeDecodeError) as exc:
            raise TokenValidationError(f"Invalid token encoding: {exc}") from exc

    def _parse_saml_xml(self, xml_content: str) -> ET.Element:
        """Parse *xml_content* into an :class:`xml.etree.ElementTree.Element`.

        Raises:
            :exc:`TokenValidationError`: on malformed XML.
        """
        try:
            return ET.fromstring(xml_content)
        except ParseError as exc:
            raise TokenValidationError(f"Invalid XML structure: {exc}") from exc
        except ET.XMLParserError as exc:
            raise TokenValidationError(f"XML parsing failed: {exc}") from exc

    def _extract_signed_content(self, xml_content: str) -> str:
        """Extract the ``<saml:Assertion>`` element that was signed.

        Note
        ----
        A fully standards-compliant implementation would canonicalise the XML
        (C14N) before verification.  This implementation extracts the raw
        assertion string as a pragmatic approach suitable for the current IDP
        configuration.

        Raises:
            :exc:`TokenValidationError`: if the signature or assertion element
                is absent.
        """
        if xml_content.find("<ds:Signature") == -1:
            raise TokenValidationError("Signature element not found in SAML response.")

        assertion_match = re.search(
            r"<saml:Assertion[^>]*>.*?</saml:Assertion>", xml_content, re.DOTALL
        )
        if not assertion_match:
            raise TokenValidationError("No SAML assertion found in token")
        return assertion_match.group(0)

    def _extract_email(self, xml_content: str) -> Optional[str]:
        """Return the ``email`` SAML attribute value, or ``None`` if absent."""
        match = re.search(
            r'<saml:Attribute Name="email"[^>]*>\s*'
            r"<saml:AttributeValue[^>]*>(.*?)</saml:AttributeValue>",
            xml_content,
            re.DOTALL,
        )
        return match.group(1).strip() if match else None

    def _extract_role(self, xml_content: str) -> Optional[str]:
        """Return the ``role`` SAML attribute value, or ``None`` if absent."""
        match = re.search(
            r'<saml:Attribute Name="role"[^>]*>\s*'
            r"<saml:AttributeValue[^>]*>(.*?)</saml:AttributeValue>",
            xml_content,
            re.DOTALL,
        )
        return match.group(1).strip() if match else None

    def _extract_signature(self, xml_content: str) -> str:
        """Extract the base64-encoded ``<ds:SignatureValue>`` string.

        Raises:
            :exc:`TokenValidationError`: if the element is missing.
        """
        match = re.search(
            r"<ds:SignatureValue>(.*?)</ds:SignatureValue>", xml_content, re.DOTALL
        )
        if not match:
            raise TokenValidationError("No SignatureValue found in the XML")
        return match.group(1).replace("\n", "").strip()

    # ------------------------------------------------------------------
    # Signature verification
    # ------------------------------------------------------------------

    def _verify_signature(self, signed_content: str, signature: str) -> bool:
        """Verify *signature* over *signed_content* using PKCS1v15/SHA-256.

        Returns:
            ``True`` if the signature is valid, ``False`` otherwise.
        """
        try:
            public_key = self._load_public_key()
            public_key.verify(
                base64.b64decode(signature),
                signed_content.encode("utf-8"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            logger.info("SAML signature verification successful")
            return True
        except InvalidSignature:
            logger.warning("SAML signature verification failed — token may be tampered")
            return False
        except Exception as exc:
            logger.error("Signature verification error: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate_token_and_get_role(self, token: str) -> ValidationResult:
        """Validate a SAML token and return role/email information.

        Args:
            token: Base64-encoded SAML ``<samlp:Response>`` XML.

        Returns:
            A :class:`ValidationResult` — always returned (never raises).
            Inspect ``result.is_valid`` to determine success.
        """
        try:
            logger.info("Starting SAML token validation")
            xml_content = self._validate_token_format(token)

            signed_content = self._extract_signed_content(xml_content)
            role = self._extract_role(xml_content)
            email = self._extract_email(xml_content)
            signature = self._extract_signature(xml_content)

            is_valid = self._verify_signature(signed_content, signature)

            result = ValidationResult(
                role=role,
                email=email,
                is_valid=is_valid,
                is_tampered=not is_valid,
                error_message=None if is_valid else "Signature verification failed",
            )
            logger.info(
                "SAML token validation completed: valid=%s role=%s", is_valid, role
            )
            return result

        except TokenValidationError as exc:
            logger.error("SAML token validation failed: %s", exc)
            return ValidationResult(
                is_valid=False, is_tampered=True, error_message=str(exc)
            )
        except Exception as exc:
            logger.critical("Unexpected SAML token validation error: %s", exc)
            return ValidationResult(
                is_valid=False,
                is_tampered=True,
                error_message="Internal validation error",
            )
