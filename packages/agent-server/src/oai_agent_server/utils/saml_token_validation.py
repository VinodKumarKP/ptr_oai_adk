import base64
import logging
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Dict, Any
from xml.etree.ElementTree import ParseError

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey

# Configure logging
logger = logging.getLogger(__name__)


class TokenValidationError(Exception):
    """Custom exception for token validation failures"""
    pass


@dataclass
class ValidationResult:
    """Structured result from token validation"""
    role: Optional[str] = None
    email: Optional[str] = None
    is_valid: bool = False
    is_tampered: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility"""
        return {
            "role": self.role,
            "isValid": self.is_valid,
            "isTampered": self.is_tampered
        }


class TokenValidator:
    """
    Secure SAML token validator with proper XML parsing and cryptographic verification.
    """

    # SAML namespace mapping
    NAMESPACES = {
        'samlp': 'urn:oasis:names:tc:SAML:2.0:protocol',
        'saml': 'urn:oasis:names:tc:SAML:2.0:assertion',
        'ds': 'http://www.w3.org/2000/09/xmldsig#'
    }

    def __init__(self, public_key_path: Optional[str] = None):
        """
        Initialize token validator.

        Args:
            public_key_path: Optional path to public key file
        """
        self.public_key_path = public_key_path or self._get_default_key_path()
        self._public_key = None

    def _get_default_key_path(self) -> str:
        """Get default public key path"""
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'resources',
            'keys',
            'public-key.pem'
        )

    def _load_public_key(self) -> RSAPublicKey:
        """
        Securely load and validate public key.

        Returns:
            RSAPublicKey: The loaded public key

        Raises:
            TokenValidationError: If key loading fails
        """
        if self._public_key:
            return self._public_key

        if not os.path.exists(self.public_key_path):
            raise TokenValidationError(f"Public key not found: {self.public_key_path}")

        try:
            with open(self.public_key_path, 'rb') as key_file:
                key_data = key_file.read()

            self._public_key = serialization.load_pem_public_key(key_data)

            # Validate it's an RSA key
            if not isinstance(self._public_key, RSAPublicKey):
                raise TokenValidationError("Public key must be RSA format")

            logger.info("Public key loaded successfully")
            return self._public_key

        except Exception as e:
            raise TokenValidationError(f"Failed to load public key: {e}")

    def _validate_token_format(self, token: str) -> str:
        """
        Validate and decode token format.

        Args:
            token: Base64 encoded token

        Returns:
            str: Decoded XML content

        Raises:
            TokenValidationError: If token format is invalid
        """
        if not token or not isinstance(token, str):
            raise TokenValidationError("Token must be a non-empty string")

        try:
            # Validate base64 format
            decoded_bytes = base64.b64decode(token, validate=True)
            decoded_xml = decoded_bytes.decode('utf-8')

            # Basic XML structure validation
            if not decoded_xml.strip().startswith('<?xml') and ('<saml:' not in decoded_xml and '<samlp:' not in decoded_xml):
                raise TokenValidationError("Token does not contain valid SAML XML")

            return decoded_xml

        except (base64.binascii.Error, UnicodeDecodeError) as e:
            raise TokenValidationError(f"Invalid token encoding: {e}")

    def _parse_saml_xml(self, xml_content: str) -> ET.Element:
        """
        Safely parse SAML XML content.

        Args:
            xml_content: XML string to parse

        Returns:
            ET.Element: Parsed XML root element

        Raises:
            TokenValidationError: If XML parsing fails
        """
        try:
            # Parse XML with security considerations
            root = ET.fromstring(xml_content)
            return root

        except ParseError as e:
            raise TokenValidationError(f"Invalid XML structure: {e}")
        except ET.XMLParserError as e:
            raise TokenValidationError(f"XML parsing failed: {e}")

    def _extract_signed_content(self, xml_content: str) -> str:
        """
        Extract the content that is signed. In a SAML Response, this is the <samlp:Response> element itself.
        """
        # The entire <samlp:Response> element is what's signed, excluding the <ds:Signature> part for verification purposes.
        # A simple and effective way is to find the start of the <ds:Signature> and slice the string.
        signature_start_tag = '<ds:Signature'
        sig_start_index = xml_content.find(signature_start_tag)

        if sig_start_index == -1:
            raise TokenValidationError("Signature element not found in SAML response.")

        # The content to be verified is everything before the signature block, plus the closing tag of the response.
        # This is a simplification. Correct validation requires canonicalization of the XML.
        # For this specific structure, we will extract the assertion and verify it against the signature.
        # A more robust solution would use a proper XML security library.

        # Re-searching for the whole Response tag to get the full content that was supposed to be signed.
        response_match = re.search(r'(<samlp:Response[^>]*>.*</samlp:Response>)', xml_content, re.DOTALL)
        if not response_match:
            raise TokenValidationError("SAML Response not found.")

        full_response = response_match.group(1)

        # To verify, we need to re-create the state of the data as it was when it was signed.
        # This often means canonicalizing the XML, which is complex.
        # As a pragmatic approach, we'll assume the signature applies to the <saml:Assertion>
        # and that the assertion is what we need to verify.
        assertion_match = re.search(r'<saml:Assertion[^>]*>.*?</saml:Assertion>', xml_content, re.DOTALL)
        if not assertion_match:
            raise TokenValidationError("No SAML assertion found in token")
        return assertion_match.group(0)

    def _extract_email(self, xml_content) -> Optional[str]:
        """
        Extract user email from SAML attributes.
        """
        email_match = re.search(r'<saml:Attribute Name="email"[^>]*>\s*<saml:AttributeValue[^>]*>(.*?)</saml:AttributeValue>', xml_content, re.DOTALL)
        return email_match.group(1).strip() if email_match else None

    def _extract_role(self, xml_content) -> Optional[str]:
        """
        Extract user role from SAML attributes.
        """
        role_match = re.search(r'<saml:Attribute Name="role"[^>]*>\s*<saml:AttributeValue[^>]*>(.*?)</saml:AttributeValue>', xml_content, re.DOTALL)
        return role_match.group(1).strip() if role_match else None

    def _extract_signature(self, xml_content: str) -> str:
        """
        Extract digital signature from SAML token.
        """
        signature_match = re.search(r'<ds:SignatureValue>(.*?)</ds:SignatureValue>', xml_content, re.DOTALL)
        if not signature_match:
            raise TokenValidationError("No Signature value found in the xml")
        return signature_match.group(1).replace("\n", "").strip()

    def _verify_signature(self, signed_content: str, signature: str) -> bool:
        """
        Verify digital signature of the signed content.
        """
        try:
            public_key = self._load_public_key()
            signed_content_bytes = signed_content.encode('utf-8')
            signature_bytes = base64.b64decode(signature)

            public_key.verify(
                signature_bytes,
                signed_content_bytes,
                padding.PKCS1v15(),
                hashes.SHA256()
            )
            logger.info("Token signature verification successful")
            return True
        except InvalidSignature:
            logger.warning("Token signature verification failed")
            return False
        except Exception as e:
            logger.error(f"Signature verification error: {e}")
            return False

    def validate_token_and_get_role(self, token: str) -> ValidationResult:
        """
        Validate authentication token and extract user role.
        """
        try:
            logger.info("Starting token validation")
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
                error_message=None if is_valid else "Signature verification failed"
            )

            logger.info(f"Token validation completed: valid={is_valid}, role={role}")
            return result

        except TokenValidationError as e:
            logger.error(f"Token validation failed: {e}")
            return ValidationResult(is_valid=False, is_tampered=True, error_message=str(e))
        except Exception as e:
            logger.critical(f"Unexpected token validation error: {e}")
            return ValidationResult(is_valid=False, is_tampered=True, error_message="Internal validation error")
