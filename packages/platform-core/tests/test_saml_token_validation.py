
import pytest
from unittest.mock import patch, MagicMock
from oai_platform_core.security.saml_token_validation import TokenValidator, ValidationResult, TokenValidationError, is_saml_token

@pytest.fixture
def token_validator():
    return TokenValidator(public_key_path="dummy_path")

def test_is_saml_token_valid():
    """Test is_saml_token with a valid-looking SAML token."""
    import base64
    valid_token = base64.b64encode(b"<saml:Assertion></saml:Assertion>").decode('utf-8')
    assert is_saml_token(valid_token) == True

def test_is_saml_token_invalid_base64():
    """Test is_saml_token with invalid base64."""
    assert is_saml_token("invalid_base64_string!") == False

def test_is_saml_token_not_saml():
    """Test is_saml_token with valid base64 but not SAML XML."""
    import base64
    non_saml = base64.b64encode(b"<other:XML></other:XML>").decode('utf-8')
    assert is_saml_token(non_saml) == False

def test_is_saml_token_empty():
    """Test is_saml_token with an empty string."""
    assert is_saml_token("") == False

def test_validation_result_to_dict():
    """Test ValidationResult to_dict conversion."""
    result = ValidationResult(role="admin", email="test@example.com", is_valid=True, is_tampered=False)
    assert result.to_dict() == {
        "role": "admin",
        "isValid": True,
        "isTampered": False
    }

def test_token_validator_get_default_key_path(monkeypatch):
    """Test default key path resolution order."""
    monkeypatch.setenv("SAML_PUBLIC_KEY_PATH", "/env/path/key.pem")
    validator = TokenValidator()
    assert validator._get_default_key_path() == "/env/path/key.pem"

@patch('os.path.exists')
@patch('builtins.open', new_callable=MagicMock)
@patch('oai_platform_core.security.saml_token_validation.serialization.load_pem_public_key')
def test_token_validator_load_public_key_success(mock_load_pem, mock_open, mock_exists, token_validator):
    """Test successful public key loading."""
    mock_exists.return_value = True
    from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
    mock_rsapubkey = MagicMock(spec=RSAPublicKey)
    mock_load_pem.return_value = mock_rsapubkey

    key = token_validator._load_public_key()
    assert key == mock_rsapubkey
    mock_load_pem.assert_called_once()

@patch('os.path.exists')
def test_token_validator_load_public_key_missing_file(mock_exists, token_validator):
    """Test public key loading when file is missing."""
    mock_exists.return_value = False
    with pytest.raises(TokenValidationError):
        token_validator._load_public_key()

def test_token_validator_validate_token_format_success(token_validator):
    """Test successful token format validation."""
    import base64
    valid_xml = "<samlp:Response></samlp:Response>"
    valid_token = base64.b64encode(valid_xml.encode('utf-8')).decode('utf-8')
    assert token_validator._validate_token_format(valid_token) == valid_xml

def test_token_validator_validate_token_format_invalid_xml(token_validator):
    """Test token format validation with invalid XML."""
    import base64
    invalid_xml = "<other:Response></other:Response>"
    invalid_token = base64.b64encode(invalid_xml.encode('utf-8')).decode('utf-8')
    with pytest.raises(TokenValidationError):
        token_validator._validate_token_format(invalid_token)

def test_token_validator_extract_signed_content(token_validator):
    """Test extracting signed content from XML."""
    xml = "<root><ds:Signature/><saml:Assertion>test</saml:Assertion></root>"
    assert token_validator._extract_signed_content(xml) == "<saml:Assertion>test</saml:Assertion>"

def test_token_validator_extract_signed_content_missing_signature(token_validator):
    """Test extracting signed content when signature is missing."""
    xml = "<root><saml:Assertion>test</saml:Assertion></root>"
    with pytest.raises(TokenValidationError):
        token_validator._extract_signed_content(xml)

def test_token_validator_extract_email(token_validator):
    """Test extracting email from XML."""
    xml = '<saml:Attribute Name="email"><saml:AttributeValue>test@example.com</saml:AttributeValue></saml:Attribute>'
    assert token_validator._extract_email(xml) == "test@example.com"

def test_token_validator_extract_role(token_validator):
    """Test extracting role from XML."""
    xml = '<saml:Attribute Name="role"><saml:AttributeValue>admin</saml:AttributeValue></saml:Attribute>'
    assert token_validator._extract_role(xml) == "admin"

def test_token_validator_extract_signature(token_validator):
    """Test extracting signature from XML."""
    xml = '<ds:SignatureValue>sig_value</ds:SignatureValue>'
    assert token_validator._extract_signature(xml) == "sig_value"

@patch.object(TokenValidator, '_load_public_key')
def test_token_validator_verify_signature_success(mock_load_key, token_validator):
    """Test successful signature verification."""
    mock_key = MagicMock()
    mock_load_key.return_value = mock_key
    assert token_validator._verify_signature("content", "c2ln") == True
    mock_key.verify.assert_called_once()

@patch.object(TokenValidator, '_load_public_key')
def test_token_validator_verify_signature_invalid(mock_load_key, token_validator):
    """Test failed signature verification."""
    from cryptography.exceptions import InvalidSignature
    mock_key = MagicMock()
    mock_key.verify.side_effect = InvalidSignature()
    mock_load_key.return_value = mock_key
    assert token_validator._verify_signature("content", "c2ln") == False

@patch.object(TokenValidator, '_validate_token_format')
@patch.object(TokenValidator, '_extract_signed_content')
@patch.object(TokenValidator, '_extract_role')
@patch.object(TokenValidator, '_extract_email')
@patch.object(TokenValidator, '_extract_signature')
@patch.object(TokenValidator, '_verify_signature')
def test_token_validator_validate_token_and_get_role_success(mock_verify, mock_extract_sig, mock_extract_email, mock_extract_role, mock_extract_content, mock_validate_format, token_validator):
    """Test successful token validation and role retrieval."""
    mock_validate_format.return_value = "xml"
    mock_extract_content.return_value = "content"
    mock_extract_role.return_value = "admin"
    mock_extract_email.return_value = "test@example.com"
    mock_extract_sig.return_value = "sig"
    mock_verify.return_value = True

    result = token_validator.validate_token_and_get_role("token")
    assert result.is_valid == True
    assert result.role == "admin"
    assert result.email == "test@example.com"
    assert result.is_tampered == False

@patch.object(TokenValidator, '_validate_token_format')
def test_token_validator_validate_token_and_get_role_failure(mock_validate_format, token_validator):
    """Test failed token validation and role retrieval."""
    mock_validate_format.side_effect = TokenValidationError("Invalid format")

    result = token_validator.validate_token_and_get_role("token")
    assert result.is_valid == False
    assert result.is_tampered == True
    assert result.error_message == "Invalid format"
