
import pytest
from oai_platform_core.exceptions import OAIBaseException, AuthenticationException

def test_oaibaseexception_is_exception():
    """Test that OAIBaseException is a subclass of Exception."""
    assert issubclass(OAIBaseException, Exception)

def test_authenticationexception_subclass():
    """Test that AuthenticationException is a subclass of OAIBaseException."""
    assert issubclass(AuthenticationException, OAIBaseException)

def test_authenticationexception_defaults():
    """Test the default values of AuthenticationException."""
    exc = AuthenticationException()
    assert exc.reason == "Invalid or expired token"
    assert exc.status_code == 401
    assert str(exc) == "Invalid or expired token"

def test_authenticationexception_custom_values():
    """Test custom values for AuthenticationException."""
    exc = AuthenticationException(reason="Custom reason", status_code=403)
    assert exc.reason == "Custom reason"
    assert exc.status_code == 403
    assert str(exc) == "Custom reason"

def test_raise_and_catch_oaibaseexception():
    """Test that OAIBaseException can be raised and caught."""
    with pytest.raises(OAIBaseException):
        raise OAIBaseException("Base exception")

def test_raise_and_catch_authenticationexception():
    """Test that AuthenticationException can be raised and caught."""
    with pytest.raises(AuthenticationException) as excinfo:
        raise AuthenticationException(reason="Auth failed", status_code=400)
    
    assert excinfo.value.reason == "Auth failed"
    assert excinfo.value.status_code == 400
