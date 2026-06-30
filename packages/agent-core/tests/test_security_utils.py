import pytest
import os
import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

from oai_agent_core.utils.security import (
    ShellCommandValidator,
    CredentialDetector,
    PathValidator,
    SecurityAudit,
    log_security_warning,
    log_security_error,
    ENV_SHELL_ENABLED
)

def test_shell_command_validator():
    validator = ShellCommandValidator()
    
    # Disabled by default
    with patch.dict(os.environ, {ENV_SHELL_ENABLED: "false"}):
        assert not validator.is_shell_enabled()
        with pytest.raises(PermissionError):
            validator.validate_command("ls")
            
    with patch.dict(os.environ, {}, clear=True):
        assert not validator.is_shell_enabled()
        with pytest.raises(PermissionError):
            validator.validate_command("ls")

    # Enabled
    with patch.dict(os.environ, {ENV_SHELL_ENABLED: "true"}):
        assert validator.is_shell_enabled()
        
        # Whitelisted commands
        assert validator.validate_command("ls -la") is True
        assert validator.validate_command("grep -rn 'hello'") is True
        
        # Not whitelisted command
        with pytest.raises(PermissionError):
            validator.validate_command("rm -rf /")
            
        # Malformed command
        with pytest.raises(PermissionError):
            validator.validate_command("ls 'unclosed quote")
            
        # Dangerous chaining patterns
        with pytest.raises(PermissionError):
            validator.validate_command("ls ; rm -rf")
            
        with pytest.raises(PermissionError):
            validator.validate_command("ls | dd")

class StringifiableDict(dict):
    def __str__(self):
        return "api_key: supersecret123\npassword: mypassword"

def test_credential_detector():
    detector = CredentialDetector()
    
    config_insecure = StringifiableDict({
        "api_key": "supersecret123",
        "nested": {
            "password": "mypassword"
        }
    })
    
    # Scan config
    warnings = detector.scan_config(config_insecure)
    assert len(warnings) == 2
    
    # Check config security
    mock_logger = MagicMock()
    detector_with_logger = CredentialDetector(logger=mock_logger)
    detector_with_logger.check_config_security(config_insecure)
    assert mock_logger.warning.call_count == 2
    
    # Get safe config
    safe_config = detector.get_safe_config(config_insecure)
    assert safe_config["api_key"] == "***REDACTED***"
    assert safe_config["nested"]["password"] == "***REDACTED***"

def test_path_validator(tmp_path):
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    
    validator = PathValidator(allowed_roots=[str(allowed_dir)])
    
    # Safe path inside root
    safe_file = allowed_dir / "test.txt"
    assert validator.is_safe_path(str(safe_file)) is True
    assert validator.validate_path(str(safe_file)) == safe_file.resolve()
    
    # Path outside root
    unsafe_file = tmp_path / "outside.txt"
    assert validator.is_safe_path(str(unsafe_file)) is False
    with pytest.raises(PermissionError):
        validator.validate_path(str(unsafe_file))
        
    # Invalid path syntax
    assert validator.is_safe_path("") is False

def test_security_audit():
    audit = SecurityAudit()
    
    config = StringifiableDict({
        "api_key": "secret"
    })
    
    with patch.dict(os.environ, {ENV_SHELL_ENABLED: "true"}):
        env_res = audit.audit_environment()
        assert env_res["shell_enabled"] is True
        
        config_res = audit.audit_config(config)
        assert config_res["credential_warnings"] == 2
        
        report = audit.audit_report(config)
        assert "SECURITY AUDIT REPORT" in report
        assert "Shell Execution Enabled: True" in report
        assert "credential warnings found" in report

def test_security_loggers():
    mock_logger = MagicMock()
    log_security_warning("Careful!", logger=mock_logger)
    mock_logger.warning.assert_called_once_with("🔒 SECURITY: Careful!")
    
    log_security_error("Crash!", logger=mock_logger)
    mock_logger.error.assert_called_once_with("🔒 SECURITY ERROR: Crash!")
