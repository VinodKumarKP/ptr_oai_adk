"""Security hardening utilities for the OAI Agent Framework.

This module provides:
1. Secure shell command execution
2. Credential validation and warnings
3. Path safety checks
4. Security audit functions

Best Practices:
- Always validate user input before execution
- Warn on insecure patterns (plain-text credentials)
- Use whitelists instead of blacklists
- Never execute untrusted shell commands
"""

import os
import logging
import re
import shlex
from pathlib import Path
from typing import List, Optional, Dict, Any

# ============================================================================
# Constants
# ============================================================================

# Safe read-only shell commands allowed for execution
SHELL_COMMAND_WHITELIST = {
    # File operations
    'ls', 'dir', 'find', 'grep', 'cat', 'head', 'tail', 'wc', 'file',
    # Information
    'echo', 'date', 'whoami', 'pwd', 'uname',
    # Text processing
    'sed', 'awk', 'cut', 'sort', 'uniq', 'tr',
}

# Pattern for detecting plain-text credentials in configs
CREDENTIAL_PATTERNS = {
    'api_key': r'api[_-]?key\s*[:=]\s*[\'"]?([^\'"\s,}]+)',
    'password': r'pass(?:word)?\s*[:=]\s*[\'"]?([^\'"\s,}]+)',
    'token': r'token\s*[:=]\s*[\'"]?([^\'"\s,}]+)',
    'secret': r'secret\s*[:=]\s*[\'"]?([^\'"\s,}]+)',
    'aws_key': r'aws[_-]?(?:access[_-])?key\s*[:=]\s*[\'"]?([A-Z0-9]{20,})',
}

# Environment variables for security settings
ENV_SHELL_ENABLED = 'AGENT_SHELL_ENABLED'


# ============================================================================
# Shell Command Security
# ============================================================================

class ShellCommandValidator:
    """Validates shell commands for safety."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def is_shell_enabled(self) -> bool:
        """Check if shell execution is enabled via environment.
        
        Returns:
            True if AGENT_SHELL_ENABLED=true
            
        Note:
            Shell execution is DISABLED by default for security.
            Must be explicitly enabled in production.
        """
        return os.environ.get(ENV_SHELL_ENABLED, '').lower() == 'true'
    
    def validate_command(self, command: str) -> bool:
        """Validate a shell command for safety.
        
        Args:
            command: The shell command to validate
            
        Returns:
            True if command is safe to execute
            
        Raises:
            PermissionError: If command violates security policy
        """
        if not self.is_shell_enabled():
            raise PermissionError(
                "Shell command execution is disabled. "
                "Set AGENT_SHELL_ENABLED=true to enable."
            )
        
        # Extract the main command (first word)
        try:
            main_command = shlex.split(command)[0]
        except ValueError:
            raise PermissionError(f"Malformed shell command: {command}")
        
        # Check against whitelist
        if main_command not in SHELL_COMMAND_WHITELIST:
            raise PermissionError(
                f"Command '{main_command}' is not allowed. "
                f"Allowed commands: {', '.join(sorted(SHELL_COMMAND_WHITELIST))}"
            )
        
        # Additional checks for dangerous patterns
        dangerous_patterns = [
            r';\s*(?:rm|dd|mkfs|:(){:|:|,|;)',  # Command chaining to dangerous commands
            r'[|&<>].*(?:rm|dd|mkfs)',  # Piping to dangerous commands
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                raise PermissionError(
                    f"Command contains dangerous pattern: {pattern}"
                )
        
        return True


# ============================================================================
# Credential Detection
# ============================================================================

class CredentialDetector:
    """Detects and warns about plain-text credentials in configurations."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize detector.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def scan_config(self, config: Dict[str, Any]) -> List[str]:
        """Scan configuration for potential credential exposure.
        
        Args:
            config: Configuration dictionary to scan
            
        Returns:
            List of warning messages for found issues
        """
        warnings = []
        
        # Convert config to string for scanning
        config_str = str(config)
        
        for cred_type, pattern in CREDENTIAL_PATTERNS.items():
            matches = re.finditer(pattern, config_str, re.IGNORECASE)
            for match in matches:
                warnings.append(
                    f"Potential {cred_type} found in configuration. "
                    f"Consider using environment variables instead."
                )
        
        return warnings
    
    def check_config_security(self, config: Dict[str, Any]) -> None:
        """Check configuration security and log warnings.
        
        Args:
            config: Configuration to check
        """
        warnings = self.scan_config(config)
        
        for warning in warnings:
            self.logger.warning(f"⚠️  Security: {warning}")
    
    def get_safe_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Return configuration with credential values masked.
        
        Args:
            config: Original configuration
            
        Returns:
            New config with credentials replaced by '***REDACTED***'
        """
        import copy
        safe_config = copy.deepcopy(config)
        
        # Recursively mask credentials
        def mask_dict(d):
            for key, value in d.items():
                if isinstance(value, dict):
                    mask_dict(value)
                elif isinstance(value, str):
                    # Check if this is a credential field
                    if any(cred in key.lower() for cred in ['api_key', 'password', 'token', 'secret']):
                        d[key] = '***REDACTED***'
        
        mask_dict(safe_config)
        return safe_config


# ============================================================================
# Path Security
# ============================================================================

class PathValidator:
    """Validates file paths for safe access."""
    
    def __init__(self, allowed_roots: Optional[List[str]] = None,
                 logger: Optional[logging.Logger] = None):
        """Initialize validator.
        
        Args:
            allowed_roots: List of allowed root directories
            logger: Optional logger instance
        """
        self.allowed_roots = [Path(r).resolve() for r in (allowed_roots or [])]
        self.logger = logger or logging.getLogger(__name__)
    
    def is_safe_path(self, path: str) -> bool:
        """Check if a path is safe to access.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is safe, False otherwise
        """
        try:
            resolved = Path(path).resolve()
            
            # Check if path exists within allowed roots
            if self.allowed_roots:
                for root in self.allowed_roots:
                    try:
                        resolved.relative_to(root)
                        return True
                    except ValueError:
                        continue
                return False
            
            # No restrictions if no roots specified
            return True
            
        except (OSError, ValueError):
            return False
    
    def validate_path(self, path: str) -> Path:
        """Validate and return a safe path.
        
        Args:
            path: Path to validate
            
        Returns:
            Validated Path object
            
        Raises:
            PermissionError: If path is not safe
        """
        if not self.is_safe_path(path):
            raise PermissionError(f"Path '{path}' is not allowed")
        
        return Path(path).resolve()


# ============================================================================
# Security Audit
# ============================================================================

class SecurityAudit:
    """Audit and report on security settings."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize auditor.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.shell_validator = ShellCommandValidator(logger)
        self.credential_detector = CredentialDetector(logger)
    
    def audit_environment(self) -> Dict[str, Any]:
        """Audit current environment security settings.
        
        Returns:
            Dictionary of security settings
        """
        return {
            'shell_enabled': self.shell_validator.is_shell_enabled(),
            'python_version': os.__name__,  # Just an example
        }
    
    def audit_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Audit configuration security.
        
        Args:
            config: Configuration to audit
            
        Returns:
            Audit results
        """
        warnings = self.credential_detector.scan_config(config)
        
        return {
            'credential_warnings': len(warnings),
            'warnings': warnings,
        }
    
    def audit_report(self, config: Dict[str, Any]) -> str:
        """Generate security audit report.
        
        Args:
            config: Configuration to audit
            
        Returns:
            Human-readable audit report
        """
        env_audit = self.audit_environment()
        config_audit = self.audit_config(config)
        
        report = "SECURITY AUDIT REPORT\n"
        report += "=" * 50 + "\n\n"
        
        report += "Environment Settings:\n"
        report += f"  Shell Execution Enabled: {env_audit['shell_enabled']}\n\n"
        
        report += "Configuration Security:\n"
        if config_audit['credential_warnings']:
            report += f"  ⚠️  {config_audit['credential_warnings']} credential warnings found\n"
            for warning in config_audit['warnings']:
                report += f"    - {warning}\n"
        else:
            report += "  ✅ No obvious credential issues detected\n"
        
        return report


# ============================================================================
# Security Helpers
# ============================================================================

def log_security_warning(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a security warning.
    
    Args:
        message: Warning message
        logger: Optional logger instance
    """
    _logger = logger or logging.getLogger(__name__)
    _logger.warning(f"🔒 SECURITY: {message}")


def log_security_error(message: str, logger: Optional[logging.Logger] = None) -> None:
    """Log a security error.
    
    Args:
        message: Error message
        logger: Optional logger instance
    """
    _logger = logger or logging.getLogger(__name__)
    _logger.error(f"🔒 SECURITY ERROR: {message}")
