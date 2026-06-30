import pytest
import logging
from unittest.mock import MagicMock

from oai_agent_core.services.config_validator import ConfigValidator

def test_config_validator_empty():
    validator = ConfigValidator()
    assert validator.validate_agent_config({}) is True
    assert len(validator.get_errors()) == 0
    assert len(validator.get_warnings()) == 0

def test_config_validator_missing_type_warning():
    validator = ConfigValidator()
    config = {
        "model": {"name": "gpt-4"}
    }
    assert validator.validate_agent_config(config) is True
    assert "missing 'type' field" in validator.get_warnings()[0]

def test_config_validator_invalid_model():
    validator = ConfigValidator()
    
    # Model must be dictionary
    assert validator.validate_agent_config({"model": "invalid_model_type"}) is False
    assert "'model' must be a dictionary" in validator.get_errors()

    # Real settings validation
    insecure_model_config = {
        "model": {
            "temperature": 0.7,
            # missing name & provider
        }
    }
    assert validator.validate_agent_config(insecure_model_config) is False
    assert any("missing required 'name'" in err for err in validator.get_errors())
    assert any("missing required 'provider'" in err for err in validator.get_errors())

    # Minimal settings check warning
    validator_warning = ConfigValidator()
    warning_model_config = {
        "model": {
            "name": 123,
            "provider": 456
        }
    }
    assert validator_warning.validate_agent_config(warning_model_config) is True
    assert any("should be a string" in w for w in validator_warning.get_warnings())

def test_config_validator_tools():
    validator = ConfigValidator()
    
    # Must be dict
    assert validator.validate_agent_config({"tools": "invalid"}) is False
    
    # Tool config must be dict
    assert validator.validate_agent_config({"tools": {"my_tool": "invalid"}}) is False
    
    # Missing module/command warning
    validator_warn = ConfigValidator()
    assert validator_warn.validate_agent_config({"tools": {"my_tool": {}}}) is True
    assert any("missing 'module'" in w for w in validator_warn.get_warnings())

def test_config_validator_kb():
    validator = ConfigValidator()
    
    # Dictionary style source type warning
    assert validator.validate_agent_config({"knowledge_base": {"sources": "invalid"}}) is True
    assert any("sources' should be a list" in w for w in validator.get_warnings())
    
    # List style check
    validator_list = ConfigValidator()
    assert validator_list.validate_agent_config({"knowledge_base": ["invalid_entry"]}) is False
    assert any("must be a dictionary" in err for err in validator_list.get_errors())
    
    # Invalid overall type
    validator_invalid = ConfigValidator()
    assert validator_invalid.validate_agent_config({"knowledge_base": "invalid"}) is False
    assert any("must be a dictionary or list" in err for err in validator_invalid.get_errors())

def test_config_validator_skills():
    validator = ConfigValidator()
    
    # Must be dict
    assert validator.validate_agent_config({"skills": "invalid"}) is False
    
    # Missing skill_dir
    validator_missing = ConfigValidator()
    assert validator_missing.validate_agent_config({"skills": {"some_key": "some_val"}}) is False
    assert any("missing required 'skill_dir'" in err for err in validator_missing.get_errors())
    
    # Registry dict check
    validator_registry = ConfigValidator()
    assert validator_registry.validate_agent_config({"skills": {"skill_dir": "/path", "registry": "invalid"}}) is False
    assert any("registry' must be a dictionary" in err for err in validator_registry.get_errors())

def test_config_validator_memory():
    validator = ConfigValidator()
    
    # Must be dict
    assert validator.validate_agent_config({"memory": "invalid"}) is False
    
    # Warning for missing type
    validator_warn = ConfigValidator()
    assert validator_warn.validate_agent_config({"memory": {"some_key": "some_val"}}) is True
    assert any("missing 'type' field" in w for w in validator_warn.get_warnings())
    
    # Warning for unknown type
    validator_unknown = ConfigValidator()
    assert validator_unknown.validate_agent_config({"memory": {"type": "unknown_type"}}) is True
    assert any("Unknown memory type" in w for w in validator_unknown.get_warnings())

def test_config_validator_guardrails():
    validator = ConfigValidator()
    
    # Must be dict
    assert validator.validate_agent_config({"guardrails": "invalid"}) is False
    
    # Warning for missing guard or definition_file
    validator_warn = ConfigValidator()
    assert validator_warn.validate_agent_config({"guardrails": {"some_key": "some_val"}}) is True
    assert any("missing 'guard' or 'definition_file'" in w for w in validator_warn.get_warnings())

def test_config_validator_schema_export():
    validator = ConfigValidator()
    schema = validator.export_json_schema()
    assert schema["title"] == "Agent Configuration Schema"
    assert "type" in schema["properties"]
    assert "model" in schema["properties"]
