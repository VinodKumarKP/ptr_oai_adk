import pytest
import os
from unittest.mock import patch, MagicMock

from oai_agent_core.utils.env_resolver import (
    EnvResolver,
    ConfigResolver,
    EnvResolutionError,
    CallableResolutionError
)

def test_env_resolver_basic():
    resolver = EnvResolver(strict=True)
    
    with patch.dict(os.environ, {"TEST_VAR": "resolved_val"}):
        assert resolver.resolve_value("${TEST_VAR}") == "resolved_val"
        assert resolver.resolve_value("$TEST_VAR") == "resolved_val"
        assert resolver.resolve_value("Value: ${TEST_VAR}") == "Value: resolved_val"
        
        # Default value
        assert resolver.resolve_value("${TEST_VAR:-default_val}") == "resolved_val"
        assert resolver.resolve_value("${NONEXISTENT:-default_val}") == "default_val"

def test_env_resolver_strict_vs_lenient():
    # Strict mode should raise error
    resolver_strict = EnvResolver(strict=True)
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(EnvResolutionError):
            resolver_strict.resolve_value("${REQUIRED_VAR}")
            
    # Lenient mode should keep the placeholder
    resolver_lenient = EnvResolver(strict=False)
    with patch.dict(os.environ, {}, clear=True):
        assert resolver_lenient.resolve_value("${REQUIRED_VAR}") == "${REQUIRED_VAR}"

def test_env_resolver_recursive():
    resolver = EnvResolver(strict=True)
    
    config = {
        "key1": "Value ${MY_VAR}",
        "key2": ["list_val", "$MY_VAR"],
        "key3": {
            "nested_key": "${MY_VAR:-default}"
        },
        "non_string": 123
    }
    
    with patch.dict(os.environ, {"MY_VAR": "ok"}):
        resolved = resolver.resolve(config)
        assert resolved["key1"] == "Value ok"
        assert resolved["key2"] == ["list_val", "ok"]
        assert resolved["key3"]["nested_key"] == "ok"
        assert resolved["non_string"] == 123
        
        # Test resolve_dict convenience wrapper
        resolved_dict = resolver.resolve_dict(config)
        assert resolved_dict["key1"] == "Value ok"

def test_config_resolver_builtins():
    resolver = ConfigResolver()
    
    # filter:endswith
    f_endswith = resolver.resolve_dict({"filter": "filter:endswith:.md,.txt"})["filter"]
    assert f_endswith("test.md") is True
    assert f_endswith("test.py") is False
    
    # filter:startswith
    f_startswith = resolver.resolve_dict({"filter": "filter:startswith:docs/,src/"})["filter"]
    assert f_startswith("docs/intro.md") is True
    assert f_startswith("tests/intro.md") is False
    
    # filter:regex
    f_regex = resolver.resolve_dict({"filter": "filter:regex:^.*_test\\.py$"})["filter"]
    assert f_regex("module_test.py") is True
    assert f_regex("module.py") is False
    
    # filter:contains
    f_contains = resolver.resolve_dict({"filter": "filter:contains:secret"})["filter"]
    assert f_contains("my_secret_token") is True
    assert f_contains("my_token") is False

def test_config_resolver_invalid_filter():
    resolver = ConfigResolver()
    with pytest.raises(CallableResolutionError):
        resolver.resolve("filter:invalid_type:arg")
        
    with pytest.raises(CallableResolutionError):
        resolver.resolve("filter:endswith")  # Missing argument

def test_config_resolver_dotted_path():
    resolver = ConfigResolver()
    
    # Callable exists
    resolved = resolver.resolve("callable:os.path.join")
    assert resolved is os.path.join
    
    # Dotted path malformed
    with pytest.raises(CallableResolutionError):
        resolver.resolve("callable:os_path_join")
        
    # Module does not exist
    with pytest.raises(CallableResolutionError):
        resolver.resolve("callable:nonexistent.module.func")
        
    # Attribute does not exist
    with pytest.raises(CallableResolutionError):
        resolver.resolve("callable:os.path.nonexistent_attr")
        
    # Attribute is not callable
    with pytest.raises(CallableResolutionError):
        resolver.resolve("callable:os.name")

def test_config_resolver_eval():
    # Disabled by default
    resolver_no_eval = ConfigResolver(allow_eval=False)
    with pytest.raises(CallableResolutionError):
        resolver_no_eval.resolve("eval:lambda x: x")
        
    # Enabled
    resolver_eval = ConfigResolver(allow_eval=True)
    func = resolver_eval.resolve("eval:lambda x: x + 1")
    assert func(2) == 3
    
    # Eval syntax error / execution failure
    with pytest.raises(CallableResolutionError):
        resolver_eval.resolve("eval:invalid python syntax here")
