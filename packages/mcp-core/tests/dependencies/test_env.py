import os
import pytest
from unittest.mock import MagicMock, patch
from oai_mcp_server_core.dependencies.env import get_request_env, get_all_request_env
from oai_mcp_server_core.core.context import request_env, RequestAwareEnviron

class TestEnvDependencies:
    
    def test_get_request_env_from_context(self):
        token = request_env.set({"TEST_KEY": "context_value"})
        try:
            assert get_request_env("TEST_KEY") == "context_value"
        finally:
            request_env.reset(token)

    def test_get_request_env_case_insensitive(self):
        token = request_env.set({"TEST_KEY": "context_value"})
        try:
            assert get_request_env("test_key") == "context_value"
        finally:
            request_env.reset(token)

    def test_get_request_env_fallback_to_os(self):
        with patch.dict(os.environ, {"OS_KEY": "os_value"}):
            # Ensure we are not using the wrapper for this test to verify fallback logic
            # But the code handles both wrapped and unwrapped os.environ
            assert get_request_env("OS_KEY") == "os_value"

    def test_get_request_env_default(self):
        assert get_request_env("MISSING_KEY", "default") == "default"

    def test_get_request_env_with_wrapped_environ(self):
        # Mock os.environ as RequestAwareEnviron
        original_environ = {"OS_KEY": "os_value"}
        wrapped_environ = RequestAwareEnviron(original_environ)
        
        with patch("os.environ", wrapped_environ):
            assert get_request_env("OS_KEY") == "os_value"

    def test_get_all_request_env(self):
        token = request_env.set({"CONTEXT_KEY": "context_value", "SHARED_KEY": "context_override"})
        try:
            with patch.dict(os.environ, {"OS_KEY": "os_value", "SHARED_KEY": "os_value"}):
                all_env = get_all_request_env()
                
                assert all_env["CONTEXT_KEY"] == "context_value"
                assert all_env["OS_KEY"] == "os_value"
                assert all_env["SHARED_KEY"] == "context_override"
        finally:
            request_env.reset(token)

    def test_get_all_request_env_with_wrapped_environ(self):
        original_environ = {"OS_KEY": "os_value"}
        wrapped_environ = RequestAwareEnviron(original_environ)
        
        token = request_env.set({"CONTEXT_KEY": "context_value"})
        try:
            with patch("os.environ", wrapped_environ):
                all_env = get_all_request_env()
                assert all_env["OS_KEY"] == "os_value"
                assert all_env["CONTEXT_KEY"] == "context_value"
        finally:
            request_env.reset(token)
