import os
import pytest
from oai_mcp_server_core.core.context import RequestAwareEnviron, request_env

class TestRequestAwareEnviron:
    def test_get_original_env(self):
        """Test getting value from original environment."""
        os.environ["TEST_VAR"] = "original_value"
        wrapper = RequestAwareEnviron(os.environ)
        assert wrapper.get("TEST_VAR") == "original_value"
        del os.environ["TEST_VAR"]

    def test_get_request_env(self):
        """Test getting value from request context."""
        os.environ["TEST_VAR"] = "original_value"
        wrapper = RequestAwareEnviron(os.environ)
        
        token = request_env.set({"TEST_VAR": "request_value"})
        try:
            assert wrapper.get("TEST_VAR") == "request_value"
        finally:
            request_env.reset(token)
            del os.environ["TEST_VAR"]

    def test_case_insensitive_lookup(self):
        """Test case-insensitive lookup in request context."""
        wrapper = RequestAwareEnviron(os.environ)
        
        token = request_env.set({"TEST_VAR": "value"})
        try:
            assert wrapper.get("test_var") == "value"
            assert wrapper["test_var"] == "value"
        finally:
            request_env.reset(token)

    def test_contains(self):
        """Test __contains__ behavior."""
        os.environ["ORIGINAL"] = "1"
        wrapper = RequestAwareEnviron(os.environ)
        
        token = request_env.set({"REQUEST": "2"})
        try:
            assert "ORIGINAL" in wrapper
            assert "REQUEST" in wrapper
            assert "MISSING" not in wrapper
        finally:
            request_env.reset(token)
            del os.environ["ORIGINAL"]

    def test_iteration(self):
        """Test iteration over keys."""
        # Clear env for clean test or mock it, but here we just check inclusion
        wrapper = RequestAwareEnviron(os.environ)
        token = request_env.set({"UNIQUE_REQ_VAR": "1"})
        
        try:
            keys = list(wrapper)
            assert "UNIQUE_REQ_VAR" in keys
            assert "PATH" in keys  # Assuming PATH is always in os.environ
        finally:
            request_env.reset(token)
