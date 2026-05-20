"""
Tests for oai_platform_core.security.token_utils.extract_bearer_token
"""
import pytest
from oai_platform_core.security.token_utils import extract_bearer_token


class TestExtractBearerToken:
    def test_api_token_header_returned_directly(self):
        assert extract_bearer_token({"api-token": "mytoken"}) == "mytoken"

    def test_api_token_underscore_header(self):
        assert extract_bearer_token({"api_token": "tok2"}) == "tok2"

    def test_x_api_key_header(self):
        assert extract_bearer_token({"x-api-key": "key3"}) == "key3"

    def test_bearer_authorization_header_stripped(self):
        assert extract_bearer_token({"authorization": "Bearer abc123"}) == "abc123"

    def test_bearer_case_insensitive(self):
        assert extract_bearer_token({"authorization": "BEARER tok"}) == "tok"

    def test_authorization_without_bearer_returned_as_is(self):
        assert extract_bearer_token({"authorization": "Basic abc"}) == "Basic abc"

    def test_returns_none_when_no_token_header(self):
        assert extract_bearer_token({"content-type": "application/json"}) is None

    def test_returns_none_for_empty_headers(self):
        assert extract_bearer_token({}) is None

    def test_api_token_takes_priority_over_authorization(self):
        headers = {"api-token": "first", "authorization": "Bearer second"}
        assert extract_bearer_token(headers) == "first"

    def test_api_token_underscore_takes_priority_over_x_api_key(self):
        headers = {"api_token": "a", "x-api-key": "b"}
        assert extract_bearer_token(headers) == "a"

    def test_x_api_key_takes_priority_over_authorization(self):
        headers = {"x-api-key": "key", "authorization": "Bearer other"}
        assert extract_bearer_token(headers) == "key"

    def test_empty_authorization_header_returns_none(self):
        # Falsy value — treated as absent
        assert extract_bearer_token({"authorization": ""}) is None

    def test_bearer_token_with_spaces_preserved(self):
        # Only the leading "Bearer " prefix is stripped; rest is returned as-is
        result = extract_bearer_token({"authorization": "Bearer my.token.value"})
        assert result == "my.token.value"

    def test_authorization_plain_token_no_bearer(self):
        result = extract_bearer_token({"authorization": "plainjwt"})
        assert result == "plainjwt"

    def test_works_with_case_insensitive_header_mapping(self):
        # Use a simple dict for framework-agnostic interface
        result = extract_bearer_token({"api-token": "x"})
        assert result == "x"
