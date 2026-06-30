"""Proper coverage for trusted_peers module."""
import os
from unittest.mock import patch

import pytest

from oai_platform_core.security.trusted_peers import (
    is_trusted_peer,
    trusted_subnets,
    TRUSTED_PEER_NAMES,
)


class TestTrustedPeerNames:
    """Test exact-match trusted names."""

    def test_loopback_ipv4(self):
        """Test IPv4 loopback."""
        assert is_trusted_peer("127.0.0.1") is True

    def test_loopback_ipv6(self):
        """Test IPv6 loopback."""
        assert is_trusted_peer("::1") is True

    def test_localhost_name(self):
        """Test localhost hostname."""
        assert is_trusted_peer("localhost") is True

    def test_docker_internal(self):
        """Test Docker internal name."""
        assert is_trusted_peer("host.docker.internal") is True

    def test_any_address(self):
        """Test 0.0.0.0."""
        assert is_trusted_peer("0.0.0.0") is True

    def test_untrusted_hostname(self):
        """Test untrusted hostname."""
        assert is_trusted_peer("untrusted.example.com") is False

    def test_empty_peer(self):
        """Test empty peer string."""
        assert is_trusted_peer("") is False

    def test_none_peer(self):
        """Test None as peer."""
        assert is_trusted_peer(None) is False


class TestTrustedSubnets:
    """Test subnet-based trust."""

    def test_default_docker_bridge(self):
        """Test Docker bridge subnet 172.17.x.x."""
        # Clear cache first
        trusted_subnets.cache_clear()
        assert is_trusted_peer("172.17.0.2") is True

    def test_docker_bridge_range(self):
        """Test Docker bridge full range."""
        trusted_subnets.cache_clear()
        # 172.16-31.x.x is trusted
        assert is_trusted_peer("172.20.0.1") is True
        assert is_trusted_peer("172.31.255.255") is True

    def test_docker_desktop_macos(self):
        """Test Docker Desktop gateway on macOS."""
        trusted_subnets.cache_clear()
        assert is_trusted_peer("192.168.65.1") is True

    def test_untrusted_subnet(self):
        """Test untrusted subnet."""
        trusted_subnets.cache_clear()
        assert is_trusted_peer("10.0.0.1") is False

    def test_custom_trusted_subnet(self):
        """Test custom TRUSTED_SUBNETS environment variable."""
        trusted_subnets.cache_clear()

        with patch.dict(os.environ, {"TRUSTED_SUBNETS": "10.8.0.0/24,10.9.0.0/24"}):
            # Clear cache to pick up new env var
            trusted_subnets.cache_clear()
            assert is_trusted_peer("10.8.0.5") is True
            assert is_trusted_peer("10.9.0.100") is True
            assert is_trusted_peer("10.7.0.1") is False

    def test_invalid_custom_subnet(self):
        """Test invalid TRUSTED_SUBNETS entry."""
        trusted_subnets.cache_clear()

        with patch.dict(os.environ, {"TRUSTED_SUBNETS": "invalid-cidr,10.8.0.0/24"}):
            trusted_subnets.cache_clear()
            # Should skip invalid and use valid ones
            assert is_trusted_peer("10.8.0.1") is True


class TestTrustedSubnetsFunction:
    """Test trusted_subnets() directly."""

    def test_returns_tuple(self):
        """Test that trusted_subnets returns a tuple."""
        trusted_subnets.cache_clear()
        result = trusted_subnets()
        assert isinstance(result, tuple)
        assert len(result) > 0

    def test_includes_defaults(self):
        """Test that defaults are included."""
        trusted_subnets.cache_clear()
        result = trusted_subnets()
        # Should have at least the default subnets
        assert len(result) >= 2
