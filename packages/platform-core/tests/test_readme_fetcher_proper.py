"""Proper coverage for readme_fetcher module."""
import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from tempfile import TemporaryDirectory

import pytest

from oai_platform_core.readme_fetcher import (
    invalidate_readme_cache,
    clear_readme_cache,
    readme_cache_stats,
    read_local_readme,
    _parse_github_repo,
    _raw_url,
)


class TestCacheFunctions:
    """Test cache management functions."""

    def test_clear_cache(self):
        """Test clearing cache."""
        # Populate cache
        from oai_platform_core import readme_fetcher
        readme_fetcher._readme_cache["test"] = {"content": "test"}

        count = clear_readme_cache()
        assert count >= 1
        assert "test" not in readme_fetcher._readme_cache

    def test_invalidate_cache_exists(self):
        """Test invalidating existing cache entry."""
        from oai_platform_core import readme_fetcher
        readme_fetcher._readme_cache["test_key"] = {"content": "test"}

        result = invalidate_readme_cache("test_key")
        assert result is True

    def test_invalidate_cache_missing(self):
        """Test invalidating non-existent cache entry."""
        result = invalidate_readme_cache("nonexistent_key_xyz")
        assert result is False

    def test_cache_stats(self):
        """Test getting cache stats."""
        clear_readme_cache()  # Start fresh
        stats = readme_cache_stats()
        assert "ttl_seconds" in stats
        assert "entries" in stats
        assert isinstance(stats["entries"], list)


class TestLocalReadme:
    """Test local README reading."""

    def test_read_local_readme_found(self):
        """Test reading existing local README."""
        with TemporaryDirectory() as tmpdir:
            # Create test README
            item_dir = Path(tmpdir) / "test_item"
            item_dir.mkdir()
            readme_file = item_dir / "README.md"
            readme_file.write_text("# Test README")

            content, meta = read_local_readme(tmpdir, "test_item")
            assert content == "# Test README"
            assert meta["local"] is True
            assert meta["cached"] is False

    def test_read_local_readme_not_found(self):
        """Test reading non-existent README."""
        with TemporaryDirectory() as tmpdir:
            content, meta = read_local_readme(tmpdir, "nonexistent")
            assert content is None
            assert meta["local"] is True

    def test_read_local_readme_cache_hit(self):
        """Test cache hit for local README."""
        clear_readme_cache()
        with TemporaryDirectory() as tmpdir:
            item_dir = Path(tmpdir) / "test_item"
            item_dir.mkdir()
            readme_file = item_dir / "README.md"
            readme_file.write_text("# Test")

            # First read (cache miss)
            content1, meta1 = read_local_readme(tmpdir, "test_item")
            assert meta1["cached"] is False

            # Second read (cache hit)
            content2, meta2 = read_local_readme(tmpdir, "test_item")
            assert content2 == content1
            assert meta2["cached"] is True


class TestGitHubHelpers:
    """Test GitHub URL parsing."""

    def test_parse_github_https(self):
        """Test parsing HTTPS GitHub URL."""
        url = "https://github.com/owner/repo.git"
        result = _parse_github_repo(url)
        assert result == "owner/repo"

    def test_parse_github_https_no_git(self):
        """Test parsing HTTPS GitHub URL without .git."""
        url = "https://github.com/owner/repo"
        result = _parse_github_repo(url)
        assert result == "owner/repo"

    def test_parse_github_ssh(self):
        """Test parsing SSH GitHub URL."""
        url = "git@github.com:owner/repo.git"
        result = _parse_github_repo(url)
        assert result == "owner/repo"

    def test_parse_github_invalid(self):
        """Test parsing non-GitHub URL."""
        url = "https://gitlab.com/owner/repo"
        result = _parse_github_repo(url)
        assert result is None

    def test_raw_url(self):
        """Test generating raw GitHub URL."""
        url = _raw_url("owner/repo", "main", "path/to/README.md")
        assert url == "https://raw.githubusercontent.com/owner/repo/main/path/to/README.md"
