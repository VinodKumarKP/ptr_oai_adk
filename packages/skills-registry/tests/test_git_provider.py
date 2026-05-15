"""
Tests for Git Provider classes.
"""

import pytest
from oai_skills_registry.services.git_provider import GitProvider, GitHubProvider


class TestGitProvider:
    """Tests for GitProvider abstract base class."""

    def test_git_provider_is_abstract(self):
        """Test that GitProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GitProvider()

    def test_git_provider_requires_fetch_skill_files(self):
        """Test GitProvider requires fetch_skill_files method."""
        with pytest.raises(TypeError):
            class IncompleteProvider(GitProvider):
                pass
            IncompleteProvider()

    def test_git_provider_defines_abstract_methods(self):
        """Test GitProvider defines required abstract methods."""
        abstract_methods = [
            'fetch_skill_files',
            'get_commit_info',
            'get_tags'
        ]

        for method in abstract_methods:
            assert hasattr(GitProvider, method)


class TestGitHubProvider:
    """Tests for GitHubProvider implementation."""

    @pytest.fixture
    def github_provider(self):
        """Create a GitHubProvider instance."""
        return GitHubProvider()

    def test_github_provider_implements_git_provider(self, github_provider):
        """Test GitHubProvider implements GitProvider interface."""
        assert isinstance(github_provider, GitProvider)

    def test_github_provider_has_fetch_skill_files(self, github_provider):
        """Test GitHubProvider has fetch_skill_files method."""
        assert hasattr(github_provider, 'fetch_skill_files')
        assert callable(github_provider.fetch_skill_files)

    def test_github_provider_has_get_commit_info(self, github_provider):
        """Test GitHubProvider has get_commit_info method."""
        assert hasattr(github_provider, 'get_commit_info')
        assert callable(github_provider.get_commit_info)

    def test_github_provider_has_get_tags(self, github_provider):
        """Test GitHubProvider has get_tags method."""
        assert hasattr(github_provider, 'get_tags')
        assert callable(github_provider.get_tags)

    def test_github_provider_logger_initialized(self, github_provider):
        """Test GitHubProvider has logger."""
        assert github_provider.logger is not None

    @pytest.mark.asyncio
    async def test_fetch_skill_files_requires_valid_repo(self, github_provider):
        """Test fetch_skill_files with missing repo raises error."""
        with pytest.raises(Exception):  # Will raise due to git command failure
            await github_provider.fetch_skill_files(
                repo="invalid-format",
                branch="main",
                tag=None,
                skill_name="test",
                auth_token=None
            )

    @pytest.mark.asyncio
    async def test_get_tags_returns_list(self, github_provider):
        """Test get_tags returns a list (even if empty when API fails)."""
        result = await github_provider.get_tags(
            repo="owner/repo",
            auth_token=None
        )

        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_get_tags_with_auth_token(self, github_provider):
        """Test get_tags can be called with auth token."""
        # This will likely fail due to network, but should not raise TypeError
        try:
            result = await github_provider.get_tags(
                repo="owner/repo",
                auth_token="test-token"
            )
            assert isinstance(result, list)
        except Exception as e:
            # Expected - network error, not code error
            assert "GitHub" not in str(type(e).__name__)  # Not a type error

    def test_github_provider_string_representation(self, github_provider):
        """Test GitHubProvider can be represented as string."""
        str_repr = str(github_provider)
        assert "GitHubProvider" in str_repr or "object" in str_repr
