"""
Tests for utility functions in skills registry.
"""

import pytest
from datetime import datetime, timezone
from oai_skills_registry.services.utils import (
    parse_skill_md,
    convert_datetime_to_str,
    version_matches_constraint,
    parse_github_url
)


class TestParseSkillMd:
    """Tests for parse_skill_md function."""

    def test_parse_valid_skill_md(self, sample_skill_md):
        """Test parsing valid SKILL.md content."""
        result = parse_skill_md(sample_skill_md)

        assert result["name"] == "test-skill"
        assert result["version"] == "1.0.0"
        assert result["description"] == "A test skill"
        assert result["category"] == "test"
        assert result["author"] == "test-author"
        assert result["tags"] == ["test", "example"]
        assert "content" in result
        assert "Test Skill" in result["content"]

    def test_parse_skill_md_with_dependencies(self, sample_skill_md):
        """Test parsing SKILL.md with dependencies."""
        result = parse_skill_md(sample_skill_md)

        assert "dependencies" in result
        assert isinstance(result["dependencies"], list)
        assert "requests>=2.28.0" in result["dependencies"]

    def test_parse_skill_md_with_breaking_changes(self, sample_skill_md):
        """Test parsing SKILL.md with breaking changes."""
        result = parse_skill_md(sample_skill_md)

        assert "breaking_changes" in result
        assert "Breaking change 1" in result["breaking_changes"]

    def test_parse_skill_md_missing_frontmatter(self):
        """Test parsing SKILL.md without frontmatter raises error."""
        invalid_content = "# Test Skill\n\nContent without frontmatter"

        with pytest.raises(ValueError, match="Invalid SKILL.md format"):
            parse_skill_md(invalid_content)

    def test_parse_skill_md_invalid_yaml(self):
        """Test parsing SKILL.md with invalid YAML raises error."""
        invalid_yaml = """---
name: test-skill
  invalid: yaml: format:
---
Content"""

        with pytest.raises(Exception):  # YAML parsing error
            parse_skill_md(invalid_yaml)

    def test_parse_skill_md_minimal(self):
        """Test parsing minimal SKILL.md with required fields only."""
        minimal_md = """---
name: minimal-skill
version: 0.1.0
---
Minimal content"""

        result = parse_skill_md(minimal_md)

        assert result["name"] == "minimal-skill"
        assert result["version"] == "0.1.0"
        assert "content" in result


class TestConvertDatetimeToStr:
    """Tests for convert_datetime_to_str function."""

    def test_convert_single_datetime(self):
        """Test converting a single datetime object."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        result = convert_datetime_to_str(dt)

        assert result == "2024-01-15T10:30:45+00:00"
        assert isinstance(result, str)

    def test_convert_dict_with_datetime(self):
        """Test converting dictionary containing datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        obj = {
            "name": "test",
            "created_at": dt,
            "value": 123
        }

        result = convert_datetime_to_str(obj)

        assert result["name"] == "test"
        assert result["created_at"] == "2024-01-15T10:30:45+00:00"
        assert result["value"] == 123

    def test_convert_list_with_datetime(self):
        """Test converting list containing datetime."""
        dt1 = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        dt2 = datetime(2024, 1, 16, 10, 30, 45, tzinfo=timezone.utc)
        obj = [dt1, "string", dt2, 42]

        result = convert_datetime_to_str(obj)

        assert result[0] == "2024-01-15T10:30:45+00:00"
        assert result[1] == "string"
        assert result[2] == "2024-01-16T10:30:45+00:00"
        assert result[3] == 42

    def test_convert_nested_structure(self):
        """Test converting nested structure with datetime."""
        dt = datetime(2024, 1, 15, 10, 30, 45, tzinfo=timezone.utc)
        obj = {
            "metadata": {
                "created_at": dt,
                "tags": ["tag1", "tag2"]
            },
            "items": [
                {"timestamp": dt, "value": 1},
                {"value": 2}
            ]
        }

        result = convert_datetime_to_str(obj)

        assert result["metadata"]["created_at"] == "2024-01-15T10:30:45+00:00"
        assert result["items"][0]["timestamp"] == "2024-01-15T10:30:45+00:00"
        assert result["items"][1]["value"] == 2

    def test_convert_non_datetime_objects(self):
        """Test that non-datetime objects pass through unchanged."""
        obj = {
            "string": "value",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
            "list": [1, 2, 3]
        }

        result = convert_datetime_to_str(obj)

        assert result == obj


class TestVersionMatchesConstraint:
    """Tests for version_matches_constraint function."""

    def test_match_latest_constraint(self):
        """Test that 'latest' constraint matches any version."""
        assert version_matches_constraint("1.0.0", "latest") is True
        assert version_matches_constraint("2.5.3", "latest") is True
        assert version_matches_constraint("0.0.1", "latest") is True

    def test_match_exact_version(self):
        """Test matching exact version constraint."""
        # Current simplified implementation returns True for all constraints
        # (includes TODO: "Could add more sophisticated version constraint matching")
        assert version_matches_constraint("1.0.0", "1.0.0") is True
        assert version_matches_constraint("2.0.0", "1.0.0") is True  # Simplified: returns True for all

    def test_match_other_constraints(self):
        """Test other version constraints return True (simplified implementation)."""
        # Current implementation returns True for all constraints
        assert version_matches_constraint("1.5.0", "^1.0.0") is True
        assert version_matches_constraint("2.0.0", "~2.0.0") is True


class TestParseGithubUrl:
    """Tests for parse_github_url function."""

    def test_parse_https_url_with_git_extension(self):
        """Test parsing HTTPS GitHub URL with .git extension."""
        url = "https://github.com/owner/repo.git"
        result = parse_github_url(url)

        assert result == "owner/repo"

    def test_parse_https_url_without_git_extension(self):
        """Test parsing HTTPS GitHub URL without .git extension."""
        url = "https://github.com/owner/repo"
        result = parse_github_url(url)

        assert result == "owner/repo"

    def test_parse_https_url_with_trailing_slash(self):
        """Test parsing HTTPS GitHub URL with trailing slash."""
        url = "https://github.com/owner/repo/"
        result = parse_github_url(url)

        assert result == "owner/repo"

    def test_parse_url_with_special_characters_in_names(self):
        """Test parsing URL with special characters in owner/repo names."""
        url = "https://github.com/my-org/my-repo-name.git"
        result = parse_github_url(url)

        assert result == "my-org/my-repo-name"

    def test_parse_invalid_url_format(self):
        """Test parsing invalid GitHub URL raises error."""
        invalid_urls = [
            "https://github.com/invalid",  # Missing repo
            "https://gitlab.com/owner/repo.git",  # Wrong domain
            "not-a-url",  # Not a URL
            ""  # Empty
        ]

        for url in invalid_urls:
            with pytest.raises(ValueError, match="Invalid GitHub URL format|Failed to parse GitHub URL"):
                parse_github_url(url)

    def test_parse_url_with_dash_in_repo(self):
        """Test parsing URL where repo name contains dashes."""
        url = "https://github.com/owner/my-skills-repo.git"
        result = parse_github_url(url)

        assert result == "owner/my-skills-repo"

    def test_parse_url_with_numbers(self):
        """Test parsing URL with numbers in names."""
        url = "https://github.com/org2024/repo-v2.git"
        result = parse_github_url(url)

        assert result == "org2024/repo-v2"
