"""Tests for backend configuration models."""

import pytest
from pydantic import ValidationError

from oai_agent_core.langgraph_core.config import (
    BackendConfig,
    CrewConfig,
    FilesystemBackendConfig,
    StoreBackendConfig,
)


class TestFilesystemBackendConfig:
    """Test FilesystemBackendConfig model."""

    def test_filesystem_defaults(self):
        """Filesystem backend should use defaults."""
        config = FilesystemBackendConfig()
        assert config.type == "filesystem"
        assert config.root_dir == "./workspace"
        assert config.virtual_mode is None
        assert config.max_file_size_mb is None

    def test_filesystem_custom_root_dir(self):
        """Filesystem backend should accept custom root_dir."""
        config = FilesystemBackendConfig(root_dir="/custom/workspace")
        assert config.root_dir == "/custom/workspace"

    def test_filesystem_to_deepagents_format(self):
        """Filesystem config should convert to deepagents format."""
        config = FilesystemBackendConfig(
            root_dir="/workspace",
            virtual_mode=True,
            max_file_size_mb=5120
        )
        result = config.to_deepagents_format()
        # 'type' should be excluded - it's not used by FilesystemBackend
        assert "type" not in result
        assert result["root_dir"] == "/workspace"
        assert result["virtual_mode"] is True
        assert result["max_file_size_mb"] == 5120


class TestStoreBackendConfig:
    """Test StoreBackendConfig model."""

    def test_store_defaults(self):
        """Store backend should use defaults."""
        config = StoreBackendConfig()
        assert config.type == "store"

    def test_store_to_deepagents_format(self):
        """Store config should convert to deepagents format."""
        config = StoreBackendConfig()
        result = config.to_deepagents_format()
        # 'type' should be excluded - it's not used by StoreBackend
        assert "type" not in result


class TestBackendConfig:
    """Test BackendConfig model."""

    def test_backend_state_default(self):
        """Backend should default to state type."""
        config = BackendConfig()
        assert config.type == "state"
        assert config.filesystem is None
        assert config.store is None

    def test_backend_state_to_deepagents_format(self):
        """State backend should return None (default)."""
        config = BackendConfig(type="state")
        result = config.to_deepagents_format()
        assert result is None

    def test_backend_filesystem_creation(self):
        """Backend should accept filesystem config."""
        filesystem_config = FilesystemBackendConfig(root_dir="./my_workspace")
        config = BackendConfig(
            type="filesystem",
            filesystem=filesystem_config
        )
        assert config.type == "filesystem"
        assert config.filesystem is not None

    def test_backend_filesystem_to_deepagents_format(self):
        """Filesystem backend should convert correctly."""
        config = BackendConfig(
            type="filesystem",
            filesystem=FilesystemBackendConfig(root_dir="./workspace")
        )
        result = config.to_deepagents_format()
        # 'type' should be excluded from backend config
        assert "type" not in result
        assert result["root_dir"] == "./workspace"

    def test_backend_filesystem_missing_config_raises(self):
        """Filesystem type without config should raise."""
        config = BackendConfig(type="filesystem")
        with pytest.raises(ValueError, match="filesystem backend selected"):
            config.to_deepagents_format()

    def test_backend_store_creation(self):
        """Backend should accept store config."""
        config = BackendConfig(
            type="store",
            store=StoreBackendConfig()
        )
        assert config.type == "store"
        assert config.store is not None

    def test_backend_store_to_deepagents_format(self):
        """Store backend should convert correctly."""
        config = BackendConfig(
            type="store",
            store=StoreBackendConfig()
        )
        result = config.to_deepagents_format()
        # 'type' should be excluded from backend config
        assert "type" not in result
        assert result == {}

    def test_backend_store_missing_config_raises(self):
        """Store type without config should raise."""
        config = BackendConfig(type="store")
        with pytest.raises(ValueError, match="store backend selected"):
            config.to_deepagents_format()

    def test_backend_invalid_type_raises(self):
        """Invalid backend type should raise ValidationError."""
        with pytest.raises(ValidationError):
            BackendConfig(type="invalid_type")


class TestCrewConfig:
    """Test CrewConfig model."""

    def test_crew_config_defaults(self):
        """CrewConfig should use sensible defaults."""
        config = CrewConfig()
        assert config.pattern == "deep"
        assert config.enable_lazy_loading is False
        assert config.backend is None
        assert config.name is None
        assert config.tools is None
        assert config.mcps is None

    def test_crew_config_with_backend(self):
        """CrewConfig should accept backend configuration."""
        backend = BackendConfig(
            type="filesystem",
            filesystem=FilesystemBackendConfig(root_dir="./workspace")
        )
        config = CrewConfig(pattern="deep", backend=backend)
        assert config.backend is not None
        assert config.backend.type == "filesystem"

    def test_crew_config_to_deepagents_format_minimal(self):
        """Minimal CrewConfig should convert correctly."""
        config = CrewConfig()
        result = config.to_deepagents_format()
        assert result["pattern"] == "deep"
        assert result["enable_lazy_loading"] is False
        # backend should not be in dict if None

    def test_crew_config_to_deepagents_format_with_backend(self):
        """CrewConfig with backend should include it in format."""
        config = CrewConfig(
            pattern="deep",
            backend=BackendConfig(
                type="filesystem",
                filesystem=FilesystemBackendConfig(root_dir="./my_workspace")
            ),
            enable_lazy_loading=True,
            name="root_agent",
            tools=["tool1", "tool2"],
            mcps=["mcp1"]
        )
        result = config.to_deepagents_format()
        assert result["pattern"] == "deep"
        assert result["enable_lazy_loading"] is True
        # 'type' should be excluded from backend config
        assert "type" not in result["backend"]
        assert result["backend"]["root_dir"] == "./my_workspace"
        assert result["name"] == "root_agent"
        assert result["tools"] == ["tool1", "tool2"]
        assert result["mcps"] == ["mcp1"]

    def test_crew_config_from_dict(self):
        """CrewConfig should accept dict initialization."""
        config_dict = {
            "pattern": "deep",
            "backend": {
                "type": "filesystem",
                "filesystem": {"root_dir": "./workspace"}
            }
        }
        config = CrewConfig(**config_dict)
        assert config.pattern == "deep"
        assert config.backend.type == "filesystem"

    def test_crew_config_state_backend_not_in_output(self):
        """State backend should not appear in deepagents format."""
        config = CrewConfig(backend=BackendConfig(type="state"))
        result = config.to_deepagents_format()
        # state backend returns None, so not included in output
        assert "backend" not in result or result.get("backend") is None
