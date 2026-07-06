"""Crew configuration models for LangGraph agents."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class FilesystemBackendConfig(BaseModel):
    """Filesystem-based backend configuration for deep agents.

    Stores agent state and files in the local filesystem. Useful for
    development and single-machine deployments.

    Attributes:
        type: Backend type identifier (filesystem)
        root_dir: Root directory for workspace storage (default: ./workspace)
        virtual_mode: Use virtual filesystem (optional, default: None)
        max_file_size_mb: Maximum file size in megabytes (optional)
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["filesystem"] = Field(
        default="filesystem",
        description="Backend type identifier"
    )
    root_dir: str = Field(
        default="./workspace",
        description="Root directory for storing agent workspace and files"
    )
    virtual_mode: Optional[bool] = Field(
        default=None,
        description="Use virtual filesystem mode (optional)"
    )
    max_file_size_mb: Optional[int] = Field(
        default=None,
        description="Maximum file size in megabytes (None = unlimited)"
    )

    def to_deepagents_format(self) -> Dict[str, Any]:
        """Convert to deepagents-compatible format.

        Returns:
            Dictionary with backend configuration for deepagents
            (excludes 'type' field as it's not used by FilesystemBackend)
        """
        return self.model_dump(exclude_none=True, exclude={"type"})


class StoreBackendConfig(BaseModel):
    """Persistent store backend configuration for deep agents.

    Uses a persistent store (e.g., database, cache) for agent state.
    Useful for multi-machine deployments and production environments.

    Attributes:
        type: Backend type identifier (store)
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["store"] = Field(
        default="store",
        description="Backend type identifier"
    )

    def to_deepagents_format(self) -> Dict[str, Any]:
        """Convert to deepagents-compatible format.

        Returns:
            Dictionary with backend configuration for deepagents
            (excludes 'type' field as it's not used by StoreBackend)
        """
        return self.model_dump(exclude_none=True, exclude={"type"})


class BackendConfig(BaseModel):
    """Backend configuration for deep agents.

    Specifies how agent state and files are persisted. Supports three
    backend types:
    - state: In-memory (default)
    - filesystem: File-based storage
    - store: Persistent store

    Attributes:
        type: Backend type (state, filesystem, or store)
        filesystem: Configuration for filesystem backend
        store: Configuration for store backend
    """

    type: Literal["state", "filesystem", "store"] = Field(
        default="state",
        description="Backend type: state (in-memory), filesystem (file-based), or store (persistent)"
    )
    filesystem: Optional[FilesystemBackendConfig] = Field(
        default=None,
        description="Filesystem backend configuration (required if type=filesystem)"
    )
    store: Optional[StoreBackendConfig] = Field(
        default=None,
        description="Store backend configuration (required if type=store)"
    )

    def to_deepagents_format(self) -> Optional[Dict[str, Any]]:
        """Convert to deepagents-compatible format.

        Returns:
            Dictionary with backend configuration for deepagents, or None for state backend

        Raises:
            ValueError: If backend configuration is incomplete or invalid
        """
        if self.type == "state":
            return None  # state backend is default, no explicit config needed

        if self.type == "filesystem":
            if not self.filesystem:
                raise ValueError(
                    "filesystem backend selected but filesystem config not provided"
                )
            return self.filesystem.to_deepagents_format()

        if self.type == "store":
            if not self.store:
                raise ValueError(
                    "store backend selected but store config not provided"
                )
            return self.store.to_deepagents_format()

        raise ValueError(f"Unknown backend type: {self.type}")


class CrewConfig(BaseModel):
    """Crew configuration for agent orchestration.

    Configures how agents are orchestrated and executed. Supports both
    single-agent and deep-agent (multi-agent) patterns.

    Attributes:
        pattern: Orchestration pattern (deep, sequential, hierarchical)
        backend: Backend configuration for state and file persistence
        enable_lazy_loading: Whether to lazy-load MCP tools (default: False)
        name: Optional name for the root agent
        tools: Optional list of tool names for root agent
        mcps: Optional list of MCP server names for root agent
    """

    model_config = ConfigDict(extra="allow")

    pattern: str = Field(
        default="deep",
        description="Agent orchestration pattern: deep (multi-agent with delegation), sequential, or hierarchical"
    )
    backend: Optional[BackendConfig] = Field(
        default=None,
        description="Backend configuration for state and file persistence"
    )
    enable_lazy_loading: bool = Field(
        default=False,
        description="Enable lazy-loading of MCP tools (lighter schemas, execute-time binding)"
    )
    name: Optional[str] = Field(
        default=None,
        description="Optional name for the root agent"
    )
    tools: Optional[list[str]] = Field(
        default=None,
        description="Optional list of tool names available to root agent"
    )
    mcps: Optional[list[str]] = Field(
        default=None,
        description="Optional list of MCP server names available to root agent"
    )

    def to_deepagents_format(self) -> Dict[str, Any]:
        """Convert to deepagents-compatible format.

        Returns:
            Dictionary with crew configuration for deepagents
        """
        config_dict = {}

        if self.pattern:
            config_dict["pattern"] = self.pattern

        if self.backend:
            backend_config = self.backend.to_deepagents_format()
            if backend_config:
                config_dict["backend"] = backend_config

        if self.enable_lazy_loading is not None:
            config_dict["enable_lazy_loading"] = self.enable_lazy_loading

        if self.name:
            config_dict["name"] = self.name

        if self.tools:
            config_dict["tools"] = self.tools

        if self.mcps:
            config_dict["mcps"] = self.mcps

        return config_dict
