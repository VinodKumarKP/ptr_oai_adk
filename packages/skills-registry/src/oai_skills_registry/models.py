from typing import Dict, Optional, Literal, List, Any
from datetime import datetime
from pydantic import BaseModel, Field


class SkillGitSource(BaseModel):
    """Represents a Git repository as skill source."""
    name: str
    git_provider: Literal["github", "gitlab", "gitea"] = "github"
    repository: str  # owner/repo
    git_url: str
    branch: str = "main"
    auth_type: Literal["token", "ssh"] = "token"
    # auth_token is not exposed in API responses (stored encrypted in DB)


class SkillVersion(BaseModel):
    """Represents a specific version of a skill."""
    id: Optional[int] = None
    skill_name: Optional[str] = None  # Optional - may not be in all queries
    version: str  # semantic version: 1.0.0
    git_source_id: Optional[int] = None
    git_branch: Optional[str] = None
    git_commit_sha: Optional[str] = None
    git_tag: Optional[str] = None
    content: Optional[str] = None  # Full SKILL.md
    config: Optional[Dict] = None  # skill_config.yaml - can be dict or JSON string
    dependencies: Optional[List[str]] = None  # List of dependency strings
    breaking_changes: Optional[List[str]] = None
    status: Literal["draft", "published", "deprecated", "archived"] = "draft"
    published_by: Optional[str] = None
    published_at: Optional[datetime] = None
    deprecated_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @classmethod
    def from_db(cls, data: Dict[str, Any]) -> "SkillVersion":
        """Create instance from database row, parsing JSON strings as needed."""
        import json

        # Parse config if it's a JSON string
        if isinstance(data.get('config'), str):
            try:
                data['config'] = json.loads(data['config'])
            except (json.JSONDecodeError, TypeError):
                data['config'] = None

        # Parse dependencies if it's a JSON string
        if isinstance(data.get('dependencies'), str):
            try:
                data['dependencies'] = json.loads(data['dependencies'])
            except (json.JSONDecodeError, TypeError):
                data['dependencies'] = None

        return cls(**data)


class SkillAction(BaseModel):
    """Record of a skill lifecycle action."""
    id: int
    skill_name: str
    action: Literal["publish", "upgrade", "downgrade", "deprecate", "delete"]
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    performed_by: str
    message: Optional[str] = None
    created_at: str  # ISO 8601 timestamp


class SkillActionHistory(BaseModel):
    """Complete action history for a skill."""
    skill_name: str
    total_count: int
    actions: List[SkillAction]


class SkillMetadata(BaseModel):
    """Core skill metadata."""
    id: Optional[int] = None
    name: str
    description: str
    category: str
    tags: List[str] = Field(default_factory=list)
    current_version: Optional[str] = None
    status: Literal["active", "deprecated", "archived"] = "active"
    author: str
    git_repository_url: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


class SkillDetails(BaseModel):
    """Full skill details with all versions."""
    skill: SkillMetadata
    versions: List[SkillVersion]
    version_count: int


class SkillRegistration(BaseModel):
    """Payload for registering a new skill (initial creation)."""
    name: str
    description: str
    category: str
    tags: Optional[List[str]] = Field(default_factory=list)
    author: str
    git_repository_url: Optional[str] = None
    initial_version: Optional[str] = None


class SkillImportConfig(BaseModel):
    """Config for importing skill from Git."""
    git_source_id: Optional[int] = None  # For registered git sources
    git_repository_url: Optional[str] = None  # For direct GitHub imports
    git_tag: Optional[str] = None  # If None, use latest commit on branch
    version: Optional[str] = None  # If None, extract from SKILL.md

    class Config:
        @classmethod
        def validate_imports(cls, values):
            """At least one of git_source_id or git_repository_url must be provided."""
            if not values.get('git_source_id') and not values.get('git_repository_url'):
                raise ValueError('Either git_source_id or git_repository_url must be provided')
            return values


class SkillPublishConfig(BaseModel):
    """Config for publishing a skill version."""
    version: str
    message: Optional[str] = None


class SkillUpgradeConfig(BaseModel):
    """Config for upgrading to new skill version."""
    to_version: str
    message: Optional[str] = None


class SkillDowngradeConfig(BaseModel):
    """Config for downgrading to previous skill version."""
    to_version: str
    message: Optional[str] = None


class SkillDeprecateConfig(BaseModel):
    """Config for deprecating a skill version."""
    version: str
    message: Optional[str] = None


class GitVersionInfo(BaseModel):
    """Information about a version available in a GitHub repository."""
    version: Optional[str] = None           # semantic version from SKILL.md
    git_tag: str                             # git tag name (e.g. v1.2.0)
    commit_sha: Optional[str] = None        # 40-char commit SHA
    message: Optional[str] = None           # tag / commit message
    created_at: Optional[str] = None        # ISO-8601 timestamp from GitHub
    author: Optional[str] = None            # tagger or SKILL.md author field
    description: Optional[str] = None       # description from SKILL.md
    category: Optional[str] = None          # category from SKILL.md
    tags: Optional[List[str]] = None        # tags array from SKILL.md


class SkillVersionRefresh(BaseModel):
    """Response from the refresh-versions endpoint."""
    status: str
    skill: str
    versions: List[GitVersionInfo]
    versions_found: int


class SkillImportPreview(BaseModel):
    """Preview of skill being imported from Git."""
    skill_name: str
    version: str
    description: str
    category: str
    tags: List[str]
    changes: Optional[List[str]] = None
    git_commit: str
    git_tag: Optional[str] = None
    breaking_changes: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None  # Can be list of requirements or dict


class SkillSyncHistory(BaseModel):
    """History of manual syncs/imports."""
    id: int
    skill_name: str
    action: str  # "import", "publish", "upgrade"
    version: str
    git_source_id: int
    git_tag: str
    git_commit_sha: str
    synced_at: datetime
    synced_by: str
    status: Literal["success", "failed"]
    error_message: Optional[str] = None


class AgentSkillMapping(BaseModel):
    """Agent's installed skill with version constraint."""
    agent_id: str
    skill_name: str
    version_constraint: str  # "latest", "1.0.0", ">=2.0.0,<3.0.0"
    current_resolved_version: Optional[str] = None
    installed_at: datetime
    auto_upgrade: bool = False


class DiscoveredSkill(BaseModel):
    """A skill discovered in a Git repository."""
    name: str
    version: str
    description: str
    category: str
    author: str
    tags: List[str] = Field(default_factory=list)
    status: str  # "available", "already_registered", "invalid"
    path_in_repo: str  # e.g., "skills/web-scraper"


class SkillDiscoveryResult(BaseModel):
    """Result of discovering skills in a Git repository."""
    git_repository_url: str
    total_found: int
    available_to_register: int
    already_registered: int
    invalid: int
    skills: List[DiscoveredSkill]


class BulkRegistrationRequest(BaseModel):
    """Request to register multiple skills."""
    git_repository_url: str
    skill_names: List[str]  # Which discovered skills to register
    author: Optional[str] = None  # Who's registering (defaults to authenticated user)


class BulkRegistrationResult(BaseModel):
    """Result of bulk registration."""
    total_registered: int
    successful: List[str]  # Skill names that were registered
    failed: List[Dict[str, str]]  # {skill_name: error_message}


class SkillPublishedVersion(BaseModel):
    """Published skill version with repository info for agents."""
    # Skill metadata
    skill_name: str
    description: str
    category: str
    author: str
    git_repository_url: Optional[str] = None  # Where agents can clone from

    # Published version info
    version: str  # Current published version
    git_tag: Optional[str] = None  # Git tag to checkout
    git_commit_sha: Optional[str] = None  # Specific commit
    config: Optional[Dict] = None  # Input/output schemas
    dependencies: Optional[List[str]] = None  # pip packages
    capabilities: Optional[List[str]] = None  # What this skill can do
    breaking_changes: Optional[List[str]] = None

    # Metadata
    published_at: Optional[datetime] = None
    published_by: Optional[str] = None
