"""
Loader catalog — describes all supported LangChain document loaders.

Each entry is keyed by its ``source_type`` ID (used throughout the API).

Field schema:
  name         — Python key used in the config dict
  type         — text | url | secret | boolean | number | textarea | select
  label        — human-readable label shown in the UI
  required     — whether the field must be non-empty
  default      — default value (optional)
  placeholder  — input placeholder text (optional)
  hint         — short hint shown below the field (optional)
  help         — longer tooltip / help text (optional)
  env_var_hint — if set, UI shows "or set ENV_VAR" alternative (optional)
  options      — list of {value, label} for select-type fields (optional)

Secret fields (type="secret"):
  - UI renders a masked password input with show/hide toggle
  - Backend encrypts at rest and never returns the value in GET responses
  - If the value is "${VAR_NAME}", it is resolved from env at sync time

Adding a new loader = add one entry here. No UI code changes needed.
"""

from typing import Any, Dict, List

LOADER_CATALOG: Dict[str, Dict[str, Any]] = {

    # ── Collaboration ───────────────────────────────────────────────────────

    "confluence": {
        "id": "confluence",
        "display_name": "Confluence",
        "category": "collaboration",
        "description": "Index pages and spaces from Atlassian Confluence Cloud or Data Center.",
        "pip_extra": "atlassian-python-api",
        "langchain_class": "langchain_community.document_loaders.ConfluenceLoader",
        "fields": [
            {
                "name": "url",
                "type": "url",
                "label": "Confluence URL",
                "placeholder": "https://your-org.atlassian.net/wiki",
                "required": True,
            },
            {
                "name": "username",
                "type": "text",
                "label": "Username / Email",
                "required": True,
            },
            {
                "name": "api_key",
                "type": "secret",
                "label": "API Token",
                "hint": "Generate at id.atlassian.com/manage/api-tokens",
                "env_var_hint": "CONFLUENCE_API_KEY",
                "required": True,
            },
            {
                "name": "space_key",
                "type": "text",
                "label": "Space Key(s)",
                "placeholder": "ENG, HR",
                "help": "Comma-separated space keys to index. Leave blank to index all accessible spaces.",
                "required": False,
            },
            {
                "name": "include_attachments",
                "type": "boolean",
                "label": "Include Attachments",
                "default": False,
            },
            {
                "name": "limit",
                "type": "number",
                "label": "Max Pages per Space",
                "default": 50,
                "required": False,
            },
        ],
    },

    "sharepoint": {
        "id": "sharepoint",
        "display_name": "SharePoint",
        "category": "collaboration",
        "description": "Index files and documents from Microsoft SharePoint Online via Microsoft Graph API.",
        "pip_extra": "O365",
        "langchain_class": "langchain_community.document_loaders.SharePointLoader",
        "fields": [
            {
                "name": "client_id",
                "type": "text",
                "label": "Azure App Client ID",
                "help": "Found in Azure Portal → App Registrations → your app → Overview.",
                "required": True,
            },
            {
                "name": "client_secret",
                "type": "secret",
                "label": "Client Secret",
                "env_var_hint": "SHAREPOINT_CLIENT_SECRET",
                "required": True,
            },
            {
                "name": "tenant_id",
                "type": "text",
                "label": "Tenant ID",
                "help": "Found in Azure Portal → Azure Active Directory → Overview.",
                "required": True,
            },
            {
                "name": "site_name",
                "type": "text",
                "label": "Site Name",
                "placeholder": "my-intranet",
                "help": "The SharePoint site name (part of the site URL).",
                "required": True,
            },
            {
                "name": "document_library_path",
                "type": "text",
                "label": "Document Library Path",
                "placeholder": "Shared Documents/Policies",
                "help": "Relative path within the site. Leave blank for the default Documents library.",
                "required": False,
            },
        ],
    },

    # ── Cloud Storage ───────────────────────────────────────────────────────

    "s3_directory": {
        "id": "s3_directory",
        "display_name": "Amazon S3",
        "category": "cloud_storage",
        "description": "Index all documents in an S3 bucket or under a specific prefix.",
        "pip_extra": None,  # boto3 already a core dependency
        "langchain_class": "langchain_community.document_loaders.S3DirectoryLoader",
        "fields": [
            {
                "name": "bucket",
                "type": "text",
                "label": "Bucket Name",
                "required": True,
            },
            {
                "name": "prefix",
                "type": "text",
                "label": "Prefix / Folder",
                "placeholder": "docs/",
                "help": "Only objects under this prefix will be indexed. Leave blank for the whole bucket.",
                "required": False,
            },
            {
                "name": "region",
                "type": "text",
                "label": "AWS Region",
                "default": "us-east-1",
            },
            {
                "name": "aws_access_key_id",
                "type": "text",
                "label": "Access Key ID",
                "env_var_hint": "AWS_ACCESS_KEY_ID",
                "help": "Leave blank to use instance profile / environment credentials.",
                "required": False,
            },
            {
                "name": "aws_secret_access_key",
                "type": "secret",
                "label": "Secret Access Key",
                "env_var_hint": "AWS_SECRET_ACCESS_KEY",
                "required": False,
            },
        ],
    },

    # ── Web ─────────────────────────────────────────────────────────────────

    "web": {
        "id": "web",
        "display_name": "Web Pages / Sitemap",
        "category": "web",
        "description": "Crawl and index publicly accessible web pages or discover pages via an XML sitemap.",
        "pip_extra": None,
        "langchain_class": "langchain_community.document_loaders.WebBaseLoader",
        "fields": [
            {
                "name": "urls",
                "type": "textarea",
                "label": "URLs (one per line)",
                "placeholder": "https://docs.example.com/overview\nhttps://docs.example.com/api",
                "required": True,
            },
            {
                "name": "use_sitemap",
                "type": "boolean",
                "label": "Parse as XML Sitemap",
                "help": "If enabled, the first URL is treated as a sitemap.xml and all linked pages are crawled automatically.",
                "default": False,
            },
        ],
    },

    # ── Code / Docs-as-Code ─────────────────────────────────────────────────

    "github": {
        "id": "github",
        "display_name": "GitHub Repository",
        "category": "code",
        "description": "Index Markdown, text, and code files from a public or private GitHub repository.",
        "pip_extra": "PyGithub",
        "langchain_class": "langchain_community.document_loaders.GithubFileLoader",
        "fields": [
            {
                "name": "repo",
                "type": "text",
                "label": "Repository (owner/name)",
                "placeholder": "your-org/docs-repo",
                "required": True,
            },
            {
                "name": "branch",
                "type": "text",
                "label": "Branch",
                "default": "main",
            },
            {
                "name": "access_token",
                "type": "secret",
                "label": "GitHub Token",
                "hint": "Required for private repos. Generate at github.com/settings/tokens.",
                "env_var_hint": "GITHUB_TOKEN",
                "required": False,
            },
            {
                "name": "file_filter",
                "type": "text",
                "label": "File Pattern",
                "placeholder": "*.md",
                "help": "Glob pattern to filter files (e.g. *.md, docs/**/*.txt). Defaults to *.md.",
                "required": False,
            },
        ],
    },
}


def get_catalog_entry(source_type: str) -> Dict[str, Any]:
    """Return the catalog entry for a source type, or raise KeyError."""
    entry = LOADER_CATALOG.get(source_type)
    if not entry:
        raise KeyError(f"Unknown source type: {source_type!r}")
    return entry


def get_public_config(source_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Return config with secret fields masked — safe to return in API responses."""
    entry = LOADER_CATALOG.get(source_type, {})
    secret_fields = {
        f["name"]
        for f in entry.get("fields", [])
        if f.get("type") == "secret"
    }
    return {
        k: ("••••••••" if k in secret_fields and v else v)
        for k, v in config.items()
    }
