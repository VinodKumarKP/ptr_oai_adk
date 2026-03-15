"""Constants and configuration defaults."""

from pathlib import Path

# Paths
TEMPLATES_DIR = Path(__file__).parent / "templates"

# Placeholders
TOKEN_PROJECT_NAME = "{{PROJECT_NAME}}"
TOKEN_AUTHOR = "{{AUTHOR}}"
TOKEN_EMAIL = "{{EMAIL}}"
TOKEN_DESCRIPTION = "{{DESCRIPTION}}"
TOKEN_TITLE = "{{PROJECT_TITLE}}"
TOKEN_AGENT_CORE_PYPROJECT = "{{AGENT_CORE_PYPROJECT_PLACEHOLDER}}"
TOKEN_AGENT_CORE_REQUIREMENTS = "{{AGENT_CORE_REQUIREMENTS_PLACEHOLDER}}"


# Defaults
DEFAULT_AUTHOR = "Your Name"
DEFAULT_EMAIL_DOMAIN = "@capgemini.com"
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Options
FRAMEWORKS = ["langgraph", "crewai", "strands", "openai"]
VECTOR_STORES = ["chroma", "postgres", "s3"]

MODEL_OPTIONS = [
    "bedrock/global.amazon.nova-2-lite-v1:0",
    "bedrock/global.anthropic.claude-haiku-4-5-20251001-v1:0",
    "bedrock/global.anthropic.claude-opus-4-5-20251101-v1:0",
    "bedrock/global.anthropic.claude-opus-4-6-v1",
    "bedrock/global.anthropic.claude-sonnet-4-6",
    "bedrock/global.anthropic.claude-sonnet-4-20250514-v1:0",
    "bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "Custom..."
]

FRAMEWORK_INFO = {
    "langgraph": {"core": "langgraph_core", "pkg": "langgraph_agent", "cls": "LangGraphAgent"},
    "crewai": {"core": "crewai_core", "pkg": "crewai_agent", "cls": "CrewAIAgent"},
    "strands": {"core": "aws_strands_core", "pkg": "aws_strands_agent", "cls": "StrandsAgent"},
    "openai": {"core": "openai_core", "pkg": "openai_agent", "cls": "OpenAIAgent"}
}

# Framework specific patterns
FRAMEWORK_PATTERNS = {
    "langgraph": ["supervisor", "agent-as-tool", "swarm"],
    "openai": ["supervisor", "agent-as-tool", "swarm", "handoff"],
    "strands": ["graph", "swarm", "sequential", "hierarchical", "agent-as-tool"],
    "crewai": ["crew", "flow"]
}

# Agent Core Dependency Mapping
AGENT_CORE_DEPS = {
    "langgraph": "oai-langgraph-agent-core",
    "crewai": "oai-crewai-agent-core",
    "strands": "oai-aws-strands-core",
    "openai": "oai-openai-agent-core"
}

AGENT_CORE_SUBDIRS = {
    "langgraph": "packages/langgraph-agent-core",
    "crewai": "packages/crewai-agent-core",
    "strands": "packages/aws-strands-core",
    "openai": "packages/openai-agent-core"
}

VECTOR_STORE_EXTRAS = {
    "chroma": "chromadb",
    "postgres": "postgres",
    "s3": "s3"
}
