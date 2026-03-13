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

# Defaults
DEFAULT_AUTHOR = "Your Name"
DEFAULT_EMAIL_DOMAIN = "@capgemini.com"
DEFAULT_REGION = "us-east-1"
DEFAULT_MODEL_ID = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Options
FRAMEWORKS = ["langgraph", "crewai", "strands", "openai"]
VECTOR_STORES = ["chroma", "postgres", "s3"]

MODEL_OPTIONS = [
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0",
    "meta.llama3-70b-instruct-v1:0",
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
