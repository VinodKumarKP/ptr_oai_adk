class Constants:
    MCP = 'mcp'
    LANGCHAIN = 'langchain'
    BEDROCK = 'bedrock'
    MULTI_AGENT = 'multi-agent'
    CUSTOM = 'custom'
    REMOTE = 'remote'
    CREWAI = 'crewai'
    AWS_STRANDS = 'strands'
    LANGGRAPH = 'langgraph'
    OPENAI = 'openai'

    STREAMABLE_AGENT = [MCP, LANGCHAIN]
    NON_STREAMABLE_AGENT = [BEDROCK]
    AGENT_TYPES = [
        MCP,
        LANGCHAIN,
        BEDROCK,
        CUSTOM,
        REMOTE,
        MULTI_AGENT,
        CREWAI,
    ]

    AWS = 'aws'

    CLOUD_PROVIDER_LIST = [
        AWS
    ]

    # Pattern Constants
    PATTERN_SUPERVISOR = "supervisor"
    PATTERN_HANDOFF = "handoff"
    PATTERN_AGENT_AS_TOOL = "agent-as-tool"
    PATTERN_SWARM = "swarm"
    PATTERN_GRAPH = 'graph'
    PATTERN_SEQUENTIAL = 'sequential'
    PATTERN_HIERARCHICAL = 'hierarchical'
    PATTERN_DEEP = 'deep'

    # Agent Message Types
    AGENT_MESSAGE_TYPE_TEXT = "text"
    AGENT_MESSAGE_TYPE_STREAM_CHUNK = "stream_chunk"
    AGENT_MESSAGE_TYPE_STREAM_END = "stream_end"
    AGENT_MESSAGE_TYPE_TOOL_INPUT = "tool_input"
    AGENT_MESSAGE_TYPE_TOOL_OUTPUT = "tool_output"
    AGENT_MESSAGE_TYPE_ERROR = "error"
    AGENT_MESSAGE_TYPE_INFO = "info"
    AGENT_MESSAGE_TYPE_AGENT_ACTION = "agent_action"
    AGENT_MESSAGE_TYPE_AGENT_RESPONSE = "agent_response"

    # General Message Types
    MESSAGE_TYPE_TOOL = "tool"
    MESSAGE_TYPE_AI = "ai"
    MESSAGE_TYPE_HANDOFF = "handoff"
    MESSAGE_TYPE_SYSTEM = "system"
    MESSAGE_TYPE_USER = "user"


from enum import Enum


class ConfigKeys(Enum):
    """Top-level agent configuration keys.
    
    These are the main configuration sections supported in agent YAML files.
    Using an Enum prevents typos and enables IDE autocomplete.
    
    Example:
        >>> config_keys = ConfigKeys.KNOWLEDGE_BASE.value
        >>> # Access: config.get(ConfigKeys.KNOWLEDGE_BASE.value)
    """
    TYPE = "type"
    MODEL = "model"
    TOOLS = "tools"
    KNOWLEDGE_BASE = "knowledge_base"
    MEMORY = "memory"
    GUARDRAILS = "guardrails"
    SKILLS = "skills"
    OBSERVABILITY = "observability"
    EXTENSIONS = "extensions"


class KnowledgeBaseConfigKeys(Enum):
    """Knowledge base section configuration keys."""
    SOURCES = "sources"
    DATA_SOURCES = "data_sources"
    VECTOR_STORE_TYPE = "vector_store_type"
    REGISTRY = "registry"
    REGISTRY_URL = "registry_url"
    AUTH_TOKEN = "auth_token"
    EMBEDDING_MODEL = "embedding_model"
    COLLECTION_NAME = "collection_name"
    CHUNK_SIZE = "chunk_size"
    CHUNK_OVERLAP = "chunk_overlap"


class ToolConfigKeys(Enum):
    """Tools section configuration keys."""
    FRAMEWORK_TOOLS = "framework_tools"
    CUSTOM_TOOLS = "custom_tools"
    MCP_TOOLS = "mcp_tools"
    SHELL_ENABLED = "shell_enabled"


class MemoryConfigKeys(Enum):
    """Memory section configuration keys."""
    TYPE = "type"
    STORE_TYPE = "store_type"
    PERSIST = "persist"
    DB_PATH = "db_path"
    MAX_MESSAGES = "max_messages"


class GuardrailConfigKeys(Enum):
    """Guardrails section configuration keys."""
    ENABLED = "enabled"
    POLICIES = "policies"
    INPUT_RULES = "input_rules"
    OUTPUT_RULES = "output_rules"


class ModelConfigKeys(Enum):
    """Model configuration keys."""
    NAME = "name"
    PROVIDER = "provider"
    TEMPERATURE = "temperature"
    TOP_P = "top_p"
    MAX_TOKENS = "max_tokens"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"

