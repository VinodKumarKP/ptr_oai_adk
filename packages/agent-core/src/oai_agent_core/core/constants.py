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
