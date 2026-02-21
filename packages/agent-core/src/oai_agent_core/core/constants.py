class Constants:
    MCP = 'mcp'
    LANGCHAIN = 'langchain'
    BEDROCK = 'bedrock'
    MULTI_AGENT = 'multi-agent'
    CUSTOM = 'custom'
    REMOTE = 'remote'
    CREWAI = 'crewai'
    AWS_STRANDS = 'aws-strands'
    LANGGRAPH = 'langgraph'

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
