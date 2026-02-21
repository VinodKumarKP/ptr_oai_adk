# OpenAI Multi-Agent Framework

A powerful, YAML-based configuration system for building multi-agent AI workflows with OpenAI models. Build complex agent orchestrations without writing extensive boilerplate code—just configure and run.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Key Features](#key-features)
- [Configuration Structure](#configuration-structure)
- [Orchestration Patterns](#orchestration-patterns)
- [Agents Configuration](#agents-configuration)
- [Tools System](#tools-system)
- [Knowledge Base Integration](#knowledge-base-integration)
- [Data Sources](#data-sources)
- [Memory Management](#memory-management)
- [MCP Integration](#mcp-integration)
- [Dynamic Input Variables](#dynamic-input-variables)
- [Usage Examples](#usage-examples)
- [Streaming Support](#streaming-support)
- [Observability](#observability)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The OpenAI Multi-Agent Framework enables you to create sophisticated agent orchestrations using OpenAI's models through simple YAML configuration files. It provides a declarative way to define multi-agent systems with support for various orchestration patterns, tool integration, and memory management.

### What Can You Build?

- Research and analysis pipelines with agent handoffs
- Complex decision-making systems with multiple specialists
- Data processing workflows with parallel execution
- Autonomous agent systems with dynamic collaboration
- Enterprise-grade AI applications

## Quick Start

### 1. Installation

```bash
pip install openai-agents
```

To install with specific optional dependencies:

```bash
# For vector store support (required for any vector DB)
pip install "openai-agents[vector-required]"

# For ChromaDB support
pip install "openai-agents[chromadb]"

# For Postgres (pgvector) support
pip install "openai-agents[postgres]"

# For S3 vector store support
pip install "openai-agents[s3]"

# For all features
pip install "openai-agents[all]"
```

### 2. Create Your Configuration File

Create a YAML file (e.g., `research_agent.yaml`):

```yaml
model:
  model_id: gpt-4o
  cloud_provider: openai

tools:
  calculator:
    module: strands_tools

agent_list:
  - researcher:
      system_prompt: You research topics and gather information. Hand off to analyst when you have findings.
  - analyst:
      system_prompt: You analyze information and identify key insights. Hand off to writer for final output.
      context:
        - researcher
  - writer:
      system_prompt: You write clear, engaging summaries of analyzed information.
      context:
        - researcher
        - analyst

crew_config:
  pattern: supervisor  # Agents are coordinated by a supervisor
  verbose: true
```

### 3. Initialize and Run

```python
from oai_openai_agent_core.agents.openai_agent import OpenAIAgent
import yaml
import asyncio

async def main():
    # Load configuration
    with open("research_agent.yaml", "r") as f:
        config = yaml.safe_load(f)

    agent = OpenAIAgent(
        agent_name="research_crew",
        agent_config=config
    )

    # Initialize
    await agent.initialize()

    # Execute
    result = await agent.ainvoke("Research the latest trends in quantum computing")
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Key Features

### 🤖 OpenAI Model Integration
Built on OpenAI's powerful models (GPT-4o, GPT-4-turbo, etc.) via the `agents` library.

### 🔄 Multiple Orchestration Patterns
Support for Supervisor, Handoff, and Single Agent patterns.

### 🎯 Autonomous Agent Handoffs
Agents can intelligently hand off tasks to each other based on expertise.

### 🛠️ Extensible Tools System
Integrate standard Python functions, custom tools, and MCP servers seamlessly.

### 📚 Knowledge Base Support
Easily integrate custom knowledge bases (RAG) for agents to access domain-specific information.

### 🧠 Long-Term Memory
Persistent memory store for maintaining context across sessions with semantic search capabilities.

### 🔌 MCP Server Support
Connect to Model Context Protocol servers for enhanced capabilities (filesystem, databases, etc.).

### 📊 Session Management
Built-in session tracking and output serialization for conversation continuity.

### 📈 Observability with Langfuse
Optional Langfuse integration for tracing, monitoring, and debugging.

### ⚡ Streaming Support
Real-time streaming of agent outputs and task handoffs.

## Configuration Structure

### Complete YAML Template

```yaml
# Model Configuration
model:
  model_id: gpt-4o
  cloud_provider: openai
  params:  # Optional
    temperature: 0.7
    max_tokens: 4096

# Tools Definition
tools:
  tool_name:
    module: "module_name"
    class: "ToolClassName"  # Optional
    function_list:  # Optional: load specific functions
      - function_name
    base_path: "./path"  # Optional

# Knowledge Base Definition (Optional)
knowledge_base:
  - name: company_policies
    description:  "Search company policies, HR guidelines, and internal procedures"
    vector_store:
      type: chroma
      settings:
        collection_name: "company policies"
        persist_directory: "./rag_db"
    embedding:
      model_id: "bedrock/amazon.titan-embed-text-v1"
      region_name: "us-west-2"
    data_sources:
      - path: "docs/sample_policy.pdf"
    text_splitter:
      type: "recursive_character"
      chunk_size: 1000
      chunk_overlap: 200
    retrieval_settings:
        top_k: 5
        score_threshold: 0.7

# Memory Configuration (Optional)
memory:
  vector_store:
    type: chroma # Options: chroma, postgres, s3
    settings:
      collection_name: "chat_memory"
      persist_directory: "./memory_db"
      # For Postgres:
      # db_host: "localhost"
      # db_port: "5432"
      # db_user: "user"
      # db_name: "mydb"
      # For S3:
      # bucket_name: "my-bucket"
  embedding:
    model_id: "bedrock/amazon.titan-embed-text-v1"
    region_name: "us-west-2"
  settings:
    max_recent_turns: 5
    max_relevant_turns: 3
    similarity_threshold: 0.6

# MCP Servers (Optional)
mcps:
  server_name:
    command: "server_command"
    args: ["arg1", "arg2"]

# Agent Definitions
agent_list:
  - agent_key:
      system_prompt: "Detailed instructions for the agent"
      context:  # Optional: agents this agent can see
        - other_agent_key
      tools:  # Optional: tools available to this agent
        - tool_name
      knowledge_base: # Optional: assign specific KB to agent
        - name: "company_policies"
          description: "Search company policies"
          vector_store: ...

# Orchestration Configuration
crew_config:
  pattern: "supervisor"  # supervisor, handoff, agent-as-tool
  verbose: true
```

## Orchestration Patterns

The framework supports multiple orchestration patterns for different use cases:

### 1. Multi-Agent Supervisor

**When to use:** A central supervisor agent delegates tasks to specialized sub-agents. The supervisor maintains the state and decides the next step.

```yaml
crew_config:
  pattern: supervisor
  verbose: true

system_prompt: You are a supervisor. Delegate tasks to the appropriate worker.

agent_list:
  - researcher:
      system_prompt: You research topics.
  - writer:
      system_prompt: You write summaries based on research.
```

**Flow:**
```
User Input → Supervisor → (Delegates) → Worker Agent → (Returns) → Supervisor → Final Output
```

### 2. Handoff Pattern

**When to use:** Agents autonomously hand off tasks to one another. There is no central supervisor.

```yaml
crew_config:
  pattern: handoff
  verbose: true

agent_list:
  - researcher:
      system_prompt: You research topics. Hand off to writer when done.
  - writer:
      system_prompt: You write summaries.
```

**Flow:**
```
User Input → Researcher → Writer → Final Output
```

### 3. Agent as Tool

**When to use:** When you want a main agent to treat other specialized agents as tools. The main agent calls the sub-agent, waits for the result, and then continues.

```yaml
crew_config:
  pattern: agent-as-tool

system_prompt: You are a helpful assistant. Use the specialized agents as tools to answer questions.

agent_list:
  - math_expert:
      system_prompt: You solve math problems.
      description: "Agent useful for solving math problems"
  - physics_expert:
      system_prompt: You explain physics concepts.
      description: "Agent useful for physics questions"
```

**Flow:**
```
User Input → Main Agent → (Calls Tool) → Sub-Agent → (Returns Result) → Main Agent → Final Output
```

### 4. Single Agent

**When to use:** Simple workflows handled by one agent.

```yaml
# No crew_config needed or pattern: single
agent_list:
  - assistant:
      system_prompt: You are a helpful assistant.
```

## Agents Configuration

### Agent Properties

```yaml
agent_list:
  - agent_key:
      system_prompt: "Detailed instructions"  # Required
      tools: [tool_name]  # Optional: tools available to agent
      mcps: # Optional: Agent specific MCP servers
         server_name: ...
```

### System Prompt Best Practices

```yaml
# ✅ Good - Clear role and instructions
system_prompt: |
  You are a research analyst who gathers information from reliable sources.
  
  Your responsibilities:
  - Search for relevant information on the given topic
  - Verify source credibility
  - Summarize key findings

# ❌ Bad - Vague instructions
system_prompt: You help with research.
```

## Tools System

### Defining Tools

#### Load All Functions from Module

```yaml
tools:
  random_generator:
    module: random_generator
    base_path: ./../utils
```

#### Load Specific Functions

```yaml
tools:
  random_generator:
    module: random_generator
    function_list:
      - generate_random_number
      - generate_random_string
    base_path: ./../utils
```

### Setting Default Parameter Values

You can configure default parameter values for tool functions using the `function_params` field.

```yaml
tools:
  random_generator:
    module: random_generator
    base_path: ./../utils
    function_params:
      generate_random_number:
        lower: 10
        upper: 100
```

### Custom Tools

Create a Python file with your tool:

```python
# my_tools/data_processor.py
def process_data(data: str, max_records: int = 100) -> str:
    """Process and clean data.
    
    Args:
        data: Raw data to process
        max_records: Maximum number of records to process (default: 100)
    """
    # Your implementation
    return processed_data
```

Configure in YAML:

```yaml
tools:
  data_tools:
    module: data_processor
    function_list:
      - process_data
    base_path: ./my_tools
```

## Knowledge Base Integration

You can equip your agents with a custom knowledge base (RAG) to answer questions based on specific documents.

### Common Configuration

All knowledge base configurations share these common settings, which are optional and configurable:

```yaml
knowledge_base:
  - name: my_kb
    description: "Description of the knowledge base"
    # ... vector_store config ...
    
    # Optional: Configure embedding model
    embedding:
      model_id: "text-embedding-3-small"
      region_name: "us-west-2" # Optional
      
    # Optional: List of documents to index
    data_sources:
      - path: "docs/document.pdf"
      
    # Optional: Configure text splitting strategy
    text_splitter:
      type: "recursive_character"
      chunk_size: 1000
      chunk_overlap: 200
      
    # Optional: Configure retrieval parameters
    retrieval_settings:
      top_k: 5
      score_threshold: 0.7
```

### Vector Store Configuration

The framework supports multiple vector store backends. Choose the one that fits your needs.

#### 1. ChromaDB (Default)
Good for local development and simple deployments.

```yaml
knowledge_base:
  - name: local_kb
    description: "Local document search"
    vector_store:
      type: chroma
      settings:
        collection_name: "my_collection"
        persist_directory: "./data/chroma"
    # ... common settings ...
```

#### 2. Postgres (pgvector)
Ideal for production environments using PostgreSQL.

```yaml
knowledge_base:
  - name: postgres_kb
    description: "Enterprise document search"
    vector_store:
      type: postgres
      settings:
        collection_name: "enterprise_docs"
        # Option 1: Connection String (Not recommended for passwords)
        # connection_string: "postgresql://user:pass@localhost:5432/mydb"
        
        # Option 2: Individual Parameters (Recommended)
        db_host: "localhost"
        db_port: "5432"
        db_user: "user"
        db_name: "mydb"
        # Note: db_password should NOT be in YAML.
        # Use environment variables with DB_NAME suffix:
        # DB_HOST_MYDB, DB_PORT_MYDB, DB_USER_MYDB, DB_PASSWORD_MYDB
    # ... common settings ...
```

#### 3. S3 (Simple Vector Store)
Lightweight, serverless option storing index in S3. Good for read-heavy, low-write scenarios.

```yaml
knowledge_base:
  - name: s3_kb
    description: "Cloud document search"
    vector_store:
      type: s3
      settings:
        collection_name: "cloud_docs"
        bucket_name: "my-vector-store-bucket"
        prefix: "indexes" # Optional
    # ... common settings ...
```

### Global Knowledge Base

A global knowledge base is automatically queried and the relevant context is appended to the user's message.

```yaml
knowledge_base:
  - name: company_policies
    description:  "Search company policies"
    vector_store:
      type: chroma
      settings:
        collection_name: "policies"
        persist_directory: "./rag_db"
    embedding:
      model_id: "text-embedding-3-small"
    data_sources:
      - path: "docs/hr_policy.pdf"
    text_splitter:
      type: "recursive_character"
      chunk_size: 1000
      chunk_overlap: 200
    retrieval_settings:
        top_k: 5
        score_threshold: 0.7
```

### Agent-Specific Knowledge Base

You can also assign a knowledge base as a tool to a specific agent.

```yaml
agent_list:
  - policy_expert:
      system_prompt: You answer questions about company policies. Use the search_knowledge_base tool.
      knowledge_base:
        - name: "company_policies"
          description: "Search company policies"
          vector_store:
            type: chroma
            settings:
              collection_name: "policies"
              persist_directory: "./rag_db"
          embedding:
            model_id: "text-embedding-3-small"
          data_sources:
            - path: "docs/policy.pdf"
          text_splitter:
            type: "recursive_character"
            chunk_size: 1000
            chunk_overlap: 200
          retrieval_settings:
              top_k: 5
              score_threshold: 0.7
```

## Data Sources

The framework supports loading data from various sources to ground your agents.

### Supported Sources

1.  **Local Files**: Load documents directly from the file system.
2.  **S3 Buckets**: Download and sync documents from AWS S3 buckets.

### Configuration Example

```yaml
knowledge_base:
  - name: "my_knowledge_base"
    data_sources:
      # 1. Local File Source
      - type: "file"
        path: "/path/to/local/documents/*.pdf"
        chunk_size: 1000
        chunk_overlap: 200

      # 2. S3 Bucket Source
      - type: "s3"
        bucket: "my-company-docs-bucket"
        prefix: "manuals/"  # Optional: specific folder
        # Files are downloaded to {persist_directory}/s3_bucket/{bucket_name}/...
```

## Memory Management

The framework supports persistent memory to maintain context across sessions. This allows agents to recall previous interactions and provide more personalized responses.

### Memory Configuration

Add a `memory` section to your YAML configuration. The vector store is configurable (Chroma, Postgres, S3). Please refer to the respective vector store configuration sections above for details on settings.

```yaml
memory:
  vector_store:
    type: chroma # Options: chroma, postgres, s3
    settings:
      collection_name: "chat_memory"
      persist_directory: "./memory_db"
      # For Postgres:
      # db_host: "localhost"
      # db_port: "5432"
      # db_user: "user"
      # db_name: "mydb"
      # For S3:
      # bucket_name: "my-bucket"
  embedding:
    model_id: "bedrock/amazon.titan-embed-text-v1"
    region_name: "us-west-2"
  settings:
    max_recent_turns: 5       # Number of recent turns to include
    max_relevant_turns: 3     # Number of semantically relevant past turns to include
    similarity_threshold: 0.6 # Threshold for semantic relevance (0.0 to 1.0)
```

### How It Works

1. **Storage**: Every interaction (user query + agent response) is stored in a vector database.
2. **Retrieval**: Before processing a new query, the system retrieves:
   - The most recent conversation turns (short-term memory)
   - Semantically relevant past interactions (long-term memory)
3. **Augmentation**: This context is automatically appended to the user's prompt, allowing the agent to "remember" past details.

## MCP Integration

Model Context Protocol (MCP) servers provide enhanced capabilities like filesystem access, database queries, and more.

### MCP Configuration

```yaml
mcps:
  filesystem:
    command: "mcp-server-filesystem"
    args: ["/data"]
  
  environment:
    command: "uv"
    args:
      - "run"
      - "--with"
      - "mcp-env-server"
      - "env_lookup_server"
```

### Using MCP Servers

Agents automatically get access to MCP tools defined in the configuration.

## Dynamic Input Variables

### Variable Syntax

Use `{variable_name}` in system prompts:

```yaml
agent_list:
  - researcher:
      system_prompt: |
        Research the topic: {topic}
        Focus on {aspect} in the {industry} industry.
```

### Providing Inputs

```python
result = await agent.ainvoke(
    "Research quantum computing",
    config={
        'inputs': {
            'topic': 'Quantum Computing',
            'aspect': 'commercial applications',
            'industry': 'finance'
        }
    }
)
```

## Usage Examples

### Example 1: Simple Assistant

```yaml
model:
  model_id: gpt-4o
  cloud_provider: openai

agent_list:
  - assistant:
      system_prompt: You are a helpful assistant.
```

### Example 2: Multi-Agent Research Team

```yaml
model:
  model_id: gpt-4o
  cloud_provider: openai

agent_list:
  - researcher:
      system_prompt: Research topics and hand off to writer.
  - writer:
      system_prompt: Write summaries based on research.

crew_config:
  pattern: handoff
```

## Streaming Support

### Async Streaming

```python
async for chunk in agent.astream("Research quantum computing"):
    if chunk.get('type') == 'text':
        print(chunk['content'], end='', flush=True)
    elif chunk.get('type') == 'tool_call_item':
        print(f"\n[Tool Call: {chunk['content']}]")
```

## Observability

### Langfuse Integration

Enable tracing by setting environment variables:

```bash
export LANGFUSE_ENABLED=true
export LANGFUSE_PUBLIC_KEY=pk-xxx
export LANGFUSE_SECRET_KEY=sk-xxx
export LANGFUSE_HOST=https://cloud.langfuse.com
```

## Best Practices

1. **Clear Agent Roles**: Define specific responsibilities for each agent.
2. **Security**: Use environment variables for API keys and credentials.
3. **Tool Assignment**: Only assign necessary tools to each agent to reduce confusion and token usage.
4. **Model Selection**: Choose the appropriate model (e.g., GPT-4o for complex tasks, GPT-3.5-turbo for simple ones) to balance performance and cost.

## Troubleshooting

### Common Issues

**Issue: "Agent not initialized"**
```python
# Solution: Always call initialize() before use
await agent.initialize()
```

**Issue: "Tool not found"**
```yaml
# Solution: Ensure tool is defined in the 'tools' section and the module path is correct.
```

**Issue: "OpenAI API Key missing"**
```bash
# Solution: Set OPENAI_API_KEY environment variable
export OPENAI_API_KEY=sk-...
```

## API Reference

### OpenAIAgent Class

```python
class OpenAIAgent:
    def __init__(
        agent_name: str,
        agent_config: Dict[str, Any],
        session_id: str = "default",
        user_id: str = "default",
        **kwargs
    )
    
    async def initialize() -> None
    async def ainvoke(message: str, config: Dict = None) -> Dict
    def invoke(message: str, config: Dict = None) -> Dict
    async def astream(message: str, config: Dict = None) -> AsyncGenerator
```

---

**Need help?** Open an issue or check the documentation.
