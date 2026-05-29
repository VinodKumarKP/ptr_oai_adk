# OAI Agent Core

A robust, modular, and extensible framework for building AI agents. This core library provides the foundational building blocks for creating, managing, and orchestrating intelligent agents across different frameworks (like LangChain, CrewAI, AWS Strands) and environments.

## 📋 Table of Contents

*   [Features](#-features)
*   [Installation](#-installation)
*   [Quick Start](#-quick-start)
*   [Core Components](#-core-components)
*   [Section Lookup](#-section-lookup)
*   [Usage Example](#-usage-example)
*   [BaseAgent Abstract Methods](#-baseagent-abstract-methods)
*   [Configuration & Validation](#-configuration--validation)
*   [Async Model & Thread Safety](#-async-model--thread-safety)
*   [Guardrails & Validation](#-guardrails--validation)
*   [Memory Management](#-memory-management)
*   [Error Handling](#-error-handling)
*   [Data Sources](#-data-sources)
*   [Contributing](#-contributing)
*   [License](#-license)

## 🌟 Features

*   **Unified Agent Interface**: A consistent `BaseAgent` abstraction that standardizes interaction regardless of the underlying agent framework.
*   **Modular Architecture**: Decoupled components for configuration, tool management, document loading, and observability.
*   **Configuration Management**: Flexible YAML-based configuration with comprehensive validation for agents, models, and tools.
*   **Tool Registry**: Centralized management for registering and loading tools, including support for Model Context Protocol (MCP). Module-based tool loading for reliability.
*   **Knowledge Base Integration**: Built-in support for vector stores and document loaders to ground agents in custom data.
*   **Conversation Memory**: Multi-turn conversation support with semantic search over memory history.
*   **Guardrails & Validation**: Optional input/output validation for security and data quality.
*   **Async-First Design**: Built for high-concurrency scenarios with thread-safe concurrent initialization and request handling.
*   **Graceful Degradation**: Non-critical components fail independently without preventing agent operation.
*   **Observability**: Integrated tracing and monitoring (e.g., via Langfuse) for debugging and performance analysis.
*   **Lazy Logging**: Efficient logging with lazy evaluation to minimize overhead.
*   **Extensibility**: Designed to be extended with custom agent implementations, tools, and vector stores.

## 📦 Installation

```bash
pip install oai-agent-core
```

To install with specific optional dependencies:

```bash
# For vector store support (required for any vector DB)
pip install "oai-agent-core[vector-required]"

# For ChromaDB support
pip install "oai-agent-core[chromadb]"

# For Postgres (pgvector) support
pip install "oai-agent-core[postgres]"

# For S3 vector store support
pip install "oai-agent-core[s3]"

# For all features
pip install "oai-agent-core[all]"
```

## ⚡ Quick Start

### 5-Minute Setup

1. **Create a configuration file** (`config/my_agent.yaml`):
```yaml
type: custom
name: my_agent
description: My custom agent

model:
  name: gpt-4
  temperature: 0.7
  max_tokens: 2000

tools:
  web_search:
    module: my_tools
    class: WebSearchTool
    params:
      api_key: "${SEARCH_API_KEY}"
```

2. **Implement your agent**:
```python
from oai_agent_core.core.base_agent import BaseAgent

class MyAgent(BaseAgent):
    async def initialize(self):
        await self._load_tools_and_kb_and_memory()
        self._initialized = True
    
    async def ainvoke(self, user_message: str, config=None):
        await self._ensure_initialized()
        return {"content": f"Response to: {user_message}"}
    
    async def astream(self, user_message: str, config=None):
        await self._ensure_initialized()
        yield {"content": "Streaming response"}
    
    def invoke(self, user_message: str, config=None):
        import asyncio
        return asyncio.run(self.ainvoke(user_message, config))
    
    async def stream(self, user_message: str, config=None):
        async for chunk in self.astream(user_message, config):
            yield chunk
```

3. **Use your agent**:
```python
import asyncio

async def main():
    agent = MyAgent(
        agent_name="my_agent",
        config_root="./config"
    )
    await agent.initialize()
    response = await agent.ainvoke("Hello, agent!")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

For more details, see [QUICK_START.md](docs/QUICK_START.md).

## 🏗️ Core Components

### 1. Core Abstractions (`oai_agent_core.core`)
*   **`BaseAgent`**: The abstract base class that all specific agent implementations (e.g., `LangGraphAgent`, `CrewAIAgent`) must inherit from. It handles initialization, configuration loading, and standardizes `process_request`, `invoke`, and `stream` methods.
*   **`BaseModelConfigurationManager`**: Manages LLM configurations, ensuring consistent model instantiation across different providers.
*   **`BaseToolRegistry`**: A registry for managing tools. It supports loading tools from configuration, dynamic imports, and MCP integration.
*   **`BaseKnowledgeBaseFactory`**: Abstract factory for creating knowledge base tools that agents can use to query vector stores.
*   **`BaseVectorStore`**: Abstract interface for vector database interactions.
*   **`BaseDocumentLoader`**: Abstract interface for loading and processing documents into vector stores.

### 2. Builders (`oai_agent_core.builders`)
*   **`AgentBuilder`**: Constructs and initializes agent instances from configuration.
*   **`ToolBuilder`**: Builds and registers tools from various sources (modules, functions, MCP).
*   **`KnowledgeBaseBuilder`**: Assembles knowledge bases, including vector stores and document loaders.
*   **`MemoryBuilder`**: Constructs and configures agent memory systems.
*   **`GuardrailsBuilder`**: Integrates and configures input/output validation using Guardrails.

### 3. Components (`oai_agent_core.components`)
*   **Configuration**: Utilities for loading and validating agent and model configurations (`ConfigManager`, `ModelConfig`).
*   **Observability**: Managers for integrating with observability platforms like Langfuse (`LangfuseObservabilityManager`).
*   **Loaders**: Implementations for document loading (e.g., `DocumentLoader`).
*   **Output Parser**: Manages structured output by converting Pydantic models to JSON schemas.

### 4. Processing (`oai_agent_core.processing`)
*   **`MessageFormatter`**: Handles prompt templating and variable substitution, preparing messages for agents.
*   **`OutputSerializer`**: Standardizes the output format of agent responses, ensuring consistency across different agent types.

### 5. Utilities (`oai_agent_core.utils`)
*   **`DynamicClassLoader`**: Helper for dynamically loading classes and modules at runtime.
*   **`Logger`**: Standardized logging configuration.

## 🔍 Section Lookup

Quickly find the component you need:

| Component | Module Path | Description |
| :--- | :--- | :--- |
| **Agent Base** | `oai_agent_core.core.base_agent` | Abstract base class for all agents. |
| **Model Config** | `oai_agent_core.core.base_model_configuration_manager` | Manages LLM configurations. |
| **Tool Registry** | `oai_agent_core.core.base_tool_registry` | Manages and loads tools (including MCP). |
| **Knowledge Base** | `oai_agent_core.core.base_knowledge_base_factory` | Factory for creating KB tools. |
| **Vector Store** | `oai_agent_core.core.base_vector_store` | Interface for vector DBs. |
| **Doc Loader** | `oai_agent_core.core.base_document_loader` | Interface for loading documents. |
| **Agent Builder** | `oai_agent_core.builders.agent_builder` | Constructs agent instances. |
| **Tool Builder** | `oai_agent_core.builders.tool_builder` | Builds and registers tools. |
| **KB Builder** | `oai_agent_core.builders.knowledge_base_builder` | Assembles knowledge bases. |
| **Memory Builder** | `oai_agent_core.builders.memory_builder` | Constructs agent memory. |
| **Guardrails Builder** | `oai_agent_core.builders.guardrails_builder` | Configures Guardrails validation. |
| **Config Manager** | `oai_agent_core.components.configuration.model_config` | Utilities for loading YAML configs. |
| **Observability** | `oai_agent_core.components.observability` | Integration with tracing tools (Langfuse). |
| **Output Parser** | `oai_agent_core.components.output_parser` | Manages structured output models. |
| **Formatter** | `oai_agent_core.processing.message_formatter` | Prompt templating and variable substitution. |
| **Serializer** | `oai_agent_core.processing.output_serializer` | Standardizes agent output format. |

## 🚀 Usage Example

Here's a conceptual example of how to use the core library to build a custom agent by implementing the `BaseAgent` abstract methods:

```python
from typing import Dict, Any, Optional
from oai_agent_core.core.base_agent import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, agent_name: str, agent_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(
            agent_name=agent_name,
            agent_config=agent_config,
            agent_type="custom",
            **kwargs
        )
        
    async def initialize(self):
        """Initialize resources (e.g., connect to DB, load models)."""
        print(f"Initializing {self.agent_name}...")
        # ... setup logic ...
        self._initialized = True
        
    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Asynchronous invocation."""
        await self._ensure_initialized()
        # ... processing logic ...
        return {"content": f"Processed async: {user_message}", "final": True}

    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Synchronous invocation."""
        import asyncio
        return asyncio.run(self.ainvoke(user_message, config))
        
    async def astream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Asynchronous streaming."""
        await self._ensure_initialized()
        # ... streaming logic ...
        yield {"content": "Chunk 1", "final": False}
        yield {"content": "Chunk 2", "final": True}

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Synchronous streaming (if supported)."""
        # Often just wraps astream or raises NotImplementedError
        async for chunk in self.astream(user_message, config):
            yield chunk

# Usage
config = {"model": {"model_id": "gpt-4"}}
agent = MyCustomAgent("my-agent", config)

# Async usage
import asyncio
async def main():
    await agent.initialize()
    response = await agent.ainvoke("Hello world")
    print(response)

if __name__ == "__main__":
    asyncio.run(main())
```

## 📋 BaseAgent Abstract Methods

Any class inheriting from `BaseAgent` must implement the following methods:

### Initialization

| Method | Description |
| :--- | :--- |
| `async initialize()` | **Must implement**. Initializes agent resources. Must set `self._initialized = True` when complete. Called automatically on first request if not called explicitly. |

### Execution Methods

| Method | Description | Recommended |
| :--- | :--- | :---: |
| `async ainvoke(user_message, config)` | Asynchronously processes a message and returns the full response. **Preferred for production**. | ✅ |
| `invoke(user_message, config)` | Synchronously processes a message. Can wrap `ainvoke()`. | Legacy |
| `async astream(user_message, config)` | Asynchronously streams response chunks. Best for real-time responses. | ✅ |
| `async stream(user_message, config)` | Synchronously streams chunks (or wraps `astream`). | Legacy |

### Implementation Pattern

```python
class MyAgent(BaseAgent):
    async def initialize(self):
        """Initialize resources - called automatically on first request."""
        await self._load_tools_and_kb_and_memory()
        self._initialized = True
    
    async def ainvoke(self, user_message: str, config=None):
        """Non-blocking request processing."""
        await self._ensure_initialized()
        # Your agent logic here
        return {"content": response, "final": True}
    
    async def astream(self, user_message: str, config=None):
        """Stream response chunks."""
        await self._ensure_initialized()
        # Your streaming logic
        yield {"content": chunk, "final": False}
    
    def invoke(self, user_message: str, config=None):
        """Sync wrapper (optional)."""
        import asyncio
        return asyncio.run(self.ainvoke(user_message, config))
    
    async def stream(self, user_message: str, config=None):
        """Sync stream wrapper (optional)."""
        async for chunk in self.astream(user_message, config):
            yield chunk
```

### Helper Methods

These are provided by `BaseAgent`:

```python
# Ensure agent is initialized before processing
await agent._ensure_initialized()

# Augment user message with context (KB, memory, etc.)
augmented = agent._augment_message(user_input)

# Apply input validation via guardrails
validated_input = agent._guardrail_input_message(user_input)

# Apply output validation via guardrails
validated_output = agent._guardrail_output_message(agent_response)

# Get conversation context from memory
context = agent._get_conversation_context(session_id, user_id)
```

## ⚙️ Configuration & Validation

### Configuration Loading

BaseAgent automatically loads configuration from YAML files:

```python
agent = MyAgent(
    agent_name="my_agent",           # Loads config/my_agent.yaml
    config_root="./config",          # Default is current directory
    agent_config=custom_config       # Or provide dict directly
)
```

### Configuration Validation

The framework includes comprehensive configuration validation:

```python
agent = MyAgent("my_agent", agent_config=config)
agent.validate_config()  # Validates all required fields

# Validates:
# ✓ Required 'type' field with valid values
# ✓ Model configuration (either 'llm' or 'model' config)
# ✓ Tool configuration format
# ✓ Knowledge base vector store requirements
# ✓ Memory configuration completeness
# ✓ Guardrails and output models
```

For complete configuration reference, see [CONFIGURATION.md](docs/CONFIGURATION.md).

## 🔄 Async Model & Thread Safety

### Concurrency Model

BaseAgent is designed for high-concurrency scenarios:

```python
import asyncio

agent = MyAgent("my_agent")
await agent.initialize()

# Safe concurrent requests via asyncio
results = await asyncio.gather(
    agent.ainvoke("Request 1"),
    agent.ainvoke("Request 2"),
    agent.ainvoke("Request 3")
)
```

### Key Guarantees

- **Initialization**: Protected by internal lock, safe for concurrent initialization
- **Requests**: Multiple concurrent `ainvoke()` calls are thread-safe
- **State**: Configuration mutations are serialized to prevent race conditions
- **Components**: Tool registry, KB, memory, and guardrails are safely shared

### Best Practices

1. **Prefer async**: Use `ainvoke()` and `astream()` over sync versions
2. **Reuse agents**: Create once, use for many requests
3. **Batch requests**: Use `asyncio.gather()` for concurrent operations
4. **Avoid config mutations**: Don't modify config after initialization

For detailed information, see [ASYNC_MODEL.md](docs/ASYNC_MODEL.md).

## 🛡️ Guardrails & Validation

### Input/Output Validation

BaseAgent supports optional guardrails for message validation:

```python
# If guardrails_manager is present
agent.guardrails_manager = guardrails_instance

# Input validation
cleaned_input = agent._guardrail_input_message(user_input)

# Output validation
validated_output = agent._guardrail_output_message(agent_response)

# Without guardrails, messages pass through unchanged
if not agent.guardrails_manager:
    # Messages return as-is
    pass
```

### Configuration

Enable guardrails via configuration:

```yaml
guardrails:
  enable_agent_validation: true
  rules:
    - type: input_filter
      action: sanitize
    - type: output_validator
      action: validate
```

## 💾 Memory Management

### Conversation Memory

BaseAgent includes built-in memory for multi-turn conversations:

```python
# Memory store manages conversation history
agent.memory_store = memory_instance

# Memory augments prompts with context
context = agent._get_conversation_context(session_id, user_id)

# Automatic context retrieval during augmentation
augmented_message = agent._augment_message(user_input)
```

### Configuration

```yaml
memory:
  vector_store:
    type: chroma
    settings:
      collection_name: conversation_history
  embedding:
    model_id: text-embedding-3-small
  settings:
    max_recent_turns: 10
```

## ⚠️ Error Handling

### Graceful Degradation

The framework gracefully handles component failures:

```python
# If Knowledge Base fails to initialize, agent still works
# If Memory Store initialization fails, agent continues
# If Guardrails setup fails, messages pass through unvalidated

# Check what initialized successfully
if agent.tool_registry:
    print("Tools loaded successfully")
if agent.global_kb_factory:
    print("Knowledge base ready")
if agent.memory_store:
    print("Memory system active")
if agent.guardrails_manager:
    print("Guardrails enabled")
```

### Configuration Validation Errors

```python
try:
    agent.validate_config()
except ValueError as e:
    # Comprehensive error message with all issues
    # Example: "Configuration validation failed for agent 'my_agent':
    #  - Missing required field 'type' in configuration
    #  - 'model' configuration missing required 'name' field"
    print(f"Config errors: {e}")
```

## 💾 Data Sources

The `BaseKnowledgeBaseFactory` and `BaseDocumentLoader` support loading data from various sources to ground your agents.

### Supported Sources

1.  **Local Files**: Load documents directly from the file system.
2.  **S3 Buckets**: Download and sync documents from AWS S3 buckets.

### Configuration Example

You can configure data sources in your agent's YAML configuration under the `knowledge_base` section:

```yaml
knowledge_base:
  - name: "my_knowledge_base"
    description: "Company documentation and manuals."
    vector_store:
      type: "chroma"
      settings:
        collection_name: "docs_collection"
        persist_directory: "./data/chroma_db"
    embedding:
      model_id: "amazon.titan-embed-text-v1"
    
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

### How S3 Loading Works
*   **Syncing**: The loader checks the S3 bucket for new or modified files (based on file size) compared to what has already been loaded.
*   **Downloading**: Only new/modified files are downloaded to a local cache directory.
*   **Metadata**: The `source` metadata field in the vector store is automatically updated to reflect the `s3://` URI (e.g., `s3://my-bucket/manuals/guide.pdf`) instead of the local temporary path.

## 📚 Documentation

Comprehensive guides for all aspects of the framework:

| Document | Purpose |
| :--- | :--- |
| [QUICK_START.md](docs/QUICK_START.md) | Get up and running in 5 minutes with code examples |
| [CONFIGURATION.md](docs/CONFIGURATION.md) | Complete configuration schema reference with examples |
| [ASYNC_MODEL.md](docs/ASYNC_MODEL.md) | Detailed concurrency, thread safety, and performance guide |
| [FIXES_SUMMARY.md](FIXES_SUMMARY.md) | Recent improvements and bug fixes |

## ✅ Recent Improvements

The codebase has been enhanced with critical fixes and improvements:

### Security & Stability
- ✅ **Deep Copy Configuration**: Fixed nested config mutations that could corrupt state
- ✅ **Thread Safety**: Added initialization locks for concurrent scenarios
- ✅ **Input Validation**: Comprehensive configuration validation with detailed error messages

### Quality & Performance
- ✅ **Lazy Logging**: Optimized logging with lazy formatting to reduce CPU/memory overhead
- ✅ **Code Cleanup**: Removed unnecessary classes and simplified architecture
- ✅ **Module-Based Tools**: Function-based tools deprecated in favor of reliable module-based loading

### Testing
- ✅ **30+ Integration Tests**: Comprehensive test coverage for concurrency, configuration, and guardrails
- ✅ **Error Handling**: Tested graceful degradation when components fail
- ✅ **State Consistency**: Verified no race conditions under concurrent initialization

## 🏆 Best Practices

### Performance
1. **Reuse Agents**: Create once, use for many requests
   ```python
   agent = MyAgent("name")
   await agent.initialize()
   for _ in range(1000):
       await agent.ainvoke("message")
   ```

2. **Batch Concurrent Requests**: Use `asyncio.gather()` for parallel processing
   ```python
   results = await asyncio.gather(
       agent.ainvoke("Q1"),
       agent.ainvoke("Q2"),
       agent.ainvoke("Q3")
   )
   ```

3. **Stream Long Responses**: Use `astream()` for real-time responses
   ```python
   async for chunk in agent.astream("long query"):
       process_chunk(chunk)
   ```

### Configuration
1. **Use Environment Variables** for secrets
   ```yaml
   tools:
     api_tool:
       params:
         api_key: "${API_KEY}"  # Loaded from environment
   ```

2. **Validate Before Deployment**
   ```python
   agent.validate_config()  # Catches issues early
   ```

3. **Organize Configurations** by environment
   ```
   config/
     production/
     staging/
     development/
   ```

### Development
1. **Enable Debug Logging** for troubleshooting
   ```python
   import logging
   logging.getLogger("oai_agent_core").setLevel(logging.DEBUG)
   ```

2. **Check Initialization Status**
   ```python
   print(f"KB: {agent.global_kb_factory is not None}")
   print(f"Memory: {agent.memory_store is not None}")
   print(f"Tools: {agent.tool_registry is not None}")
   ```

3. **Use Type Hints** in custom implementations
   ```python
   async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
   ```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.