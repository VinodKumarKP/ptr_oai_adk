# OAI Agent Core

A robust, modular, and extensible framework for building AI agents. This core library provides the foundational building blocks for creating, managing, and orchestrating intelligent agents across different frameworks (like LangChain, CrewAI, AWS Strands) and environments.

## 📋 Table of Contents

*   [Features](#-features)
*   [Installation](#-installation)
*   [Core Components](#-core-components)
*   [Section Lookup](#-section-lookup)
*   [Usage Example](#-usage-example)
*   [BaseAgent Abstract Methods](#-baseagent-abstract-methods)
*   [Data Sources](#-data-sources)
*   [Contributing](#-contributing)
*   [License](#-license)

## 🌟 Features

*   **Unified Agent Interface**: A consistent `BaseAgent` abstraction that standardizes interaction regardless of the underlying agent framework.
*   **Modular Architecture**: Decoupled components for configuration, tool management, document loading, and observability.
*   **Configuration Management**: Flexible YAML-based configuration for agents, models, and tools.
*   **Tool Registry**: Centralized management for registering and loading tools, including support for Model Context Protocol (MCP).
*   **Knowledge Base Integration**: Built-in support for vector stores and document loaders to ground agents in custom data.
*   **Observability**: Integrated tracing and monitoring (e.g., via Langfuse) for debugging and performance analysis.
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

| Method | Description | Async |
| :--- | :--- | :---: |
| `initialize()` | Initializes the agent's resources and connections. Must set `self._initialized = True`. | ✅ |
| `ainvoke(user_message, config)` | Asynchronously processes a message and returns the full response. | ✅ |
| `invoke(user_message, config)` | Synchronously processes a message and returns the full response. | ❌ |
| `astream(user_message, config)` | Asynchronously streams the response chunks. | ✅ |
| `stream(user_message, config)` | Synchronously streams the response chunks (or wraps async stream). | ✅ |

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

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.
