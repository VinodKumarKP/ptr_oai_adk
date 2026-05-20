# OAI Platform Core

`oai-platform-core` is a foundational Python package that provides shared utilities and base functionalities across various OAI (Open Agent Initiative) projects. It encapsulates common patterns, helper functions, and core components that are reused by other OAI packages, ensuring consistency and reducing code duplication.

## Key Features

-   **Database Abstractions**: Provides common interfaces and implementations for database interactions, supporting both PostgreSQL and SQLite with automatic fallback mechanisms.
-   **Security Utilities**: Includes token management (generation, validation, revocation) and SAML token validation for secure communication and authentication.
-   **Networking Helpers**: Offers functions for retrieving public and private IP addresses, essential for service discovery and communication.
-   **Logging Utilities**: Standardized logging setup with rotating file handlers and console output, ensuring consistent log formats across all OAI services.
-   **Shared Exception Hierarchy**: Defines a common base exception (`OAIBaseException`) and specialized exceptions for authentication, allowing for unified error handling across the platform.
-   **Deployment Utilities**: Contains base classes and helpers for deploying services, including Docker Compose and Python package deployment strategies.

## Usage by Other Packages

This package serves as a core dependency for other OAI components, including:

-   `oai-agent-client`: Utilizes networking and logging utilities.
-   `oai-agent-registry`: Leverages database abstractions, security utilities, and deployment helpers.
-   `oai-mcp-registry`: Depends on database abstractions, security utilities, and deployment helpers.
-   `oai-skills-registry`: Uses database abstractions and logging utilities.
-   `oai-agent-core`: Integrates various core functionalities and utilities.

By centralizing these common functionalities, `oai-platform-core` promotes a consistent and maintainable codebase across the entire OAI ecosystem.

## Installation

```bash
uv pip install "oai-platform-core @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit.git@main#subdirectory=packages/platform-core"
```

## Modules Overview

-   **`db/`**: Database backend abstractions (PostgresBackend, SQLiteBackend).
-   **`security/`**: Token management (`token_manager.py`, `token_utils.py`) and SAML validation (`saml_token_validation.py`).
-   **`deployers/`**: Base classes and utilities for deployment strategies.
-   **`exceptions.py`**: Custom exception classes for platform-wide error handling.
-   **`networking.py`**: Functions for IP address retrieval.
-   **`logging_utils.py`**: Standardized logger setup.

## Development

To run tests or contribute to `oai-platform-core`, ensure you have the development dependencies installed.

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    (Note: `requirements.txt` should reflect the dependencies in `pyproject.toml`)

2.  **Run tests**:
    ```bash
    pytest
    ```
