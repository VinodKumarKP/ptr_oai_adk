# Runtime Generic Dockerfile for MCP Servers
FROM python:3.11-slim

ARG GITHUB_TOKEN
# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    unzip \
    dos2unix \
    && mkdir -p /tmp/awscli \
    && curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "/tmp/awscli/awscliv2.zip" \
    && unzip /tmp/awscli/awscliv2.zip -d /tmp/awscli \
    && /tmp/awscli/aws/install \
    && rm -rf /tmp/awscli /var/lib/apt/lists/*

COPY entrypoint.sh /app/entrypoint.sh
RUN dos2unix /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ARG GITHUB_TOKEN
# Make script executable and set ownership
RUN chmod +x /app/entrypoint.sh && \
    git config --system url.https://oauth2:${GITHUB_TOKEN}@github.com/Capgemini-Innersource.insteadOf https://github.com/Capgemini-Innersource


# Copy requirements and install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire agentic_registry_agents package
COPY agentic_registry_agents/ ./agentic_registry_agents/


ENV GITHUB_TOKEN=${GITHUB_TOKEN}

# Create a non-root user and change ownership
RUN useradd -m -u 1000 mcpuser && chown -R mcpuser:mcpuser /app
USER mcpuser

# Use the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]