# ============================================================
# Dockerfile — unified MCP server app image
#
# Build examples:
#   # Default
#   docker build \
#     --secret id=github_token,env=GITHUB_TOKEN \
#     --build-arg GITHUB_URL=https://github.com/Capgemini-Innersource/my-mcp-server \
#     -t my-mcp-server:latest .
# ============================================================

# Global ARG — available to all FROM lines
ARG BASE_IMAGE=oai-adk-mcp-base-image:latest

# ── Stage 1: dependency installer ────────────────────────────
FROM ${BASE_IMAGE} AS builder

ARG GITHUB_URL
# CACHE_BUST lets CI force a fresh clone without changing source:
#   --build-arg CACHE_BUST=$(date +%s)
ARG CACHE_BUST=default

WORKDIR /install

# Clone once; install deps from requirements.txt.
# Git auth is inherited from the base image's system git config.
RUN echo "Dep cache bust: ${CACHE_BUST}" && \
    mkdir -p /tmp/repo/mcp_registry_servers && \
    touch /tmp/repo/requirements.txt && \
    if [ -n "${GITHUB_URL}" ]; then \
        rm -rf /tmp/repo && \
        git clone "${GITHUB_URL}" /tmp/repo && \
        uv pip install --system --prefix=/install \
            -r /tmp/repo/requirements.txt; \
    fi

# ── Stage 2: final runtime image ─────────────────────────────
FROM ${BASE_IMAGE}

ARG GITHUB_URL
ARG CACHE_BUST=default

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code from the same clone (single network round-trip)
COPY --from=builder /tmp/repo/mcp_registry_servers /app/mcp_registry_servers

# Entrypoint
COPY --chown=mcpuser:mcpuser entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

USER mcpuser

# PYTHONDONTWRITEBYTECODE suppresses .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

ENTRYPOINT ["/app/entrypoint.sh"]
