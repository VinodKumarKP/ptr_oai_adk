# ============================================================
# Dockerfile — unified agent app image (ADK / CrewAI / LangGraph)
#
# Replaces: Dockerfile, Dockerfile.crewai, Dockerfile.langgraph
#
# Build examples:
#   # Default (ADK)
#   docker build \
#     --secret id=github_token,env=GITHUB_TOKEN \
#     --build-arg GITHUB_URL=https://github.com/Capgemini-Innersource/my-agent \
#     -t my-agent:latest .
#
#   # CrewAI variant
#   docker build \
#     --secret id=github_token,env=GITHUB_TOKEN \
#     --build-arg BASE_IMAGE=oai-adk-crewai-base-image:latest \
#     --build-arg GITHUB_URL=https://github.com/Capgemini-Innersource/my-crewai-agent \
#     -t my-crewai-agent:latest .
#
#   # LangGraph variant
#   docker build \
#     --secret id=github_token,env=GITHUB_TOKEN \
#     --build-arg BASE_IMAGE=oai-adk-langgraph-base-image:latest \
#     --build-arg GITHUB_URL=https://github.com/Capgemini-Innersource/my-langgraph-agent \
#     -t my-langgraph-agent:latest .
# ============================================================

# Global ARG — available to all FROM lines
ARG BASE_IMAGE=oai-adk-base-image:latest

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
    if [ -n "${GITHUB_URL}" ]; then \
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
COPY --from=builder /tmp/repo/agentic_registry_agents /app/agentic_registry_agents

# Certs and entrypoint
COPY --chown=agentuser:agentuser cert/public.pem  /app/cert/public.pem
COPY --chown=agentuser:agentuser entrypoint.sh     /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

USER agentuser

# PYTHONDONTWRITEBYTECODE suppresses .pyc files — no need to find/delete them.
# GITHUB_TOKEN is intentionally omitted: supply it at runtime via -e if needed.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    JWT_PUBLIC_KEY_PATH=/app/cert/public.pem

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]