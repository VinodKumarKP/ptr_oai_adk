"""Deployment-target awareness for the OAI ADK.

Lets agent-core components adapt their behaviour when running under a
constrained, serverless runtime such as AWS Bedrock AgentCore, where each user
session runs in an ephemeral, one-shot microVM. In that model, actions that
mutate the local filesystem at request/init time (for example installing
packages) are both wasteful and unsafe: the filesystem is discarded when the
session ends, so the work is repeated on every session and never persists.

The deployment target is configured *explicitly* via the ``DEPLOYMENT_TARGET``
environment variable (set by the deployment image), so detection is
deterministic rather than guessed from ambient signals.
"""
import os

# Environment variable names
ENV_DEPLOYMENT_TARGET = "DEPLOYMENT_TARGET"
ENV_DISABLE_RUNTIME_PACKAGE_INSTALL = "DISABLE_RUNTIME_PACKAGE_INSTALL"

# Known deployment targets
TARGET_AGENTCORE = "agentcore"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def get_deployment_target() -> str:
    """Return the configured deployment target (lower-cased), or '' if unset."""
    return os.environ.get(ENV_DEPLOYMENT_TARGET, "").strip().lower()


def is_agentcore_runtime() -> bool:
    """True when the ADK is running under AWS Bedrock AgentCore.

    Controlled by ``DEPLOYMENT_TARGET=agentcore``, which the AgentCore
    deployment image sets.
    """
    return get_deployment_target() == TARGET_AGENTCORE


def _flag_enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in _TRUTHY


def is_runtime_package_install_disabled() -> bool:
    """True when components must not install packages at request/init time.

    Runtime installs (e.g. ``guardrails hub install``) mutate the local
    filesystem, which is pointless and slow on an ephemeral per-session
    microVM — dependencies must instead be baked into the deployment image.

    This is enabled automatically under AgentCore, and can also be forced in
    any environment via ``DISABLE_RUNTIME_PACKAGE_INSTALL=true``.
    """
    return is_agentcore_runtime() or _flag_enabled(ENV_DISABLE_RUNTIME_PACKAGE_INSTALL)
