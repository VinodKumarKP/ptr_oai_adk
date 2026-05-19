"""
logger_utils — logging helper for oai_mcp_server_core.

Delegates to the canonical implementation in ``oai_platform_core.logging_utils``
so that all OAI packages share the same rotating-file + console logger setup.

Backward-compatible: existing callers that do::

    from oai_mcp_server_core.utils.logger_utils import get_logger
    logger = get_logger()

continue to work without any changes.
"""

from oai_platform_core.logging_utils import get_logger  # noqa: F401 — re-export

__all__ = ["get_logger"]
