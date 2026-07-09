"""AG-UI gateway router.

Mounts POST /agui — an AG-UI-protocol endpoint (RunAgentInput in, SSE event
stream out) that fronts the A2A endpoint served by this same app. See
``oai_agent_server.agui.gateway`` for the translation rules.

Enabled when the a2a mode is active and the optional ``ag-ui-protocol``
package is installed (``pip install oai-agent-server[agui]``). Disable
explicitly with ``ENABLE_AGUI_GATEWAY=false``. Point ``AGUI_A2A_URL`` at a
remote A2A agent to act as a standalone gateway for it; by default the
gateway calls this server's own /a2a endpoint.
"""

import os
from typing import Dict, List, Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from oai_agent_core.utils.logger import get_logger

try:
    from ag_ui.core import RunAgentInput
    from oai_agent_server.agui.gateway import stream_agui_events
    AGUI_SDK_AVAILABLE = True
except ImportError:
    RunAgentInput = None
    stream_agui_events = None
    AGUI_SDK_AVAILABLE = False


def create_agui_router(
    allowed_modes: List[str],
    a2a_path: str = "/a2a/",
) -> Optional[APIRouter]:
    logger = get_logger()

    if "a2a" not in allowed_modes:
        logger.info("AG-UI gateway skipped — requires the a2a mode")
        return None

    if os.environ.get("ENABLE_AGUI_GATEWAY", "true").lower() == "false":
        logger.info("AG-UI gateway disabled via ENABLE_AGUI_GATEWAY=false")
        return None

    if not AGUI_SDK_AVAILABLE:
        logger.info(
            "AG-UI gateway skipped — 'ag-ui-protocol' is not installed. "
            "Install with: pip install \"oai-agent-server[agui]\""
        )
        return None

    router = APIRouter()

    # AG-UI threadId -> A2A contextId, so follow-up runs on a thread continue
    # the same A2A conversation; threadId -> A2UI messages so state snapshots
    # cover the whole thread. In-memory and per-process by design: the A2A
    # task store owns durable state; losing these maps only starts a fresh
    # context, mirroring what a reconnecting A2UI client does today.
    thread_contexts: Dict[str, str] = {}
    thread_a2ui: Dict[str, list] = {}

    @router.post("/", tags=["agui"])
    async def run_agent(run_input: RunAgentInput, request: Request) -> StreamingResponse:
        a2a_url = os.environ.get("AGUI_A2A_URL") or (
            str(request.base_url).rstrip("/") + a2a_path
        )
        return StreamingResponse(
            stream_agui_events(a2a_url, run_input, thread_contexts, thread_a2ui),
            media_type="text/event-stream",
        )

    logger.info("AG-UI gateway routes included at /agui (bridging to %s)",
                os.environ.get("AGUI_A2A_URL", a2a_path))
    return router
