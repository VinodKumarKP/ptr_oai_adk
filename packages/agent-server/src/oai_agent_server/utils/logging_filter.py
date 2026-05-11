"""Logging filter that injects the active request_id onto every record.

Attached to the root logger (and uvicorn's loggers) by the server during
startup so log output — text or JSON — always carries the per-request id.
"""

import logging

from oai_agent_server.middleware.request_tracking import request_id_ctx


class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get()
        return True
