"""Additional coverage for logging_config."""
import json
import logging
from unittest.mock import MagicMock

import oai_agent_core.utils.logging_config as lc


def test_configure_logging_with_file_and_console(tmp_path):
    lc._configured = False
    log_file = tmp_path / "sub" / "agent.log"
    lc.configure_logging(level=logging.DEBUG, log_file=str(log_file), console_output=True)
    assert log_file.parent.exists()
    root = logging.getLogger("oai_agent_core")
    assert root.level == logging.DEBUG
    # Second call short-circuits because already configured
    lc.configure_logging(level=logging.INFO)
    lc._configured = False


def test_get_logger():
    assert isinstance(lc.get_logger("x.y"), logging.Logger)


def test_setup_logging_relative_file(tmp_path, monkeypatch):
    lc._configured = False
    monkeypatch.chdir(tmp_path)
    lc.setup_logging(log_level=logging.WARNING, log_file="agent.log", console_output=False)
    assert (tmp_path / "logs" / "agent.log").parent.exists()
    lc._configured = False


def test_setup_logging_no_file():
    lc._configured = False
    lc.setup_logging(log_level=logging.INFO, console_output=False)
    lc._configured = False


def test_structured_logger():
    base = MagicMock(spec=logging.Logger)
    sl = lc.StructuredLogger(base)
    sl.log_structured(logging.INFO, "evt", extra="val")
    sl.log_component_init("Comp", {"v": "1"})
    sl.log_component_init("Comp")
    sl.log_operation_start("op", k=1)
    sl.log_operation_complete("op", duration_ms=5.0, k=1)
    sl.log_error_detail("Err", "message", k=1)
    assert base.log.call_count == 6
    # Ensure JSON serializable payloads were emitted
    payload = base.log.call_args_list[0].args[1]
    assert json.loads(payload)["event"] == "evt"
