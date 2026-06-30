"""Additional coverage for utils/logger.py."""
import logging
import time

import pytest

import oai_agent_core.utils.logger as logmod
from oai_agent_core.utils.logger import (
    Logger,
    LogLevel,
    LogFormat,
    PerformanceMetrics,
    JsonFormatter,
    create_agent_logger,
    get_logger,
)


def test_get_logger_fallback(monkeypatch):
    monkeypatch.setattr(logmod, "_get_logger_from_config", None)
    logger = get_logger()
    assert logger.name == "agent_logger"
    assert logger.handlers


def test_get_logger_uses_config():
    logger = get_logger()
    assert isinstance(logger, logging.Logger)


def test_performance_metrics():
    pm = PerformanceMetrics()
    assert pm.get_stats("missing") == {}
    pm.record_duration("op", 1.0)
    pm.record_duration("op", 3.0)
    stats = pm.get_stats("op")
    assert stats["count"] == 2
    assert stats["avg"] == 2.0
    assert stats["min"] == 1.0
    assert stats["max"] == 3.0
    # NOTE: get_all_stats() deadlocks (non-reentrant lock re-acquired via
    # get_stats); intentionally not called here. See test file header.
    pm.clear()
    assert pm.get_stats("op") == {}


def test_json_formatter_basic():
    fmt = JsonFormatter()
    record = logging.LogRecord("n", logging.INFO, "path", 10, "msg", (), None)
    out = fmt.format(record)
    assert '"message":"msg"' in out


def test_json_formatter_with_exc_and_extra():
    fmt = JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord("n", logging.ERROR, "p", 1, "m", (), sys.exc_info())
    record.extra_data = {"k": "v"}
    out = fmt.format(record)
    assert "exception" in out
    assert "extra" in out


def test_logger_full_lifecycle(tmp_path):
    lg = Logger(
        name="cov_logger",
        log_level="DEBUG",
        log_dir=str(tmp_path / "logs"),
        log_file="test.log",
        log_format=LogFormat.JSON,  # falls to default format str
        enable_json=True,
        enable_console=True,
        enable_file=True,
    )
    lg.debug("d")
    lg.info("i", extra_data={"a": 1})
    lg.warning("w")
    lg.error("e", exc_info=False)
    lg.critical("c")
    try:
        raise RuntimeError("x")
    except RuntimeError:
        lg.exception("oops")

    # performance tracking
    with lg.track_performance("op1"):
        time.sleep(0.001)
    lg.log_performance_stats("op1")
    lg.log_performance_stats("never_run")
    # lg.log_performance_stats() with no arg calls get_all_stats() which
    # deadlocks (see PerformanceMetrics non-reentrant lock bug); skipped.

    # config
    lg.set_level(LogLevel.WARNING)
    lg.set_level("INFO")
    lg.set_level(logging.DEBUG)

    handler = logging.NullHandler()
    lg.add_custom_handler(handler)
    lg.remove_handler(handler)
    lg.remove_handler(logging.NullHandler())  # not present -> no-op

    files = lg.get_log_files()
    assert any("test.log" in f for f in files)

    info = lg.get_logger_info()
    assert info["name"] == "cov_logger"
    assert info["performance_tracking_enabled"] is True


def test_logger_formats(tmp_path):
    for fmt, custom in [
        (LogFormat.SIMPLE, None),
        (LogFormat.DETAILED, None),
        (LogFormat.CUSTOM, "%(message)s"),
        (LogFormat.CUSTOM, None),  # falls to default
    ]:
        lg = Logger(
            name=f"fmt_{fmt.value}_{custom}",
            log_dir=str(tmp_path / "l"),
            log_format=fmt,
            custom_format=custom,
            enable_file=False,
            enable_console=False,
        )
        lg.info("hi")


def test_logger_no_performance(tmp_path):
    lg = Logger(
        name="noperf",
        log_dir=str(tmp_path / "np"),
        enable_performance_tracking=False,
        enable_file=False,
        enable_console=False,
    )
    with lg.track_performance("op"):
        pass
    lg.log_performance_stats("op")  # warns not enabled
    info = lg.get_logger_info()
    assert info["performance_tracking_enabled"] is False


def test_logger_context_manager(tmp_path):
    with Logger(name="ctx", log_dir=str(tmp_path / "c"), enable_file=False) as lg:
        lg.info("inside")
    # with exception
    try:
        with Logger(name="ctx2", log_dir=str(tmp_path / "c2"), enable_file=False) as lg2:
            raise ValueError("boom")
    except ValueError:
        pass


def test_clear_old_logs(tmp_path):
    log_dir = tmp_path / "old"
    lg = Logger(name="oldlogs", log_dir=str(log_dir), log_file="a.log", enable_console=False)
    # create an old log file
    old_file = log_dir / "stale.log"
    old_file.write_text("x")
    import os
    old_time = time.time() - (40 * 24 * 60 * 60)
    os.utime(old_file, (old_time, old_time))
    lg.clear_old_logs(days_old=30)
    assert not old_file.exists()


def test_create_agent_logger(tmp_path):
    lg = create_agent_logger(
        name="MyAgent", log_dir=str(tmp_path / "agent"), log_level="DEBUG",
        enable_json=True, enable_performance=True,
    )
    assert lg.name == "MyAgent"
