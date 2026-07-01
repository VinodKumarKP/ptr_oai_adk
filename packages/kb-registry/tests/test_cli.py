"""
Unit tests for CLI entry point (cli.py).
"""

import sys
import os
import pytest
from unittest.mock import MagicMock, patch, ANY
from oai_kb_registry.cli import main


def test_cli_defaults():
    """Test CLI defaults when no custom arguments are provided."""
    # We need to simulate sys.argv or parse_args
    mock_args = MagicMock()
    mock_args.host = None
    mock_args.port = None
    mock_args.reload = False
    mock_args.no_auth = False
    mock_args.log_level = None
    mock_args.kb_data_dir = None
    mock_args.auto_start_infra = False
    mock_args.infra_compose_file = None
    mock_args.infra_startup_timeout = None
    mock_args.infra_services = None

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         patch("uvicorn.run") as mock_run, \
         patch.dict(os.environ, {}, clear=True):
        main()
        mock_run.assert_called_once()
        assert os.environ.get("HOST") == "0.0.0.0"
        assert os.environ.get("PORT") == "8085"


def test_cli_with_args():
    """Test CLI options correctly propagate to environment variables and uvicorn."""
    mock_args = MagicMock()
    mock_args.host = "1.2.3.4"
    mock_args.port = 9000
    mock_args.reload = True
    mock_args.no_auth = True
    mock_args.log_level = "debug"
    mock_args.kb_data_dir = "/custom/data/dir"
    mock_args.auto_start_infra = True
    mock_args.infra_compose_file = "/custom/compose.yaml"
    mock_args.infra_startup_timeout = 120
    mock_args.infra_services = ["postgres", "valkey", "pgvector"]

    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         patch("uvicorn.run") as mock_run, \
         patch.dict(os.environ, {}, clear=True):
        main()
        mock_run.assert_called_once_with(
            ANY,
            host="1.2.3.4",
            port=9000,
            workers=1,
            reload=True,
            proxy_headers=True,
            forwarded_allow_ips="*",
            timeout_keep_alive=300,
            loop=ANY,
            log_level="debug",
        )
        assert os.environ.get("KB_AUTH_ENABLED") == "false"
        assert os.environ.get("KB_DATA_DIR") == "/custom/data/dir"
        assert os.environ.get("AUTO_START_INFRA") == "true"
        assert os.environ.get("INFRA_COMPOSE_FILE") == "/custom/compose.yaml"
        assert os.environ.get("INFRA_STARTUP_TIMEOUT") == "120"


def test_cli_uvloop_platform():
    """Test loop type selection on different platforms."""
    mock_args = MagicMock()
    mock_args.host = None
    mock_args.port = None
    mock_args.reload = False
    mock_args.no_auth = False
    mock_args.log_level = None
    mock_args.kb_data_dir = None
    mock_args.auto_start_infra = False
    mock_args.infra_compose_file = None
    mock_args.infra_startup_timeout = None
    mock_args.infra_services = None

    # Test on non-Windows (should use uvloop)
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         patch("uvicorn.run") as mock_run, \
         patch("platform.system", return_value="Darwin"):
        main()
        _, kwargs = mock_run.call_args
        assert kwargs["loop"] == "uvloop"

    # Test on Windows (should use asyncio)
    with patch("argparse.ArgumentParser.parse_args", return_value=mock_args), \
         patch("uvicorn.run") as mock_run, \
         patch("platform.system", return_value="Windows"):
        main()
        _, kwargs = mock_run.call_args
        assert kwargs["loop"] == "asyncio"
