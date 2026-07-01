import os
import sys
from unittest.mock import patch, MagicMock

import pytest

from oai_skills_registry.cli import main
import oai_skills_registry.dependencies as _deps


@pytest.fixture(autouse=True)
def clean_env_and_deps():
    orig_env = dict(os.environ)
    orig_auto = _deps._auto_start_infra
    orig_compose = _deps._infra_compose_file
    orig_timeout = _deps._infra_startup_timeout
    yield
    os.environ.clear()
    os.environ.update(orig_env)
    _deps._auto_start_infra = orig_auto
    _deps._infra_compose_file = orig_compose
    _deps._infra_startup_timeout = orig_timeout


def test_cli_flags_parsing():
    """Test that main parses arguments correctly and sets environment/module variables."""
    test_args = [
        "oai-skills-registry",
        "--host", "127.0.0.1",
        "--port", "9000",
        "--auto-start-infra",
        "--infra-compose-file", "/path/to/compose.yaml",
        "--infra-startup-timeout", "45",
    ]

    with patch.object(sys, "argv", test_args):
        with patch("uvicorn.run") as mock_run:
            main()
            
            # Check dependency values are overridden
            assert _deps._auto_start_infra is True
            assert _deps._infra_compose_file == "/path/to/compose.yaml"
            assert _deps._infra_startup_timeout == 45

            # Check environment variables are populated
            assert os.environ.get("HOST") == "127.0.0.1"
            assert os.environ.get("PORT") == "9000"
            assert os.environ.get("AUTO_START_INFRA") == "true"
            assert os.environ.get("INFRA_COMPOSE_FILE") == "/path/to/compose.yaml"
            assert os.environ.get("INFRA_STARTUP_TIMEOUT") == "45"

            # Check uvicorn was called
            mock_run.assert_called_once()
            args, kwargs = mock_run.call_args
            assert kwargs["host"] == "127.0.0.1"
            assert kwargs["port"] == 9000


def test_cli_defaults():
    """Test that main uses env var defaults when flags are omitted."""
    test_args = ["oai-skills-registry"]

    with patch.dict(os.environ, {"HOST": "10.0.0.1", "PORT": "1234"}):
        # Temporarily clean module-level state of dependencies
        with patch.object(_deps, "_auto_start_infra", False), \
             patch.object(_deps, "_infra_compose_file", None), \
             patch.object(_deps, "_infra_startup_timeout", 60):
            with patch.object(sys, "argv", test_args):
                with patch("uvicorn.run") as mock_run:
                    main()
                    assert os.environ.get("HOST") == "10.0.0.1"
                    assert os.environ.get("PORT") == "1234"
                    mock_run.assert_called_once()
                    assert mock_run.call_args[1]["host"] == "10.0.0.1"
                    assert mock_run.call_args[1]["port"] == 1234
