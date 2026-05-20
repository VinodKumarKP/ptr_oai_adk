"""
Tests for oai_platform_core.deployers

Covers:
- BaseDeployer (abstract interface)
- _is_port_available / _get_next_available_port (docker_compose_base helpers)
- BaseDockerComposeManager (compose generation, registration, port allocation)
- BasePythonPackageDeployer (init, port allocation, service lifecycle helpers)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from oai_platform_core.deployers.base import BaseDeployer
from oai_platform_core.deployers.docker_compose_base import (
    _is_port_available,
    _get_next_available_port,
    BaseDockerComposeManager,
)
from oai_platform_core.deployers.python_package_base import BasePythonPackageDeployer


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# BaseDeployer (abstract — just verify the interface)
# ---------------------------------------------------------------------------

class TestBaseDeployer:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseDeployer()  # type: ignore[abstract]

    def test_concrete_subclass_with_all_methods_works(self):
        class Concrete(BaseDeployer):
            async def initialize(self): pass
            async def shutdown(self): pass
            def find_available_port(self): return 9000
            def image_exists(self, name, version): return False

        c = Concrete()
        assert c.find_available_port() == 9000
        assert c.image_exists("x", "1.0") is False

    def test_missing_abstract_method_prevents_instantiation(self):
        class Partial(BaseDeployer):
            async def initialize(self): pass
            async def shutdown(self): pass
            def find_available_port(self): return 0
            # image_exists not implemented

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Port helpers
# ---------------------------------------------------------------------------

class TestIsPortAvailable:
    def test_returns_true_for_available_port(self):
        # Find a port that has nothing bound
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            free_port = s.getsockname()[1]
        # Port is freed when context exits; should be bindable again
        assert _is_port_available(free_port) is True

    def test_returns_false_for_occupied_port(self):
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            occupied = s.getsockname()[1]
            assert _is_port_available(occupied) is False


class TestGetNextAvailablePort:
    def test_returns_a_usable_port(self):
        port = _get_next_available_port(start_port=20000)
        assert isinstance(port, int)
        assert port >= 20000

    def test_skips_already_used_ports(self):
        import socket as _socket
        with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            occupied = s.getsockname()[1]
            port = _get_next_available_port(
                used_ports=[occupied], start_port=occupied
            )
        assert port != occupied

    def test_skip_ports_respected(self):
        port = _get_next_available_port(
            skip_ports=[8000, 8001, 8002], start_port=8000
        )
        assert port not in (8000, 8001, 8002)

    def test_default_skip_ports_excluded(self):
        # Default skip list is [8080, 8081, 8082]
        port = _get_next_available_port(start_port=8080)
        assert port not in (8080, 8081, 8082)


# ---------------------------------------------------------------------------
# BaseDockerComposeManager concrete subclass for tests
# ---------------------------------------------------------------------------

class _ConcreteDockerComposeManager(BaseDockerComposeManager):
    _compose_name = "test-registry"
    _network_name = "test-network"
    _service_type_label = "test-service"

    def _get_service_env(self, service_name, port, env_overrides, base_url, local_registry_url):
        env = {"PORT": str(port), "SERVICE_NAME": service_name}
        env.update(env_overrides or {})
        return env

    def _build_service_dict(self, service_name, config, base_url, local_registry_url, refresh_repo=False):
        return {
            "image": f"test-{service_name}:latest",
            "ports": [f"{config.get('port', 8000)}:8000"],
        }

    def _build_infra_compose_dict(self):
        return {
            "name": self._compose_name,
            "services": {
                "postgres": {"image": "postgres:15"},
                "valkey": {"image": "valkey/valkey:8"},
            },
        }


class TestBaseDockerComposeManager:
    @pytest.fixture
    def tmp_compose_dir(self, tmp_path):
        return tmp_path

    @pytest.fixture
    def manager(self, tmp_compose_dir):
        return _ConcreteDockerComposeManager(
            seed_config={},
            compose_output_path=str(tmp_compose_dir / "services.yaml"),
            base_compose_path=str(tmp_compose_dir / "infra.yaml"),
            agent_base_url="localhost:8081",
            agent_local_registry_url="http://host.docker.internal:8081",
        )

    def test_instantiation(self, manager):
        assert manager._compose_name == "test-registry"
        assert manager._network_name == "test-network"

    def test_find_available_port_returns_int(self, manager):
        port = manager.find_available_port()
        assert isinstance(port, int)
        assert port > 0

    def test_find_available_port_skips_used_ports(self, manager):
        manager.used_ports = [8000, 8001, 8002]
        port = manager.find_available_port()
        assert port not in manager.used_ports

    def test_image_exists_checks_docker(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="sha256:abc")
            assert manager.image_exists("my-service", "1.0") is True
            mock_run.assert_called_once()

    def test_image_exists_returns_false_on_failure(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")
            assert manager.image_exists("my-service", "1.0") is False

    def test_write_compose_file_creates_files(self, manager, tmp_compose_dir):
        manager._seed_config = {
            "svc1": {"source": "http://git/repo", "port": 9100},
        }
        output = manager.write_compose_file()
        assert Path(str(tmp_compose_dir / "services.yaml")).exists()
        assert Path(str(tmp_compose_dir / "infra.yaml")).exists()

    def test_add_service_from_registration_stages_service(self, manager):
        config = manager._add_service_from_registration(
            "new-service",
            "http://git/new-service.git",
            framework="langchain",
            port=9200,
        )
        assert "new-service" in manager._dynamic_services
        assert config["port"] == 9200
        assert config["source"] == "http://git/new-service.git"
        assert "langchain" in config["tags"]

    def test_add_service_from_registration_auto_assigns_port(self, manager):
        config = manager._add_service_from_registration(
            "auto-port-svc",
            "http://git/svc.git",
            port=None,
        )
        assert isinstance(config["port"], int)
        assert config["port"] > 0

    def test_add_service_tags_include_framework(self, manager):
        config = manager._add_service_from_registration(
            "svc", "url", framework="openai", tags=["existing"]
        )
        assert "openai" in config["tags"]
        assert "existing" in config["tags"]

    def test_add_service_framework_not_duplicated_in_tags(self, manager):
        config = manager._add_service_from_registration(
            "svc", "url", framework="langchain", tags=["langchain"]
        )
        assert config["tags"].count("langchain") == 1

    def test_build_environment_list_returns_list(self, manager):
        result = manager._build_environment_list(
            "svc", 9000, {}, "localhost", "http://h:8081"
        )
        assert isinstance(result, list)
        assert any("PORT=9000" in item for item in result)

    def test_build_services_compose_dict_skips_no_source(self, manager):
        # Services without 'source' are skipped with a warning
        cfg = {"nosource": {"port": 9000}}  # no 'source' key
        result = manager._build_services_compose_dict(cfg)
        assert "nosource" not in result["services"]

    def test_build_services_compose_dict_structure(self, manager):
        cfg = {"svc": {"source": "http://git/r", "port": 9000}}
        result = manager._build_services_compose_dict(cfg)
        assert result["name"] == "test-registry"
        assert "svc" in result["services"]
        assert result["networks"] == {"test-network": None}

    def test_shutdown_calls_compose_down(self, manager):
        with patch.object(manager, "_run_compose_down", new=AsyncMock()) as mock_down:
            run(manager.shutdown())
            mock_down.assert_called_once()

    def test_run_compose_up_service_raises_on_failure(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stderr="some error", stdout=""
            )
            with pytest.raises(RuntimeError, match="docker compose up failed"):
                manager._run_compose_up_service("svc1")

    def test_run_compose_up_service_returns_stdout_on_success(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0, stderr="", stdout="done\n"
            )
            result = manager._run_compose_up_service("svc1")
            assert result == "done\n"

    def test_run_compose_stop_service(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = manager._run_compose_stop_service("svc1")
            assert result == "ok"

    def test_run_compose_stop_service_raises_on_failure(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail", stdout="")
            with pytest.raises(RuntimeError, match="docker compose stop failed"):
                manager._run_compose_stop_service("svc1")

    def test_run_compose_down_service(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            result = manager._run_compose_down_service("svc1")
            assert result == "ok"

    def test_run_compose_down_service_raises_on_failure(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail", stdout="")
            with pytest.raises(RuntimeError, match="docker compose down failed"):
                manager._run_compose_down_service("svc1")

    def test_run_compose_up_success(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="started", stderr="")
            result = manager._run_compose_up()
            assert result == "started"

    def test_run_compose_up_raises_on_failure(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail", stdout="")
            with pytest.raises(RuntimeError, match="docker compose up failed"):
                manager._run_compose_up()

    def test_run_compose_down_success(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="down", stderr="")
            result = run(manager._run_compose_down())
            assert result == "down"

    def test_run_compose_down_raises_on_failure(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="fail", stdout="")
            with pytest.raises(RuntimeError, match="docker compose down failed"):
                run(manager._run_compose_down())

    def test_initialize_calls_write_and_up(self, manager):
        with patch.object(manager, "write_compose_file") as mock_write, \
             patch.object(manager, "_run_compose_up", return_value="ok") as mock_up:
            run(manager.initialize())
            mock_write.assert_called_once()
            mock_up.assert_called_once()

    def test_start_infra_services_with_wait_flag(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            with patch.object(manager, "write_compose_file"):
                manager.start_infra_services()
            # First call should include --wait
            first_call_args = mock_run.call_args_list[0][0][0]
            assert "--wait" in first_call_args

    def test_start_infra_services_retries_without_wait(self, manager):
        """If --wait is unsupported, should retry without it."""
        def side_effect(cmd, **kw):
            if "--wait" in cmd:
                return MagicMock(
                    returncode=1,
                    stdout="unknown flag: --wait",
                    stderr="unknown flag: --wait",
                )
            return MagicMock(returncode=0, stdout="", stderr="")

        with patch("subprocess.run", side_effect=side_effect):
            with patch.object(manager, "write_compose_file"):
                manager.start_infra_services()  # should not raise

    def test_start_infra_services_raises_on_real_failure(self, manager):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="real error", stderr="real error"
            )
            with patch.object(manager, "write_compose_file"):
                with pytest.raises(RuntimeError, match="docker compose up failed"):
                    manager.start_infra_services()

    def test_write_yaml_creates_file(self, tmp_path):
        from oai_platform_core.deployers.docker_compose_base import BaseDockerComposeManager
        path = tmp_path / "test.yaml"
        BaseDockerComposeManager._write_yaml({"key": "value"}, path)
        assert path.exists()
        assert "key" in path.read_text()

    def test_build_environment_list_dollar_var_passthrough(self, manager):
        """Values starting with ${ are passed through as-is."""
        result = manager._build_environment_list(
            "svc", 9000, {"MY_VAR": "${HOST_IP:-localhost}"}, "localhost", "http://h"
        )
        assert any("MY_VAR=${HOST_IP:-localhost}" in item for item in result)


# ---------------------------------------------------------------------------
# BasePythonPackageDeployer — additional coverage tests
# ---------------------------------------------------------------------------

class TestPythonDeployerLifecycle:
    @pytest.fixture
    def deployer(self, tmp_path):
        return _ConcretePythonDeployer(
            seed_config={},
            base_dir=str(tmp_path / "services"),
            agent_base_url="localhost:8081",
            agent_local_registry_url="http://host.docker.internal:8081",
        )

    def test_stop_service_not_running(self, deployer):
        result = deployer._stop_service("not-there")
        assert result == "Not running"

    def test_stop_service_running(self, deployer):
        mock_proc = MagicMock()
        mock_proc.wait = MagicMock()
        deployer.running_processes["svc"] = mock_proc
        result = deployer._stop_service("svc")
        assert result == "Stopped"
        mock_proc.terminate.assert_called_once()
        assert "svc" not in deployer.running_processes

    def test_stop_service_kill_on_timeout(self, deployer):
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired(cmd="x", timeout=5)
        deployer.running_processes["svc"] = mock_proc
        deployer._stop_service("svc")
        mock_proc.kill.assert_called_once()

    def test_remove_service_stops_and_deregisters(self, deployer):
        mock_proc = MagicMock()
        deployer.running_processes["svc"] = mock_proc
        deployer._dynamic_services["svc"] = {"port": 9000, "repo_name": "r"}
        result = deployer._remove_service("svc")
        assert result == "Removed"
        assert "svc" not in deployer._dynamic_services

    def test_get_repo_name_strips_dot_git(self, deployer):
        assert deployer._get_repo_name("https://github.com/org/myrepo.git") == "myrepo"

    def test_get_repo_name_no_dot_git(self, deployer):
        assert deployer._get_repo_name("https://github.com/org/myrepo") == "myrepo"

    def test_start_service_launches_subprocess(self, deployer, tmp_path):
        service_dir = deployer.base_dir / "myrepo"
        service_dir.mkdir(parents=True)
        deployer._dynamic_services["svc"] = {
            "port": 9100,
            "repo_name": "myrepo",
            "env": {},
        }
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            result = deployer._start_service("svc")
        assert result == "Started"
        assert "svc" in deployer.running_processes
        mock_popen.assert_called_once()

    def test_start_service_skips_if_already_running(self, deployer):
        deployer.running_processes["svc"] = MagicMock()
        result = deployer._start_service("svc")
        assert result == "Already running"


# ---------------------------------------------------------------------------
# BasePythonPackageDeployer concrete subclass
# ---------------------------------------------------------------------------

class _ConcretePythonDeployer(BasePythonPackageDeployer):
    def _get_service_env(self, service_name, port, base_url, local_registry_url,
                         env_overrides, deployment_mode):
        return {"PORT": str(port), "SERVICE": service_name}

    def _get_server_py_path(self, service_dir, service_name):
        return service_dir / "server.py"

    def _get_start_command(self, venv_python, server_py, port):
        return [str(venv_python), str(server_py), "--port", str(port)]

    async def initialize(self): pass
    async def shutdown(self): pass
    def image_exists(self, service_name, version): return False


class TestBasePythonPackageDeployer:
    @pytest.fixture
    def deployer(self, tmp_path):
        return _ConcretePythonDeployer(
            seed_config={},
            base_dir=str(tmp_path / "services"),
            agent_base_url="localhost:8081",
            agent_local_registry_url="http://host.docker.internal:8081",
        )

    def test_instantiation(self, deployer, tmp_path):
        assert (tmp_path / "services").exists()

    def test_find_available_port_returns_int(self, deployer):
        port = deployer.find_available_port()
        assert isinstance(port, int)
        assert port > 0

    def test_find_available_port_skips_used(self, deployer):
        deployer.used_ports = [8000, 8001]
        port = deployer.find_available_port()
        assert port not in deployer.used_ports

    def test_image_exists_always_false_in_concrete(self, deployer):
        # _ConcretePythonDeployer overrides image_exists to always return False
        assert deployer.image_exists("missing", "1.0") is False

    def test_base_image_exists_checks_repo_dir(self, deployer):
        # Test BasePythonPackageDeployer.image_exists directly (not the override)
        deployer._dynamic_services["svc"] = {"repo_name": "myrepo"}
        (deployer.base_dir / "myrepo").mkdir(parents=True)
        result = BasePythonPackageDeployer.image_exists(deployer, "svc", "any")
        assert result is True

    def test_base_image_exists_false_when_no_config(self, deployer):
        result = BasePythonPackageDeployer.image_exists(deployer, "ghost", "1.0")
        assert result is False

    def test_base_image_exists_false_when_no_repo_name(self, deployer):
        deployer._dynamic_services["svc"] = {}
        result = BasePythonPackageDeployer.image_exists(deployer, "svc", "any")
        assert result is False

    def test_base_image_exists_false_when_dir_missing(self, deployer):
        deployer._dynamic_services["svc"] = {"repo_name": "nonexistent"}
        result = BasePythonPackageDeployer.image_exists(deployer, "svc", "any")
        assert result is False

    def test_start_service_raises_when_no_config(self, deployer):
        with pytest.raises(ValueError, match="configuration not found"):
            deployer._start_service("missing-service")

    def test_start_service_returns_already_running(self, deployer):
        deployer.running_processes["svc"] = MagicMock()
        result = deployer._start_service("svc")
        assert result == "Already running"

    def test_stop_service_not_running_noop(self, deployer):
        result = deployer._stop_service("not-running")
        assert result is None or isinstance(result, str)

    def test_remove_service_clears_from_dynamic(self, deployer):
        deployer._dynamic_services["svc"] = {"port": 9000, "repo_name": "r"}
        deployer._remove_service("svc")
        assert "svc" not in deployer._dynamic_services

    def test_get_repo_name_from_git_url(self, deployer):
        url = "https://github.com/org/my-service.git"
        assert deployer._get_repo_name(url) == "my-service"

    def test_get_repo_name_without_dot_git(self, deployer):
        url = "https://github.com/org/service"
        assert deployer._get_repo_name(url) == "service"

    def test_find_available_port_raises_when_no_ports(self, deployer):
        """Exhaustion: fill used_ports with first 10000 from 8000."""
        with patch("oai_platform_core.deployers.python_package_base.socket") as mock_socket:
            mock_socket.socket.return_value.__enter__.return_value.bind.side_effect = OSError
            deployer.used_ports = list(range(8000, 18001))
            with pytest.raises(RuntimeError, match="Could not find"):
                deployer.find_available_port()
