import pytest

import oai_agent_core.utils.deployment as deployment


def test_default_no_target(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_TARGET", raising=False)
    monkeypatch.delenv("DISABLE_RUNTIME_PACKAGE_INSTALL", raising=False)
    assert deployment.get_deployment_target() == ""
    assert deployment.is_agentcore_runtime() is False
    assert deployment.is_runtime_package_install_disabled() is False


def test_agentcore_target_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_TARGET", "AgentCore")
    assert deployment.get_deployment_target() == "agentcore"
    assert deployment.is_agentcore_runtime() is True
    # AgentCore implies runtime installs are disabled.
    assert deployment.is_runtime_package_install_disabled() is True


def test_disable_flag_without_agentcore(monkeypatch):
    monkeypatch.delenv("DEPLOYMENT_TARGET", raising=False)
    monkeypatch.setenv("DISABLE_RUNTIME_PACKAGE_INSTALL", "true")
    assert deployment.is_agentcore_runtime() is False
    assert deployment.is_runtime_package_install_disabled() is True


def test_unrelated_target_does_not_disable_installs(monkeypatch):
    monkeypatch.setenv("DEPLOYMENT_TARGET", "kubernetes")
    monkeypatch.delenv("DISABLE_RUNTIME_PACKAGE_INSTALL", raising=False)
    assert deployment.is_agentcore_runtime() is False
    assert deployment.is_runtime_package_install_disabled() is False


@pytest.mark.parametrize(
    "val,expected",
    [("1", True), ("true", True), ("YES", True), ("on", True),
     ("0", False), ("false", False), ("", False)],
)
def test_flag_truthiness(monkeypatch, val, expected):
    monkeypatch.delenv("DEPLOYMENT_TARGET", raising=False)
    monkeypatch.setenv("DISABLE_RUNTIME_PACKAGE_INSTALL", val)
    assert deployment.is_runtime_package_install_disabled() is expected
