"""Tests for centralized settings, route re-registration and startup phases."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from oai_agent_server.config import (
    Config, ServerSettings, env_csv, env_flag, env_int, env_str,
)
from oai_agent_server.main import AgentHTTPServer


@pytest.fixture
def server(monkeypatch):
    """A server built with external collaborators stubbed out."""
    monkeypatch.delenv("ALWAYS_ACTIVE_MODES", raising=False)
    monkeypatch.delenv("AGENT_SERVER_PROFILE", raising=False)
    agent = MagicMock()
    agent.agent_name = "test_agent"
    with patch("oai_agent_server.main.get_logger"), \
         patch("oai_agent_server.main.ConfigManager"), \
         patch("oai_agent_server.main.DatabaseLogger"):
        return AgentHTTPServer(agent, "test_agent")


# ---------------------------------------------------------------------------
# Env parsing helpers
# ---------------------------------------------------------------------------

class TestEnvHelpers:
    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("TRUE", True), ("1", True), ("yes", True), ("on", True),
        ("false", False), ("FALSE", False), ("0", False), ("no", False), ("", False),
    ])
    def test_env_flag_parsing(self, monkeypatch, value, expected):
        monkeypatch.setenv("X_FLAG", value)
        # An empty value falls back to the default, which is False here.
        assert env_flag("X_FLAG") is expected

    def test_env_flag_default_used_when_unset(self, monkeypatch):
        monkeypatch.delenv("X_FLAG", raising=False)
        assert env_flag("X_FLAG", True) is True
        assert env_flag("X_FLAG", False) is False

    def test_env_int_falls_back_on_garbage(self, monkeypatch):
        monkeypatch.setenv("X_INT", "not-a-number")
        assert env_int("X_INT", 7) == 7

    def test_env_csv_distinguishes_unset_from_empty(self, monkeypatch):
        monkeypatch.delenv("X_CSV", raising=False)
        assert env_csv("X_CSV") is None
        monkeypatch.setenv("X_CSV", "")
        assert env_csv("X_CSV") == []
        monkeypatch.setenv("X_CSV", " a , b ")
        assert env_csv("X_CSV") == ["a", "b"]

    def test_env_str_default(self, monkeypatch):
        monkeypatch.delenv("X_STR", raising=False)
        assert env_str("X_STR", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Settings read live, so timing semantics are preserved
# ---------------------------------------------------------------------------

class TestServerSettings:
    def test_values_reflect_later_env_changes(self, monkeypatch):
        settings = ServerSettings()
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert settings.log_level == "DEBUG"
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        assert settings.log_level == "WARNING"

    def test_lite_profile_modes(self, monkeypatch):
        monkeypatch.delenv("ALWAYS_ACTIVE_MODES", raising=False)
        monkeypatch.setenv("AGENT_SERVER_PROFILE", "lite")
        settings = ServerSettings()
        assert settings.is_lite is True
        assert settings.resolve_modes() == ServerSettings.LITE_MODES
        assert "monitoring" not in settings.resolve_modes()

    def test_explicit_modes_override_profile(self, monkeypatch):
        monkeypatch.setenv("AGENT_SERVER_PROFILE", "lite")
        monkeypatch.setenv("ALWAYS_ACTIVE_MODES", "health,chat")
        assert ServerSettings().resolve_modes() == {"health", "chat"}

    def test_lite_profile_implies_slim_startup(self, monkeypatch):
        monkeypatch.delenv("SLIM_STARTUP", raising=False)
        monkeypatch.setenv("AGENT_SERVER_PROFILE", "lite")
        assert ServerSettings().slim_startup is True

    def test_judge_default_follows_monitoring_mode(self, monkeypatch):
        monkeypatch.delenv("LLM_JUDGE_ENABLED", raising=False)
        settings = ServerSettings()
        assert settings.llm_judge_enabled(monitoring_active=True) is True
        assert settings.llm_judge_enabled(monitoring_active=False) is False

    def test_judge_flag_can_disable_when_monitoring_active(self, monkeypatch):
        monkeypatch.setenv("LLM_JUDGE_ENABLED", "false")
        assert ServerSettings().llm_judge_enabled(monitoring_active=True) is False

    def test_cors_wildcard_and_list(self, monkeypatch):
        monkeypatch.setenv("ALLOWED_ORIGINS", "*")
        s = ServerSettings()
        assert s.cors_allow_any_origin is True and s.cors_origins() == ["*"]
        monkeypatch.setenv("ALLOWED_ORIGINS", "https://a.com, https://b.com")
        assert ServerSettings().cors_origins() == ["https://a.com", "https://b.com"]
        monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
        assert ServerSettings().cors_origins() == ["http://localhost:3000"]

    def test_registry_url_prefers_explicit_registry(self, monkeypatch):
        monkeypatch.setenv("AGENT_BASE_URL", "http://base")
        monkeypatch.setenv("AGENT_REGISTRY_URL", "http://registry")
        assert ServerSettings().registry_url == "http://registry"
        monkeypatch.delenv("AGENT_REGISTRY_URL")
        assert ServerSettings().registry_url == "http://base"

    def test_config_force_auth_matches_auth_path_default(self, monkeypatch):
        """FORCE_AUTH defaults to auth-required, as security/dependencies does."""
        monkeypatch.delenv("FORCE_AUTH", raising=False)
        assert Config().force_auth is True
        monkeypatch.setenv("FORCE_AUTH", "false")
        assert Config().force_auth is False


# ---------------------------------------------------------------------------
# Route re-registration must not duplicate
# ---------------------------------------------------------------------------

class TestRouteReRegistration:
    @staticmethod
    def _endpoints(server):
        # Keyed by path *and* method: one path legitimately appears more than
        # once (e.g. GET and DELETE on /admin/tasks/{task_id}).
        return [
            (getattr(r, "path", None), frozenset(getattr(r, "methods", None) or []))
            for r in server.app.router.routes
        ]

    def test_set_allowed_modes_does_not_duplicate_routes(self, server):
        before = self._endpoints(server)
        assert len(before) == len(set(before)), "baseline already contains duplicates"

        server.set_allowed_modes(["chat"])
        after = self._endpoints(server)

        assert len(after) == len(set(after)), "re-registration duplicated routes"
        assert set(before) == set(after)

    def test_repeated_registration_is_stable(self, server):
        counts = []
        for _ in range(3):
            server.set_allowed_modes(None)
            counts.append(len(server.app.router.routes))
        assert len(set(counts)) == 1, f"route count drifted: {counts}"

    def test_openapi_schema_regenerated_after_rebuild(self, server):
        server.app.openapi_schema = {"stale": True}
        server.set_allowed_modes(None)
        assert server.app.openapi_schema != {"stale": True}


# ---------------------------------------------------------------------------
# agent_name consistency
# ---------------------------------------------------------------------------

class TestAgentNameConsistency:
    def test_mismatched_name_uses_agent_object_everywhere(self, monkeypatch):
        agent = MagicMock()
        agent.agent_name = "real_name"
        logger = MagicMock()
        with patch("oai_agent_server.main.get_logger", return_value=logger), \
             patch("oai_agent_server.main.ConfigManager"), \
             patch("oai_agent_server.main.DatabaseLogger"):
            server = AgentHTTPServer(agent, "display_name")

        assert server.agent_name == "real_name"
        # Telemetry and the OpenAPI title must agree with the routes/logs.
        assert "real_name" in server.app.title
        assert server.observability_manager.config.SERVICE_NAME == "real_name"
        logger.warning.assert_called()  # mismatch is surfaced, not silent


# ---------------------------------------------------------------------------
# Startup phases: failure attribution
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestStartupPhases:
    async def test_agent_failure_is_fatal_and_stops_startup(self, server):
        server.agent.initialize = AsyncMock(side_effect=RuntimeError("boom"))
        server._startup_database = AsyncMock()
        server._startup_scheduler = AsyncMock()

        await server.startup()

        assert server.server_state.is_agent_ready is False
        server._startup_database.assert_not_awaited()
        server._startup_scheduler.assert_not_awaited()

    async def test_database_failure_is_fatal(self, server):
        server.agent.initialize = AsyncMock()
        server.db_logger.initialize = AsyncMock(side_effect=RuntimeError("db down"))
        server._startup_scheduler = AsyncMock()

        await server.startup()

        assert server.server_state.is_agent_ready is False
        server._startup_scheduler.assert_not_awaited()

    async def test_resilience_failure_does_not_mark_server_unready(self, server):
        """A circuit-breaker problem must not be reported as the agent failing."""
        server.agent.initialize = AsyncMock()
        server.db_logger.initialize = AsyncMock()
        server.circuit_breaker_registry.register = AsyncMock(
            side_effect=RuntimeError("registry exploded")
        )
        server._startup_scheduler = AsyncMock()
        server._startup_registry = AsyncMock()

        await server.startup()

        assert server.server_state.is_agent_ready is True
        # Later phases still run.
        server._startup_scheduler.assert_awaited()

    async def test_registry_failure_does_not_mark_server_unready(self, server):
        server.agent.initialize = AsyncMock()
        server.db_logger.initialize = AsyncMock()
        server._register_with_registry = AsyncMock(
            side_effect=RuntimeError("registry unreachable")
        )

        await server.startup()

        assert server.server_state.is_agent_ready is True

    async def test_slim_startup_skips_warmup_phases(self, server, monkeypatch):
        monkeypatch.setenv("SLIM_STARTUP", "true")
        server.agent.initialize = AsyncMock()
        server.db_logger.initialize = AsyncMock()
        server._startup_judge = AsyncMock()
        server._startup_resilience = AsyncMock()
        server._startup_registry = AsyncMock()
        server._startup_scheduler = AsyncMock()

        await server.startup()

        server._startup_judge.assert_not_awaited()
        server._startup_resilience.assert_not_awaited()
        server._startup_registry.assert_not_awaited()
        # Non-warm-up phases still run.
        server._startup_scheduler.assert_awaited()

    async def test_slim_startup_false_runs_warmup_phases(self, server, monkeypatch):
        monkeypatch.setenv("SLIM_STARTUP", "false")
        monkeypatch.delenv("AGENT_SERVER_PROFILE", raising=False)
        server.agent.initialize = AsyncMock()
        server.db_logger.initialize = AsyncMock()
        server._startup_resilience = AsyncMock()
        server._startup_registry = AsyncMock()
        server._startup_scheduler = AsyncMock()

        await server.startup()

        server._startup_resilience.assert_awaited()
        server._startup_registry.assert_awaited()
