"""
Tests for oai_skills_registry.routers.token

Covers: generate, list, revoke, revoke-all endpoints.
Auth is bypassed via SKILLS_AUTH_ENABLED=false.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from oai_skills_registry.routers.token import router


# ---------------------------------------------------------------------------
# App fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    """Minimal FastAPI app that mounts only the token router."""
    _app = FastAPI()
    _app.include_router(router)
    return _app


@pytest.fixture()
def client(app, monkeypatch):
    """TestClient with auth disabled."""
    monkeypatch.setenv("SKILLS_AUTH_ENABLED", "false")
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def mock_tm():
    """A mock TokenManager returned by _get_token_manager()."""
    tm = MagicMock()
    with patch(
        "oai_skills_registry.routers.token._get_token_manager",
        return_value=tm,
    ):
        yield tm


# ---------------------------------------------------------------------------
# POST /tokens/generate
# ---------------------------------------------------------------------------

class TestGenerateToken:
    def test_generate_returns_token(self, client, mock_tm):
        mock_tm.generate_token.return_value = "abc123.meta"

        resp = client.post("/tokens/generate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["token"] == "abc123.meta"
        assert data["ttl_seconds"] == 3600          # default

    def test_generate_with_custom_ttl(self, client, mock_tm):
        mock_tm.generate_token.return_value = "tok.xyz"

        resp = client.post("/tokens/generate?ttl_seconds=86400")
        assert resp.status_code == 200
        assert resp.json()["ttl_seconds"] == 86400
        # max_tokens is now forwarded to enforce the per-server token limit
        mock_tm.generate_token.assert_called_once_with(
            server_key="skills-registry",
            user_id=None,
            role_id=None,
            ttl_seconds=86400,
            max_tokens=10,
        )

    def test_generate_permanent_token(self, client, mock_tm):
        """ttl_seconds=-1 should be translated to ttl_seconds=None (permanent)."""
        mock_tm.generate_token.return_value = "perm.tok"

        resp = client.post("/tokens/generate?ttl_seconds=-1")
        assert resp.status_code == 200
        # max_tokens is now forwarded to enforce the per-server token limit
        mock_tm.generate_token.assert_called_once_with(
            server_key="skills-registry",
            user_id=None,
            role_id=None,
            ttl_seconds=None,
            max_tokens=10,
        )

    def test_generate_with_user_and_role(self, client, mock_tm):
        mock_tm.generate_token.return_value = "u.tok"

        resp = client.post("/tokens/generate?user_id=alice&role_id=admin&ttl_seconds=3600")
        assert resp.status_code == 200
        data = resp.json()
        assert data["user_id"] == "alice"
        assert data["role_id"] == "admin"

    def test_generate_429_on_value_error(self, client, mock_tm):
        mock_tm.generate_token.side_effect = ValueError("Token limit reached")

        resp = client.post("/tokens/generate")
        assert resp.status_code == 429
        assert "Token limit" in resp.json()["detail"]

    def test_generate_500_on_exception(self, client, mock_tm):
        mock_tm.generate_token.side_effect = RuntimeError("redis down")

        resp = client.post("/tokens/generate")
        assert resp.status_code == 500
        assert "Token generation failed" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# GET /tokens
# ---------------------------------------------------------------------------

class TestListTokens:
    def test_list_returns_tokens(self, client, mock_tm):
        mock_tm.get_all_tokens.return_value = [
            {"token": "t1", "type": "ttl", "expires_in": 3600},
            {"token": "t2", "type": "permanent"},
        ]

        resp = client.get("/tokens")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 2
        assert len(data["tokens"]) == 2

    def test_list_empty(self, client, mock_tm):
        mock_tm.get_all_tokens.return_value = []

        resp = client.get("/tokens")
        assert resp.status_code == 200
        data = resp.json()
        # Core fields
        assert data["tokens"] == []
        assert data["total"] == 0
        # Quota fields added alongside the token list
        assert data["max_tokens"] == 10
        assert data["can_generate"] is True

    def test_list_passes_include_expired(self, client, mock_tm):
        mock_tm.get_all_tokens.return_value = []

        client.get("/tokens?include_expired=true")
        mock_tm.get_all_tokens.assert_called_once_with(
            "skills-registry", include_expired=True
        )

    def test_list_default_excludes_expired(self, client, mock_tm):
        mock_tm.get_all_tokens.return_value = []

        client.get("/tokens")
        mock_tm.get_all_tokens.assert_called_once_with(
            "skills-registry", include_expired=False
        )

    def test_list_500_on_exception(self, client, mock_tm):
        mock_tm.get_all_tokens.side_effect = RuntimeError("redis error")

        resp = client.get("/tokens")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /tokens  (revoke all)
# ---------------------------------------------------------------------------

class TestRevokeAllTokens:
    def test_revoke_all_returns_count(self, client, mock_tm):
        mock_tm.revoke_all_tokens.return_value = 3

        resp = client.delete("/tokens")
        assert resp.status_code == 200
        data = resp.json()
        assert data["revoked"] == 3
        assert "3" in data["message"]

    def test_revoke_all_zero(self, client, mock_tm):
        mock_tm.revoke_all_tokens.return_value = 0

        resp = client.delete("/tokens")
        assert resp.status_code == 200
        assert resp.json()["revoked"] == 0

    def test_revoke_all_uses_correct_server_key(self, client, mock_tm):
        mock_tm.revoke_all_tokens.return_value = 1

        client.delete("/tokens")
        mock_tm.revoke_all_tokens.assert_called_once_with("skills-registry")

    def test_revoke_all_500_on_exception(self, client, mock_tm):
        mock_tm.revoke_all_tokens.side_effect = RuntimeError("boom")

        resp = client.delete("/tokens")
        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# DELETE /tokens/{token}  (revoke single)
# ---------------------------------------------------------------------------

class TestRevokeToken:
    def test_revoke_existing_token(self, client, mock_tm):
        mock_tm.revoke_token.return_value = True

        resp = client.delete("/tokens/mytoken123")
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True

    def test_revoke_missing_token_returns_404(self, client, mock_tm):
        mock_tm.revoke_token.return_value = False

        resp = client.delete("/tokens/ghost-token")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_revoke_500_on_exception(self, client, mock_tm):
        mock_tm.revoke_token.side_effect = RuntimeError("redis gone")

        resp = client.delete("/tokens/some-token")
        assert resp.status_code == 500

    def test_revoke_passes_token_string(self, client, mock_tm):
        mock_tm.revoke_token.return_value = True

        client.delete("/tokens/abc.def")
        mock_tm.revoke_token.assert_called_once_with("abc.def")


# ---------------------------------------------------------------------------
# Router structure
# ---------------------------------------------------------------------------

class TestTokenRouterStructure:
    def _collect_app_paths(self):
        """Extract all registered paths from the main FastAPI app."""
        from oai_skills_registry.main import app
        paths = set()
        for route in app.routes:
            if hasattr(route, 'path'):
                p = route.path
                if p.startswith("/api/v1/skills-registry"):
                    p = p[len("/api/v1/skills-registry"):]
                    if not p:
                        p = "/"
                paths.add(p)
        return paths

    def _collect_paths(self, rtr):
        paths = set()
        for entry in rtr.routes:
            if hasattr(entry, 'path') and isinstance(entry.path, str):
                paths.add(entry.path)
            if hasattr(entry, 'routes'):
                paths.update(self._collect_paths(entry))
        return paths

    def test_router_importable(self):
        from oai_skills_registry.routers.token import router as r
        assert r is not None

    def test_all_routes_present(self):
        from oai_skills_registry.routers.token import router as r
        paths = self._collect_paths(r)
        assert "/tokens/generate" in paths
        assert "/tokens" in paths
        assert "/tokens/{token}" in paths

    def test_token_router_included_in_composed_router(self):
        paths = self._collect_app_paths()
        assert "/tokens/generate" in paths
        assert "/tokens" in paths
        assert "/tokens/{token}" in paths
