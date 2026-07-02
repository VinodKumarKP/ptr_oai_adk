"""
Structural tests for the Skills Registry router modules.

Each router module is tested independently so that failures are
immediately attributed to the right file.
"""

import pytest


# ---------------------------------------------------------------------------
# skills router
# ---------------------------------------------------------------------------

class TestSkillsRouter:
    """Tests for oai_skills_registry.routers.skills"""

    def test_router_is_defined(self):
        from oai_skills_registry.routers.skills import router
        assert router is not None
        assert hasattr(router, "routes")
        assert len(router.routes) > 0

    def test_root_endpoint_defined(self):
        from oai_skills_registry.routers.skills import root
        assert callable(root)

    def test_health_check_endpoint_defined(self):
        from oai_skills_registry.routers.skills import health_check
        assert callable(health_check)

    def test_list_skills_endpoint_defined(self):
        from oai_skills_registry.routers.skills import list_skills
        assert callable(list_skills)

    def test_register_skill_endpoint_defined(self):
        from oai_skills_registry.routers.skills import register_skill
        assert callable(register_skill)

    def test_get_skill_details_endpoint_defined(self):
        from oai_skills_registry.routers.skills import get_skill_details
        assert callable(get_skill_details)

    def test_get_published_skill_endpoint_defined(self):
        from oai_skills_registry.routers.skills import get_published_skill
        assert callable(get_published_skill)

    def test_download_template_endpoint_defined(self):
        from oai_skills_registry.routers.skills import download_skill_template
        assert callable(download_skill_template)


# ---------------------------------------------------------------------------
# git router
# ---------------------------------------------------------------------------

class TestGitRouter:
    """Tests for oai_skills_registry.routers.git"""

    def test_router_is_defined(self):
        from oai_skills_registry.routers.git import router
        assert router is not None
        assert hasattr(router, "routes")
        assert len(router.routes) > 0

    def test_discover_skills_endpoint_defined(self):
        from oai_skills_registry.routers.git import discover_skills
        assert callable(discover_skills)

    def test_preview_versions_endpoint_defined(self):
        from oai_skills_registry.routers.git import preview_skill_versions
        assert callable(preview_skill_versions)

    def test_register_bulk_endpoint_defined(self):
        from oai_skills_registry.routers.git import register_skills_bulk
        assert callable(register_skills_bulk)

    def test_import_from_git_endpoint_defined(self):
        from oai_skills_registry.routers.git import import_skill_from_git
        assert callable(import_skill_from_git)

    def test_refresh_versions_endpoint_defined(self):
        from oai_skills_registry.routers.git import refresh_skill_versions
        assert callable(refresh_skill_versions)


# ---------------------------------------------------------------------------
# lifecycle router
# ---------------------------------------------------------------------------

class TestLifecycleRouter:
    """Tests for oai_skills_registry.routers.lifecycle"""

    def test_router_is_defined(self):
        from oai_skills_registry.routers.lifecycle import router
        assert router is not None
        assert hasattr(router, "routes")
        assert len(router.routes) > 0

    def test_publish_skill_endpoint_defined(self):
        from oai_skills_registry.routers.lifecycle import publish_skill
        assert callable(publish_skill)

    def test_upgrade_skill_endpoint_defined(self):
        from oai_skills_registry.routers.lifecycle import upgrade_skill
        assert callable(upgrade_skill)

    def test_downgrade_skill_endpoint_defined(self):
        from oai_skills_registry.routers.lifecycle import downgrade_skill
        assert callable(downgrade_skill)

    def test_deprecate_skill_endpoint_defined(self):
        from oai_skills_registry.routers.lifecycle import deprecate_skill
        assert callable(deprecate_skill)

    def test_get_skill_history_endpoint_defined(self):
        from oai_skills_registry.routers.lifecycle import get_skill_history
        assert callable(get_skill_history)


# ---------------------------------------------------------------------------
# composed router (__init__)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Helper: collect route paths regardless of FastAPI version
# ---------------------------------------------------------------------------

def _collect_subrouter_paths(rtr):
    """Recursively collect paths from a sub-router."""
    paths = set()
    for entry in rtr.routes:
        if hasattr(entry, 'path') and isinstance(entry.path, str):
            paths.add(entry.path)
        if hasattr(entry, 'routes'):
            paths.update(_collect_subrouter_paths(entry))
    return paths

class TestComposedRouter:
    """Tests for oai_skills_registry.routers (composed router)."""

    def test_composed_router_importable(self):
        from oai_skills_registry.routers import router
        assert router is not None

    def test_composed_router_includes_all_routes(self):
        from oai_skills_registry.main import app
        from fastapi.testclient import TestClient
        from oai_skills_registry.routers.git import router as git_r
        from oai_skills_registry.routers.lifecycle import router as lc_r
        from oai_skills_registry.routers.skills import router as skills_r

        client = TestClient(app)
        for sub_router in (skills_r, git_r, lc_r):
            sub_paths = _collect_subrouter_paths(sub_router)
            for route_path in sub_paths:
                test_path = f"/api/v1/skills-registry{route_path}"
                test_path = test_path.replace("{skill_name}", "test_skill")
                
                # OPTIONS returns 405 (Method Not Allowed) or 200 (OK) if route exists, 404 if it doesn't
                response = client.options(test_path)
                assert response.status_code != 404, (
                    f"Route {route_path!r} (tested as {test_path!r}) from {sub_router} not found in main app"
                )

    def test_all_lifecycle_routes_present(self):
        from oai_skills_registry.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        for expected in (
            "/api/v1/skills-registry/skills/test_skill/publish",
            "/api/v1/skills-registry/skills/test_skill/upgrade",
            "/api/v1/skills-registry/skills/test_skill/downgrade",
            "/api/v1/skills-registry/skills/test_skill/deprecate",
            "/api/v1/skills-registry/skills/test_skill/history",
        ):
            assert client.options(expected).status_code != 404, f"Missing lifecycle route: {expected}"

    def test_all_git_routes_present(self):
        from oai_skills_registry.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        for expected in (
            "/api/v1/skills-registry/skills/discover",
            "/api/v1/skills-registry/skills/preview-versions",
            "/api/v1/skills-registry/skills/register-bulk",
            "/api/v1/skills-registry/skills/test_skill/import-from-git",
            "/api/v1/skills-registry/skills/test_skill/refresh-versions",
        ):
            assert client.options(expected).status_code != 404, f"Missing git route: {expected}"

    def test_all_catalog_routes_present(self):
        from oai_skills_registry.main import app
        from fastapi.testclient import TestClient
        client = TestClient(app)
        for expected in (
            "/api/v1/skills-registry/",
            "/api/v1/skills-registry/health",
            "/api/v1/skills-registry/skills",
            "/api/v1/skills-registry/skills/test_skill",
            "/api/v1/skills-registry/skills/test_skill/published",
            "/api/v1/skills-registry/skills/template/download",
        ):
            assert client.options(expected).status_code != 404, f"Missing catalog route: {expected}"


# ---------------------------------------------------------------------------
# Async smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestSkillsRouterAsync:
    """Async smoke tests for endpoint callability."""

    async def test_root_endpoint_returns_response(self):
        from oai_skills_registry.routers.skills import root
        result = await root()
        assert result is not None

    async def test_health_check_is_callable(self):
        from oai_skills_registry.routers.skills import health_check
        assert callable(health_check)
