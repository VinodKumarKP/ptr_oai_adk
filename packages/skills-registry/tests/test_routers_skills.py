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

class TestComposedRouter:
    """Tests for oai_skills_registry.routers (composed router)."""

    def test_composed_router_importable(self):
        from oai_skills_registry.routers import router
        assert router is not None

    def test_composed_router_includes_all_routes(self):
        from oai_skills_registry.routers import router
        from oai_skills_registry.routers.git import router as git_r
        from oai_skills_registry.routers.lifecycle import router as lc_r
        from oai_skills_registry.routers.skills import router as skills_r

        composed_paths = {r.path for r in router.routes}
        for sub_router in (skills_r, git_r, lc_r):
            for route in sub_router.routes:
                assert route.path in composed_paths, (
                    f"Route {route.path!r} from {sub_router} not found in composed router"
                )

    def test_all_lifecycle_routes_present(self):
        from oai_skills_registry.routers import router
        paths = {r.path for r in router.routes}
        for expected in (
            "/skills/{skill_name}/publish",
            "/skills/{skill_name}/upgrade",
            "/skills/{skill_name}/downgrade",
            "/skills/{skill_name}/deprecate",
            "/skills/{skill_name}/history",
        ):
            assert expected in paths, f"Missing lifecycle route: {expected}"

    def test_all_git_routes_present(self):
        from oai_skills_registry.routers import router
        paths = {r.path for r in router.routes}
        for expected in (
            "/skills/discover",
            "/skills/preview-versions",
            "/skills/register-bulk",
            "/skills/{skill_name}/import-from-git",
            "/skills/{skill_name}/refresh-versions",
        ):
            assert expected in paths, f"Missing git route: {expected}"

    def test_all_catalog_routes_present(self):
        from oai_skills_registry.routers import router
        paths = {r.path for r in router.routes}
        for expected in (
            "/",
            "/health",
            "/skills",
            "/skills/{skill_name}",
            "/skills/{skill_name}/published",
            "/skills/template/download",
        ):
            assert expected in paths, f"Missing catalog route: {expected}"


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
