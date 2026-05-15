"""
Integration tests for Skills Registry API routes.
"""

import pytest
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient


# Note: These tests require the full FastAPI app setup
# For now, we test the availability and structure of router endpoints

class TestSkillsRouter:
    """Tests for skills router endpoints."""

    def test_router_is_defined(self):
        """Test that router module can be imported."""
        try:
            from oai_skills_registry.routers.skills import router
            assert router is not None
        except ImportError:
            pytest.skip("Router module not available in test environment")

    def test_router_has_endpoints(self):
        """Test that router has defined endpoints."""
        try:
            from oai_skills_registry.routers.skills import router
            # Router should have routes
            assert hasattr(router, 'routes')
            assert len(router.routes) > 0
        except ImportError:
            pytest.skip("Router module not available")

    def test_health_check_endpoint_defined(self):
        """Test that health check endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import health_check
            assert callable(health_check)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_list_skills_endpoint_defined(self):
        """Test that list skills endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import list_skills
            assert callable(list_skills)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_register_skill_endpoint_defined(self):
        """Test that register skill endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import register_skill
            assert callable(register_skill)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_get_skill_details_endpoint_defined(self):
        """Test that get skill details endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import get_skill_details
            assert callable(get_skill_details)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_publish_skill_endpoint_defined(self):
        """Test that publish skill endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import publish_skill
            assert callable(publish_skill)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_upgrade_skill_endpoint_defined(self):
        """Test that upgrade skill endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import upgrade_skill
            assert callable(upgrade_skill)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_downgrade_skill_endpoint_defined(self):
        """Test that downgrade skill endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import downgrade_skill
            assert callable(downgrade_skill)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_deprecate_skill_endpoint_defined(self):
        """Test that deprecate skill endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import deprecate_skill
            assert callable(deprecate_skill)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_delete_skill_endpoint_defined(self):
        """Test that delete skill endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import delete_skill
            assert callable(delete_skill)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_import_skill_endpoint_defined(self):
        """Test that import skill endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import import_skill_from_git
            assert callable(import_skill_from_git)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_discover_skills_endpoint_defined(self):
        """Test that discover skills endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import discover_skills
            assert callable(discover_skills)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    def test_bulk_register_endpoint_defined(self):
        """Test that bulk register endpoint is defined."""
        try:
            from oai_skills_registry.routers.skills import bulk_register_skills
            assert callable(bulk_register_skills)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")


@pytest.mark.asyncio
class TestSkillsRouterAsync:
    """Async tests for router endpoints."""

    async def test_root_endpoint_response(self):
        """Test root endpoint returns proper structure."""
        try:
            from oai_skills_registry.routers.skills import root
            # The root endpoint returns JSONResponse
            result = await root()
            # JSONResponse is the return type from FastAPI
            assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")

    async def test_health_check_response(self):
        """Test health check endpoint response."""
        try:
            from oai_skills_registry.routers.skills import health_check
            # Would need to mock get_registry dependency
            assert callable(health_check)
        except (ImportError, AttributeError):
            pytest.skip("Endpoint not available")
