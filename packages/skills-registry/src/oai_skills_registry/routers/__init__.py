"""
Skills Registry routers.

Sub-routers
-----------
skills    — catalog (list / get / register) + root + health + template download
git       — Git/GitHub operations (discover, import, refresh, preview, bulk-register)
lifecycle — skill lifecycle actions (publish, upgrade, downgrade, deprecate, history)
token     — API token management (generate, list, revoke, revoke-all)

To add another router, create the file and include it below.
"""

from fastapi import APIRouter

from oai_skills_registry.routers.git import router as git_router
from oai_skills_registry.routers.lifecycle import router as lifecycle_router
from oai_skills_registry.routers.skills import router as skills_router
from oai_skills_registry.routers.token import router as token_router

router = APIRouter()

# Order matters: static paths (e.g. /skills/template/download,
# /skills/preview-versions) must be registered before parameterised paths
# (e.g. /skills/{skill_name}).  skills_router and git_router both contain
# static /skills/... paths, so they are included before lifecycle_router.
router.include_router(skills_router)
router.include_router(git_router)
router.include_router(lifecycle_router)
router.include_router(token_router)

__all__ = ["router"]
