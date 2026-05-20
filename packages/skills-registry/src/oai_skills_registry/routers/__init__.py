"""
Skills Registry routers.

Sub-routers
-----------
skills    — catalog (list / get / register) + root + health + template download
git       — Git/GitHub operations (discover, import, refresh, preview, bulk-register)
lifecycle — skill lifecycle actions (publish, upgrade, downgrade, deprecate, history)

To add a new router (e.g. token management) create ``routers/token.py`` and
include it below::

    from oai_skills_registry.routers.token import router as token_router
    router.include_router(token_router)
"""

from fastapi import APIRouter

from oai_skills_registry.routers.git import router as git_router
from oai_skills_registry.routers.lifecycle import router as lifecycle_router
from oai_skills_registry.routers.skills import router as skills_router

router = APIRouter()

# Order matters: static paths (e.g. /skills/template/download,
# /skills/preview-versions) must be registered before parameterised paths
# (e.g. /skills/{skill_name}).  skills_router and git_router both contain
# static /skills/... paths, so they are included before lifecycle_router.
router.include_router(skills_router)
router.include_router(git_router)
router.include_router(lifecycle_router)

__all__ = ["router"]
