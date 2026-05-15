"""
OpenAI Skills Registry
Manages skill lifecycle with Git integration.
"""

__version__ = "1.0.0"
__author__ = "OpenAI"

from oai_skills_registry.services.skills_registry import SkillsRegistry
from oai_skills_registry.services.db.database_logger import SkillsDatabaseLogger
from oai_skills_registry.dependencies import initialize_registry, get_registry

__all__ = [
    "SkillsRegistry",
    "SkillsDatabaseLogger",
    "initialize_registry",
    "get_registry",
]
