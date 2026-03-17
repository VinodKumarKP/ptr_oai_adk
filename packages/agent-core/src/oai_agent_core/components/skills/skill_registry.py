import logging
from pathlib import Path
from typing import Optional, List, Dict

from .errors import ParseError, ValidationError
from .models import SkillProperties
from .parser import load_metadata, find_skill_md
from .utils import resolve_path, is_safe_path


class SkillRegistry:
    """
    A registry for discovering, loading, and managing agent skills.

    This class scans a designated directory to find all available skills,
    parses their metadata, and provides an interface to access them.
    """

    def __init__(self, logger: Optional[logging.Logger] = None, project_root: Optional[str] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.project_root = project_root
        self.skills: Dict[str, SkillProperties] = {}

    def discover_skills(self, skills_dir: str) -> None:
        """
        Discovers all skills in a directory and populates the registry.

        Scans the specified directory for subdirectories, each representing a skill,
        and loads metadata from each skill's SKILL.md file.

        Args:
            skills_dir: The path to the main skills directory.
        """
        if not skills_dir:
            self.logger.info("No skills directory provided. Skipping skill discovery.")
            return

        resolved_skills_dir = resolve_path(skills_dir, self.project_root)

        if not resolved_skills_dir.exists() or not resolved_skills_dir.is_dir():
            self.logger.warning(f"Skills directory not found or not a directory: {resolved_skills_dir}")
            return

        self.logger.info(f"Discovering skills in: {resolved_skills_dir}")

        for skill_dir in resolved_skills_dir.iterdir():
            if not skill_dir.is_dir():
                continue

            if not is_safe_path(skill_dir, resolved_skills_dir):
                self.logger.warning(f"Skipping potentially unsafe skill path: {skill_dir}")
                continue

            try:
                skill_md_path = find_skill_md(skill_dir)
                if not skill_md_path:
                    self.logger.debug(f"No SKILL.md found in {skill_dir}, skipping.")
                    continue

                skill_properties = load_metadata(skill_dir)
                self.skills[skill_properties.name] = skill_properties
                self.logger.debug(f"Successfully discovered skill: {skill_properties.name}")

            except (ParseError, ValidationError) as e:
                self.logger.warning(f"Skipping invalid skill in {skill_dir}: {e}")
            except Exception as e:
                self.logger.error(f"An unexpected error occurred while parsing skill in {skill_dir}: {e}")

        self.logger.info(f"Discovery complete. Found {len(self.skills)} skills.")

    def get_skill(self, name: str) -> Optional[SkillProperties]:
        """
        Retrieves a skill by its name.

        Args:
            name: The name of the skill to retrieve.

        Returns:
            A SkillProperties object if the skill is found, otherwise None.
        """
        return self.skills.get(name)

    def get_skills(self, skill_names: List[str]) -> List[SkillProperties]:
        """
        Retrieves a list of skills from a list of names.

        Args:
            skill_names: A list of skill names to retrieve.

        Returns:
            A list of SkillProperties objects for the found skills.
        """
        return [self.skills[name] for name in skill_names if name in self.skills]

    def get_all_skills(self) -> List[SkillProperties]:
        """
        Retrieves all discovered skills.

        Returns:
            A list of all SkillProperties objects in the registry, sorted by name.
        """
        return sorted(self.skills.values(), key=lambda s: s.name)

    def generate_skills_prompt(self, skills: List[SkillProperties]) -> str:
        """
        Generates the <available_skills> XML block for the main system prompt.
        This creates a concise list of available skills (Phase 1 of Progressive Disclosure) so the agent knows what it can do.
        Params:
        skills – A list of discovered SkillProperties objects.
        Returns:
        A string containing the formatted <available_skills> block or an empty string if no skills are provided.
        """
        from oai_agent_core.components.skills.prompt import generate_skills_prompt
        return generate_skills_prompt(skills)