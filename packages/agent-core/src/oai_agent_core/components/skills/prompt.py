"""Generate skills prompt for agent system prompts

This module provides XML-formatted prompt generation following the AgentSkills.io specification.
"""
from typing import List
from oai_agent_core.components.skills.models import SkillProperties

# This is the core instruction set for how an agent should understand and use skills.
# It's designed to be concise and clear, focusing on the progressive disclosure pattern.
SKILLS_SYSTEM_PROMPT = """
## Skills Library

You have access to a library of skills that provide specialized capabilities. Use them to break down complex tasks into manageable steps.

<skills_instructions>
**How to Use Skills:**

1.  **Recognize Need:** When a user's request matches a skill's description, decide to use that skill.
2.  **Read Instructions:** Before acting, read the skill's full instructions to understand the workflow. Use the `read_skill_instructions` tool with the skill's name (e.g., `read_skill_instructions(skill_name="web-research")`).
3.  **Follow Workflow:** Execute the steps exactly as described in the skill's instructions. This may involve using other tools or accessing resource files.
4.  **Access Resources:** If a skill mentions supporting files (e.g., Python scripts, reference docs), use their full paths to read them.

Always check the `<available_skills>` list to see which skills you can use.
</skills_instructions>

<available_skills>
{skills_list}
</available_skills>
"""

# This prompt is used to present the detailed instructions of a specific skill to the agent
# after it has decided to use it.
SKILL_INSTRUCTIONS_PROMPT = """
## Skill Instructions: {skill_name}

You have activated the "{skill_name}" skill. Follow these instructions carefully to complete the task.

<instructions>
{instructions}
</instructions>
"""


def generate_skills_prompt(skills: List[SkillProperties]) -> str:
    """
    Generates the <available_skills> XML block for the main system prompt.

    This creates a concise list of available skills (Phase 1 of Progressive Disclosure)
    so the agent knows what it can do.

    Args:
        skills: A list of discovered SkillProperties objects.

    Returns:
        A string containing the formatted <available_skills> block or an empty string if no skills are provided.
    """
    if not skills:
        return ""

    # Build the XML string for each skill, including only essential metadata.
    skill_elements = []
    for skill in sorted(skills, key=lambda s: s.name):
        skill_xml = (
            "  <skill>\n"
            f"    <name>{skill.name}</name>\n"
            f"    <description>{skill.description}</description>\n"
            f"    <path>{skill.path}</path>\n"
            "  </skill>"
        )
        skill_elements.append(skill_xml)

    skills_list_str = "\n".join(skill_elements)

    # Inject the list into the main skills prompt template.
    return SKILLS_SYSTEM_PROMPT.format(skills_list=skills_list_str)


def generate_skill_instructions_prompt(skill_name: str, instructions: str) -> str:
    """
    Generates the prompt that shows the detailed instructions for a specific skill.

    Args:
        skill_name: The name of the skill being activated.
        instructions: The full markdown content from the skill's SKILL.md file.

    Returns:
        A formatted prompt containing the skill's detailed instructions.
    """
    return SKILL_INSTRUCTIONS_PROMPT.format(
        skill_name=skill_name,
        instructions=instructions
    )


__all__ = ["generate_skills_prompt", "generate_skill_instructions_prompt"]
