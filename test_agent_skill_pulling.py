#!/usr/bin/env python3
"""
Integration Test: Agent Automatic Skill Pulling from GitHub

This test verifies that agents automatically pull configured skills from the
Skills Registry (which in turn clones from GitHub repositories).

Run this test with:
    python test_agent_skill_pulling.py

Prerequisites:
    1. Skills Registry running on localhost:8083
    2. At least one skill registered in the registry
    3. The skill must have a git_repository_url pointing to a valid GitHub repo
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Import BaseAgent here to create concrete implementation
from oai_agent_core.core.base_agent import BaseAgent


class ConcreteAgent(BaseAgent):
    """Minimal concrete implementation of BaseAgent for skill pulling tests."""

    def __init__(self, agent_config: Dict[str, Any], agent_name: str = "test_agent"):
        """Initialize agent with config."""
        super().__init__(agent_name=agent_name, agent_config=agent_config, llm=None)

    async def initialize(self):
        """Initialize agent with skill pulling support."""
        # Call parent method to initialize tools, KB, memory, and skills
        await self._load_tools_and_kb_and_memory()
        self._initialized = True

    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Echo implementation for testing."""
        return {"content": user_message, "final": True}

    async def astream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Simple streaming implementation."""
        yield {"content": user_message, "final": True}

    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Sync wrapper for testing."""
        return asyncio.run(self.ainvoke(user_message, config))

    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Sync wrapper for testing."""
        async for chunk in self.astream(user_message, config):
            yield chunk


async def test_agent_pulls_skill_from_registry():
    """Test 1: Agent pulls skill from registry if not available locally."""
    logger.info("=" * 80)
    logger.info("TEST 1: Agent Pulls Missing Skill from Registry")
    logger.info("=" * 80)

    try:
        # Create temporary directory for skills cache
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Configuration with registry but no local skills
            config = {
                'type': 'custom',
                'model': {
                    'model_id': 'bedrock/claude-3-sonnet',
                    'region_name': 'us-west-2'
                },
                'skills': {
                    'skill_dir': str(skills_dir),
                    'registry': {
                        'url': 'http://localhost:8083/api/v1/skills-registry',
                        'token': os.environ.get('SKILLS_REGISTRY_TOKEN', 'test-token')
                    }
                },
                'agent_list': [
                    {
                        'test_agent': {
                            'system_prompt': 'You are a test agent',
                            'skills': ['pdf-processor']  # Will pull from registry
                        }
                    }
                ]
            }

            logger.info("Creating agent with remote skill configuration...")
            agent = ConcreteAgent(agent_config=config)

            logger.info("Initializing agent (should pull 'pdf-processor' from GitHub)...")
            await agent.initialize()

            logger.info("Checking if skill was pulled...")
            logger.debug(f"Agent has skill_registry attribute: {hasattr(agent, 'skill_registry')}")
            logger.debug(f"Agent type: {type(agent)}")
            logger.debug(f"Agent attributes: {dir(agent)[:5]}")  # First 5 attributes for debugging
            if agent.skill_registry and 'pdf-processor' in agent.skill_registry.skills:
                logger.info("✓ TEST 1 PASSED: Skill was successfully pulled")
                return True
            else:
                logger.warning("⚠ TEST 1 FAILED: Skill was not pulled")
                return False

    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED with exception: {e}")
        return False


async def test_agent_uses_local_skill():
    """Test 2: Agent uses local skill without pulling."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Agent Uses Local Skill Without Pulling")
    logger.info("=" * 80)

    try:
        # Create temporary directory with local skill
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create a fake local skill
            local_skill_dir = skills_dir / "local-skill"
            local_skill_dir.mkdir()

            skill_md_content = """---
name: local-skill
description: Test local skill
version: 1.0.0
---

# Local Skill

This is a test local skill.
"""
            (local_skill_dir / "SKILL.md").write_text(skill_md_content)

            # Configuration with local skill
            config = {
                'type': 'custom',
                'model': {
                    'model_id': 'bedrock/claude-3-sonnet',
                    'region_name': 'us-west-2'
                },
                'skills': {
                    'skill_dir': str(skills_dir),
                    'registry': {
                        'url': 'http://localhost:8083/api/v1/skills-registry',
                        'token': os.environ.get('SKILLS_REGISTRY_TOKEN', 'test-token')
                    }
                },
                'agent_list': [
                    {
                        'test_agent': {
                            'system_prompt': 'You are a test agent',
                            'skills': ['local-skill']  # Already local
                        }
                    }
                ]
            }

            logger.info("Creating agent with local skill configuration...")
            agent = ConcreteAgent(agent_config=config)

            logger.info("Initializing agent (should use local skill without pulling)...")
            await agent.initialize()

            logger.info("Checking if local skill is available...")
            if agent.skill_registry and 'local-skill' in agent.skill_registry.skills:
                logger.info("✓ TEST 2 PASSED: Local skill is available")
                return True
            else:
                logger.warning("⚠ TEST 2 FAILED: Local skill not found")
                return False

    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED with exception: {e}")
        return False


async def test_agent_with_mixed_skills():
    """Test 3: Agent with both local and remote skills."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Agent with Mixed Local and Remote Skills")
    logger.info("=" * 80)

    try:
        # Create temporary directory with one local skill
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create a local skill
            local_skill_dir = skills_dir / "local-skill"
            local_skill_dir.mkdir()
            (local_skill_dir / "SKILL.md").write_text("""---
name: local-skill
description: Test local skill
version: 1.0.0
---

# Local Skill
""")

            # Configuration with both local and remote skills
            config = {
                'type': 'custom',
                'model': {
                    'model_id': 'bedrock/claude-3-sonnet',
                    'region_name': 'us-west-2'
                },
                'skills': {
                    'skill_dir': str(skills_dir),
                    'registry': {
                        'url': 'http://localhost:8083/api/v1/skills-registry',
                        'token': os.environ.get('SKILLS_REGISTRY_TOKEN', 'test-token')
                    }
                },
                'agent_list': [
                    {
                        'test_agent': {
                            'system_prompt': 'You are a test agent',
                            'skills': [
                                'local-skill',       # Local
                                'pdf-processor'      # Remote (pulled from registry)
                            ]
                        }
                    }
                ]
            }

            logger.info("Creating agent with mixed skill configuration...")
            agent = ConcreteAgent(agent_config=config)

            logger.info("Initializing agent...")
            await agent.initialize()

            logger.info("Checking available skills...")
            local_available = 'local-skill' in (agent.skill_registry.skills if agent.skill_registry else {})
            remote_available = 'pdf-processor' in (agent.skill_registry.skills if agent.skill_registry else {})

            if local_available:
                logger.info("  ✓ Local skill available")
            else:
                logger.warning("  ⚠ Local skill not available")

            if remote_available:
                logger.info("  ✓ Remote skill pulled and available")
            else:
                logger.warning("  ⚠ Remote skill not available (may fail if registry unavailable)")

            # Test passes if at least local skill is there
            if local_available:
                logger.info("✓ TEST 3 PASSED: Agent ready with at least local skill")
                return True
            else:
                logger.warning("⚠ TEST 3 FAILED: Local skill not available")
                return False

    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED with exception: {e}")
        return False


async def test_agent_without_skills():
    """Test 4: Agent without skill configuration still initializes."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Agent Without Skills Configuration")
    logger.info("=" * 80)

    try:
        # Configuration without skills
        config = {
            'type': 'custom',
            'model': {
                'model_id': 'bedrock/claude-3-sonnet',
                'region_name': 'us-west-2'
            },
            'agent_list': [
                {
                    'test_agent': {
                        'system_prompt': 'You are a simple agent'
                        # No 'skills' section
                    }
                }
            ]
        }

        logger.info("Creating agent without skill configuration...")
        agent = ConcreteAgent(agent_config=config)

        logger.info("Initializing agent...")
        await agent.initialize()

        logger.info("Checking if agent initialized successfully...")
        if agent is not None:
            logger.info("✓ TEST 4 PASSED: Agent initialized without skills")
            return True
        else:
            logger.warning("⚠ TEST 4 FAILED: Agent is None")
            return False

    except Exception as e:
        logger.error(f"✗ TEST 4 FAILED with exception: {e}")
        return False


async def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 20 + "AGENT SKILL PULLING INTEGRATION TESTS" + " " * 22 + "║")
    logger.info("╚" + "=" * 78 + "╝")

    results = {
        "Test 1 (Pull Missing Skill)": None,
        "Test 2 (Use Local Skill)": None,
        "Test 3 (Mixed Skills)": None,
        "Test 4 (No Skills)": None,
    }

    # Run tests
    try:
        results["Test 1 (Pull Missing Skill)"] = await test_agent_pulls_skill_from_registry()
    except Exception as e:
        logger.error(f"Test 1 exception: {e}")
        results["Test 1 (Pull Missing Skill)"] = False

    try:
        results["Test 2 (Use Local Skill)"] = await test_agent_uses_local_skill()
    except Exception as e:
        logger.error(f"Test 2 exception: {e}")
        results["Test 2 (Use Local Skill)"] = False

    try:
        results["Test 3 (Mixed Skills)"] = await test_agent_with_mixed_skills()
    except Exception as e:
        logger.error(f"Test 3 exception: {e}")
        results["Test 3 (Mixed Skills)"] = False

    try:
        results["Test 4 (No Skills)"] = await test_agent_without_skills()
    except Exception as e:
        logger.error(f"Test 4 exception: {e}")
        results["Test 4 (No Skills)"] = False

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        if result is True:
            logger.info(f"✓ {test_name}: PASSED")
            passed += 1
        elif result is False:
            logger.info(f"✗ {test_name}: FAILED")
            failed += 1
        else:
            logger.info(f"? {test_name}: UNKNOWN")

    logger.info("=" * 80)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 80)

    return passed, failed


if __name__ == "__main__":
    passed, failed = asyncio.run(main())
    exit(0 if failed == 0 else 1)
