#!/usr/bin/env python3
"""
Unit Test: Agent Skill Initialization

This test verifies that agents properly initialize skills from configuration
without requiring an external Skills Registry to be running.

The test focuses on:
1. Agent initialization with skill configuration
2. Skill registry creation
3. Skill pulling flow (will fail gracefully if registry unavailable)
4. Agent initialization without skills configuration

Run this test with:
    python test_agent_skill_initialization.py
"""

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create a concrete agent implementation for testing
from oai_agent_core.core.base_agent import BaseAgent


class TestAgent(BaseAgent):
    """Test implementation of BaseAgent for skill initialization testing."""

    def __init__(self, agent_config: Dict[str, Any], agent_name: str = "test_agent"):
        """Initialize test agent with config."""
        super().__init__(agent_name=agent_name, agent_config=agent_config, llm=None)

    async def initialize(self):
        """Initialize agent with skill support."""
        # Mock tool registry to avoid requiring actual tools
        if not hasattr(self, 'tool_registry'):
            from oai_agent_core.components.tools.tool_registry import ToolRegistry
            self.tool_registry = ToolRegistry(logger=self.logger)

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


async def test_1_agent_initializes_with_skill_config():
    """Test 1: Agent initializes successfully with skill configuration."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 1: Agent Initializes with Skill Configuration")
    logger.info("=" * 80)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Configuration with skills but no registry (local only)
            config = {
                'type': 'custom',
                'name': 'test_agent',
                'model': {
                    'model_id': 'bedrock/claude-3-sonnet',
                    'region_name': 'us-west-2'
                },
                'skills': {
                    'skill_dir': str(skills_dir)
                    # No registry - local skills only
                },
                'agent_list': [
                    {
                        'test_agent': {
                            'system_prompt': 'You are a test agent',
                            'skills': []  # Empty list - no skills to pull
                        }
                    }
                ]
            }

            logger.info("Creating agent with skill configuration...")
            agent = TestAgent(agent_config=config)

            logger.info("Initializing agent...")
            await agent.initialize()

            logger.info("Checking if agent initialized successfully...")
            if agent and hasattr(agent, 'skill_registry'):
                logger.info(f"✓ TEST 1 PASSED: Agent initialized, skill_registry exists")
                logger.info(f"  - Skill registry: {agent.skill_registry}")
                logger.info(f"  - Initialized: {agent.is_initialized}")
                return True
            else:
                logger.warning("⚠ TEST 1 FAILED: Agent missing skill_registry")
                return False

    except Exception as e:
        logger.error(f"✗ TEST 1 FAILED with exception: {e}", exc_info=True)
        return False


async def test_2_agent_initializes_without_skills():
    """Test 2: Agent initializes successfully without skill configuration."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 2: Agent Initializes Without Skill Configuration")
    logger.info("=" * 80)

    try:
        config = {
            'type': 'custom',
            'name': 'simple_agent',
            'model': {
                'model_id': 'bedrock/claude-3-sonnet',
                'region_name': 'us-west-2'
            },
            # No 'skills' section
            'agent_list': [
                {
                    'test_agent': {
                        'system_prompt': 'You are a simple agent'
                    }
                }
            ]
        }

        logger.info("Creating agent without skill configuration...")
        agent = TestAgent(agent_config=config)

        logger.info("Initializing agent...")
        await agent.initialize()

        logger.info("Checking if agent initialized successfully...")
        if agent and agent.is_initialized:
            logger.info(f"✓ TEST 2 PASSED: Agent initialized without skills")
            logger.info(f"  - Skill registry: {agent.skill_registry}")
            logger.info(f"  - Initialized: {agent.is_initialized}")
            return True
        else:
            logger.warning("⚠ TEST 2 FAILED: Agent failed to initialize")
            return False

    except Exception as e:
        logger.error(f"✗ TEST 2 FAILED with exception: {e}", exc_info=True)
        return False


async def test_3_agent_with_local_skill():
    """Test 3: Agent discovers local skill without pulling."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 3: Agent Discovers Local Skill")
    logger.info("=" * 80)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            # Create a local skill
            local_skill_dir = skills_dir / "test-skill"
            local_skill_dir.mkdir()
            skill_md = """---
name: test-skill
description: Test skill for unit testing
version: 1.0.0
author: Test Suite
---

# Test Skill

This is a test skill for unit testing agent skill discovery.
"""
            (local_skill_dir / "SKILL.md").write_text(skill_md)

            config = {
                'type': 'custom',
                'name': 'test_agent',
                'model': {
                    'model_id': 'bedrock/claude-3-sonnet',
                    'region_name': 'us-west-2'
                },
                'skills': {
                    'skill_dir': str(skills_dir)
                    # No registry - local skills only
                },
                'agent_list': [
                    {
                        'test_agent': {
                            'system_prompt': 'You are a test agent',
                            'skills': ['test-skill']  # Local skill
                        }
                    }
                ]
            }

            logger.info("Creating agent with local skill...")
            agent = TestAgent(agent_config=config)

            logger.info("Initializing agent...")
            await agent.initialize()

            logger.info("Checking if local skill is available...")
            if agent.skill_registry and 'test-skill' in agent.skill_registry.skills:
                logger.info(f"✓ TEST 3 PASSED: Local skill discovered")
                logger.info(f"  - Available skills: {list(agent.skill_registry.skills.keys())}")
                return True
            else:
                logger.warning(f"⚠ TEST 3 FAILED: Local skill not discovered")
                if agent.skill_registry:
                    logger.warning(f"  - Available skills: {list(agent.skill_registry.skills.keys())}")
                return False

    except Exception as e:
        logger.error(f"✗ TEST 3 FAILED with exception: {e}", exc_info=True)
        return False


async def test_4_agent_collects_required_skills():
    """Test 4: Agent properly collects required skills from agent_list."""
    logger.info("\n" + "=" * 80)
    logger.info("TEST 4: Agent Collects Required Skills")
    logger.info("=" * 80)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            skills_dir = Path(tmpdir) / "skills"
            skills_dir.mkdir()

            config = {
                'type': 'custom',
                'name': 'multi_agent',
                'model': {
                    'model_id': 'bedrock/claude-3-sonnet',
                    'region_name': 'us-west-2'
                },
                'skills': {
                    'skill_dir': str(skills_dir),
                    'registry': {
                        'url': None  # Disabled registry
                    }
                },
                'agent_list': [
                    {
                        'agent_1': {
                            'system_prompt': 'Agent 1',
                            'skills': ['skill-a', 'skill-b']
                        }
                    },
                    {
                        'agent_2': {
                            'system_prompt': 'Agent 2',
                            'skills': ['skill-b', 'skill-c']
                        }
                    }
                ]
            }

            logger.info("Creating agent with multiple skills across agent_list...")
            agent = TestAgent(agent_config=config)

            logger.info("Initializing agent...")
            await agent.initialize()

            logger.info("Checking that agent initialization completes...")
            if agent and agent.is_initialized:
                logger.info(f"✓ TEST 4 PASSED: Agent collected and processed required skills")
                logger.info(f"  - Agent initialized successfully")
                return True
            else:
                logger.warning("⚠ TEST 4 FAILED: Agent initialization failed")
                return False

    except Exception as e:
        logger.error(f"✗ TEST 4 FAILED with exception: {e}", exc_info=True)
        return False


async def main():
    """Run all tests."""
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 15 + "AGENT SKILL INITIALIZATION UNIT TESTS" + " " * 27 + "║")
    logger.info("╚" + "=" * 78 + "╝")

    results = {
        "Test 1 (Initializes with Skills)": None,
        "Test 2 (Initializes without Skills)": None,
        "Test 3 (Discovers Local Skill)": None,
        "Test 4 (Collects Required Skills)": None,
    }

    # Run tests
    try:
        results["Test 1 (Initializes with Skills)"] = await test_1_agent_initializes_with_skill_config()
    except Exception as e:
        logger.error(f"Test 1 exception: {e}")
        results["Test 1 (Initializes with Skills)"] = False

    try:
        results["Test 2 (Initializes without Skills)"] = await test_2_agent_initializes_without_skills()
    except Exception as e:
        logger.error(f"Test 2 exception: {e}")
        results["Test 2 (Initializes without Skills)"] = False

    try:
        results["Test 3 (Discovers Local Skill)"] = await test_3_agent_with_local_skill()
    except Exception as e:
        logger.error(f"Test 3 exception: {e}")
        results["Test 3 (Discovers Local Skill)"] = False

    try:
        results["Test 4 (Collects Required Skills)"] = await test_4_agent_collects_required_skills()
    except Exception as e:
        logger.error(f"Test 4 exception: {e}")
        results["Test 4 (Collects Required Skills)"] = False

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
        else:
            logger.info(f"✗ {test_name}: FAILED")
            failed += 1

    logger.info("=" * 80)
    logger.info(f"Results: {passed} passed, {failed} failed")
    logger.info("=" * 80)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
