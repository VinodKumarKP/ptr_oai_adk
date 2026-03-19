"""Builder for creating CrewAI Task instances."""

import logging
from typing import Dict, List, Any, Tuple, Optional

from crewai import Agent, Task

from oai_agent_core.components.output_parser.output_model_registry import OutputModelRegistry


class TaskBuilder:
    """Builds CrewAI tasks from configuration.

    Handles:
    - Task creation
    - Context dependency resolution
    - Agent mapping
    """

    def __init__(self, config: Dict[str, Any],
                 logger: Optional[logging.Logger] = None,
                 structured_output_model_registry: Optional[OutputModelRegistry] = None):
        """Initialize the task builder.

        Args:
            config: Configuration dictionary containing 'task_list' and 'agent_list'
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.structured_output_model_registry = structured_output_model_registry

    def build_tasks(self, agent_map: Dict[str, Agent]) -> Tuple[Dict[str, Task], List[Task], List[Dict]]:
        """Build all tasks from configuration.

        Args:
            agent_map: Map of agent keys to Agent instances

        Returns:
            Tuple of (task_map, task_list, agent_level_tasks)
            - task_map: Maps task keys to Task instances
            - task_list: List of all Task instances
            - agent_level_tasks: List of agent-level task configs
        """
        task_map = {}
        task_list = []

        # Collect agent-level tasks
        agent_level_tasks = self._collect_agent_level_tasks()

        # Combine crew-level and agent-level tasks
        all_tasks = self.config.get('task_list', []) + agent_level_tasks

        # Create Task instances
        for task_config in all_tasks:
            task_key = list(task_config.keys())[0]
            task_data = task_config[task_key]

            # Create task
            task = self._create_task(task_key, task_data, agent_map, task_map)

            # Register task
            task_map[task_key] = task
            task_list.append(task)

        return task_map, task_list, agent_level_tasks

    def _collect_agent_level_tasks(self) -> List[Dict]:
        """Collect tasks defined at agent level.

        Returns:
            List of agent-level task configurations
        """
        agent_level_tasks = []

        for agent_config in self.config.get('agent_list', []):
            agent_key = list(agent_config.keys())[0]
            agent_data = agent_config[agent_key]

            if 'tasks' in agent_data:
                for task_config in agent_data['tasks']:
                    task_key = list(task_config.keys())[0]
                    task_data = task_config[task_key]
                    task_data['agent'] = agent_key
                    agent_level_tasks.append({task_key: task_data})

        return agent_level_tasks

    def _create_task(self,
                     task_key: str,
                     task_data: Dict[str, Any],
                     agent_map: Dict[str, Agent],
                     task_map: Dict[str, Task]) -> Task:
        """Create a single CrewAI task.

        Args:
            task_key: Unique identifier for the task
            task_data: Task configuration dictionary
            agent_map: Map of agent keys to Agent instances
            task_map: Map of existing task keys to Task instances

        Returns:
            Configured Task instance

        Raises:
            ValueError: If agent reference is invalid
        """
        # Validate agent reference
        agent_key = task_data.get('agent')
        if agent_key not in agent_map:
            raise ValueError(
                f"Agent '{agent_key}' not found for task '{task_key}'"
            )

        # Resolve context dependencies
        context_tasks = self._resolve_context(task_key, task_data, task_map)

        # Create task
        task = Task(
            description=task_data.get('description', f"Perform {task_key} duties"),
            expected_output=task_data.get('expected_output', f"Output from {task_key}"),
            agent=agent_map[agent_key],
            context=context_tasks if context_tasks else None,
            config=task_data,
            output_pydantic=self.structured_output_model_registry.get_model(task_data.get('structured_output_model', None)),
        )

        return task

    def _resolve_context(self,
                         task_key: str,
                         task_data: Dict[str, Any],
                         task_map: Dict[str, Task]) -> List[Task]:
        """Resolve task context dependencies.

        Args:
            task_key: Current task identifier
            task_data: Task configuration dictionary
            task_map: Map of existing task keys to Task instances

        Returns:
            List of context Task instances
        """
        context_tasks = []

        if 'context' in task_data:
            for ctx_task_key in task_data['context']:
                if ctx_task_key in task_map:
                    context_tasks.append(task_map[ctx_task_key])
                else:
                    self.logger.warning(
                        f"Context task '{ctx_task_key}' not found for task '{task_key}'"
                    )

        return context_tasks
