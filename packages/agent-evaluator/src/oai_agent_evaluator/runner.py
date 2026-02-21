import asyncio
import os
import sys
import logging
import importlib
from typing import Type, Optional, List, Dict, Any

from .evaluator import AgentEvaluator
from .loader import ScenarioLoader
from .scenario import TestScenario
from .reporter import HtmlReporter

class RegressionRunner:
    """
    Runner for executing agent regression tests.
    Handles scenario loading, execution, and result reporting.
    """

    def __init__(
        self,
        agent_class: Type,
        project_root: str,
        judge_model_id: str = "gpt-4o",
        default_scenarios: Optional[List[TestScenario]] = None,
        logger: Optional[logging.Logger] = None,
        output_dir: str = "reports",
        max_concurrency: int = 1
    ):
        """
        Initialize the RegressionRunner.

        Args:
            agent_class: The class of the agent to be tested.
            project_root: The root directory of the project.
            judge_model_id: The model ID for the judge agent.
            default_scenarios: List of scenarios to run if no path is provided.
            logger: Optional logger instance.
            output_dir: Directory to save reports.
            max_concurrency: Maximum number of concurrent scenarios to run.
        """
        self.agent_class = agent_class
        self.project_root = project_root
        self.judge_model_id = judge_model_id
        self.default_scenarios = default_scenarios or []
        self.output_dir = output_dir
        self.max_concurrency = max_concurrency
        
        # Configure logging if not provided
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(
                level=logging.INFO, 
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            self.logger = logging.getLogger("RegressionRunner")

    def _load_agent_class(self, class_path: str) -> Type:
        """Dynamically load the agent class from a string path."""
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ValueError, ImportError, AttributeError) as e:
            self.logger.error(f"Error loading agent class '{class_path}': {e}")
            raise

    async def run_async(self, scenarios_path: Optional[str] = None) -> bool:
        """
        Run the regression suite asynchronously.

        Args:
            scenarios_path: Path to a scenario file or directory.

        Returns:
            True if all tests passed, False otherwise.
        """
        # Check for API Key
        # if not os.getenv("OPENAI_API_KEY"):
        #     self.logger.error("OPENAI_API_KEY environment variable is not set.")
        #     return False

        scenarios = []
        
        # Load scenarios
        if scenarios_path:
            if os.path.exists(scenarios_path):
                self.logger.info(f"Loading scenarios from: {scenarios_path}")
                if os.path.isdir(scenarios_path):
                    scenarios = ScenarioLoader.load_from_directory(scenarios_path)
                elif os.path.isfile(scenarios_path):
                    scenarios = ScenarioLoader.load_from_file(scenarios_path)
            else:
                self.logger.warning(f"Path not found: {scenarios_path}.")
        
        if not scenarios:
            if self.default_scenarios:
                self.logger.info("Using default scenarios.")
                scenarios = self.default_scenarios
            else:
                self.logger.error("No scenarios found to run.")
                return False

        self.logger.info(f"Starting regression suite with {len(scenarios)} scenarios...")

        # Group scenarios by agent class
        scenarios_by_class = {}
        for scenario in scenarios:
            # Determine agent class for this scenario
            if scenario.agent_class:
                # Load class dynamically if it's a string
                if isinstance(scenario.agent_class, str):
                    try:
                        cls = self._load_agent_class(scenario.agent_class)
                    except Exception:
                        # Fallback to default if loading fails (or handle error)
                        cls = self.agent_class
                else:
                    cls = scenario.agent_class
            else:
                cls = self.agent_class
                
            if cls not in scenarios_by_class:
                scenarios_by_class[cls] = []
            scenarios_by_class[cls].append(scenario)

        all_results = []
        
        # Run evaluation for each agent class group
        for cls, class_scenarios in scenarios_by_class.items():
            # Handle MagicMock in tests which might not have __name__
            cls_name = getattr(cls, '__name__', 'MockAgent')
            self.logger.info(f"Evaluating {len(class_scenarios)} scenarios for agent class: {cls_name}")
            
            evaluator = AgentEvaluator(
                agent_class=cls,
                project_root=self.project_root,
                judge_model_id=self.judge_model_id,
                logger=self.logger,
                max_concurrency=self.max_concurrency
            )

            results = await evaluator.run_suite(class_scenarios)
            all_results.extend(results)

        # Generate HTML Report
        reporter = HtmlReporter(output_dir=self.output_dir)
        
        # Determine report name safely
        if len(scenarios_by_class) == 1:
            report_name = getattr(self.agent_class, '__name__', 'Agent')
        else:
            report_name = "MultiAgent_Suite"

        report_path = reporter.generate_report(all_results, report_name)
        self.logger.info(f"HTML Report generated at: {report_path}")

        return self._report_results(all_results)

    def run(self, scenarios_path: Optional[str] = None):
        """
        Synchronous entry point to run the regression suite.
        Exits with code 0 on success, 1 on failure.
        """
        try:
            success = asyncio.run(self.run_async(scenarios_path))
            if not success:
                sys.exit(1)
            sys.exit(0)
        except KeyboardInterrupt:
            print("\nExecution interrupted by user.")
            sys.exit(130)

    def _report_results(self, results: List[Dict[str, Any]]) -> bool:
        """
        Print formatted results to stdout.

        Returns:
            True if all tests passed, False otherwise.
        """
        passed_count = 0
        failed_count = 0
        
        print("\n" + "="*60)
        print("REGRESSION TEST RESULTS")
        print("="*60 + "\n")
        
        for res in results:
            status = "PASSED" if res['passed'] else "FAILED"
            if res['passed']:
                passed_count += 1
                icon = "✅"
            else:
                failed_count += 1
                icon = "❌"
                
            print(f"{icon} Scenario: {res['scenario']}")
            print(f"   Status:   {status}")
            print(f"   Score:    {res.get('score', 'N/A')}/10")
            
            if not res['passed']:
                print(f"   Input:    {res.get('input')}")
                print(f"   Actual:   {res.get('actual_output')}")
                print(f"   Expected: {res.get('expected_output')}")
                print(f"   Reason:   {res.get('explanation')}")
            elif res.get('error'):
                 print(f"   Error:    {res.get('error')}")
                 
            print("-" * 40)
            
        print("\n" + "="*60)
        print(f"Total: {len(results)} | Passed: {passed_count} | Failed: {failed_count}")
        print("="*60 + "\n")
        
        if failed_count > 0:
            self.logger.error("Regression suite failed.")
            return False
        else:
            self.logger.info("Regression suite passed successfully.")
            return True
