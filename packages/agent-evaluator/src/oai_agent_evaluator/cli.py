import argparse
import os
import sys
import importlib
from typing import Type

from .runner import RegressionRunner

def load_agent_class(class_path: str) -> Type:
    """
    Dynamically load the agent class from a string path.
    Format: 'module.submodule.ClassName'
    """
    try:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ValueError, ImportError, AttributeError) as e:
        print(f"Error loading agent class '{class_path}': {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Run agent regression tests.")
    
    parser.add_argument(
        "scenarios_path", 
        help="Path to the scenario file or directory."
    )
    
    parser.add_argument(
        "--agent-class",
        required=True,
        help="Full python path to the Agent class (e.g., 'oai_openai_agent_core.agents.openai_agent.OpenAIAgent')"
    )
    
    parser.add_argument(
        "--project-root", 
        default=os.getcwd(),
        help="Root directory of the project (default: current working directory)"
    )
    
    parser.add_argument(
        "--judge-model", 
        default="bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Model ID for the judge agent (default: bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0)"
    )
    
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to save HTML reports (default: reports)"
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent scenarios to run (default: 1)"
    )

    parser.add_argument(
        "--pass-threshold",
        type=float,
        default=7.0,
        help="Pass/fail threshold score (0-10, default: 7.0)"
    )

    args = parser.parse_args()
    
    # Add project root to sys.path to ensure we can import the agent class
    if args.project_root not in sys.path:
        sys.path.insert(0, args.project_root)

    # Load the agent class dynamically
    agent_class = load_agent_class(args.agent_class)

    # Initialize the runner
    runner = RegressionRunner(
        agent_class=agent_class,
        project_root=args.project_root,
        judge_model_id=args.judge_model,
        output_dir=args.output_dir,
        max_concurrency=args.concurrency,
        pass_threshold=args.pass_threshold
    )
    
    # Run the regression suite
    runner.run(args.scenarios_path)

if __name__ == "__main__":
    main()
