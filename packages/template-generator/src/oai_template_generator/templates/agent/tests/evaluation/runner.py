#!/usr/bin/env python3
import os
import sys
import argparse

# Add project root to sys.path to allow importing the library
current_dir = os.path.dirname(os.path.abspath(__file__))
default_project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if default_project_root not in sys.path:
    sys.path.insert(0, default_project_root)

# Import the Agent Class to be tested
# from oai_openai_agent_core.agents.openai_agent import OpenAIAgent

from oai_agent_evaluator import RegressionRunner, TestScenario

# Default Scenarios
DEFAULT_SCENARIOS = [
    TestScenario(
        name="Simple Greeting",
        description="Test if the agent can greet the user.",
        input_message="Hello, who are you?",
        expected_output="I am a helpful assistant.",
        agent_config={
            'system_prompt': "You are a helpful assistant.",
            'tools': []
        }
    )
]

def main():
    parser = argparse.ArgumentParser(description="Run agent regression tests.")
    
    parser.add_argument(
        "scenarios_path", 
        nargs="?", 
        help="Path to the scenario file or directory."
    )
    
    parser.add_argument(
        "--project-root", 
        default=default_project_root,
        help=f"Root directory of the project (default: {default_project_root})"
    )
    
    parser.add_argument(
        "--judge-model", 
        default="bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0",
        help="Model ID for the judge agent (default: bedrock/global.anthropic.claude-sonnet-4-5-20250929-v1:0)"
    )

    args = parser.parse_args()

    # Initialize the runner
    runner = RegressionRunner(
        agent_class="Agent Class to be tested",
        project_root=args.project_root,
        default_scenarios=DEFAULT_SCENARIOS,
        judge_model_id=args.judge_model
    )
    
    # Run the regression suite
    runner.run(args.scenarios_path)

if __name__ == "__main__":
    main()
