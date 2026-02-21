"""Example Flow for random number and name generation.

This demonstrates how to create a CrewAI Flow that uses tools defined
in the YAML configuration.
"""

from crewai.flow.flow import Flow, start, listen, router
from typing import Dict, Any


class RandomFlow(Flow):
    """Flow that generates random numbers and names.

    This flow demonstrates:
    - Using tools injected from YAML configuration
    - Sequential flow execution with @start and @listen
    - Conditional routing with @router
    - State management across flow steps
    """

    def __init__(self, tools=None, llm=None, inputs=None):
        """Initialize the flow with injected dependencies.

        Args:
            tools: Dictionary of tool functions from YAML config
            llm: LLM instance for potential use in flow steps
            inputs: Initial inputs for the flow
        """
        super().__init__()
        self.tools = tools or {}
        self.llm = llm
        self.inputs = inputs or {}

        # Store results for final output
        self.results = {}

    @start()
    def generate_number(self):
        """Entry point: Generate a random number using configured tool.

        Returns:
            Dictionary with generated number
        """
        print("🎲 Generating random number...")

        # Get the tool from injected tools dictionary
        generate_fn = self.tools.get('generate_random_number')

        if generate_fn:
            try:
                # Call the tool
                number = generate_fn.run()
                print(f"✓ Generated number: {number}")

                self.results['number'] = number
                return {
                    "number": number,
                    "success": True
                }
            except Exception as e:
                print(f"✗ Error generating number: {e}")
                return {
                    "number": None,
                    "success": False,
                    "error": str(e)
                }
        else:
            print("✗ Tool 'generate_random_number' not found in tools")
            return {
                "number": None,
                "success": False,
                "error": "Tool not found"
            }

    @listen(generate_number)
    def generate_name(self, number_data: Dict[str, Any]):
        """Generate a random name after number generation.

        Args:
            number_data: Output from generate_number step

        Returns:
            Dictionary with both number and name
        """
        print("📝 Generating random name...")

        # Get the tool from injected tools dictionary
        generate_name_fn = self.tools.get('generate_random_name')

        if generate_name_fn and number_data.get('success'):
            try:
                # Call the tool
                name = generate_name_fn.run()
                print(f"✓ Generated name: {name}")

                self.results['name'] = name
                return {
                    "number": number_data.get("number"),
                    "name": name,
                    "success": True
                }
            except Exception as e:
                print(f"✗ Error generating name: {e}")
                return {
                    "number": number_data.get("number"),
                    "name": None,
                    "success": False,
                    "error": str(e)
                }
        else:
            error_msg = "Tool not found" if not generate_name_fn else "Previous step failed"
            print(f"✗ {error_msg}")
            return {
                "number": number_data.get("number"),
                "name": None,
                "success": False,
                "error": error_msg
            }

    @listen(generate_name)
    def format_output(self, name_data: Dict[str, Any]):
        """Format the final output with both number and name.

        Args:
            name_data: Output from generate_name step

        Returns:
            Formatted final result
        """
        print("📋 Formatting output...")

        number = name_data.get('number')
        name = name_data.get('name')
        success = name_data.get('success')

        if success:
            result = f"Random Generation Results:\n" \
                     f"  Number: {number}\n" \
                     f"  Name: {name}"
            print(f"✓ {result}")
        else:
            result = f"Generation failed: {name_data.get('error', 'Unknown error')}"
            print(f"✗ {result}")

        return {
            "result": result,
            "number": number,
            "name": name,
            "success": success
        }


class ConditionalRandomFlow(Flow):
    """Example flow with conditional routing.

    Demonstrates @router for conditional flow execution.
    """

    def __init__(self, tools=None, llm=None, inputs=None):
        super().__init__()
        self.tools = tools or {}
        self.llm = llm
        self.inputs = inputs or {}

    @start()
    def start_flow(self):
        """Entry point."""
        return {"step": "started"}

    @router(start_flow)
    def decide_path(self, data: Dict[str, Any]):
        """Route to different paths based on conditions.

        Returns:
            String indicating which path to take
        """
        # Example: route based on input or random decision
        import random
        if random.random() > 0.5:
            return "path_a"
        else:
            return "path_b"

    @listen("path_a")
    def path_a_handler(self, data: Dict[str, Any]):
        """Handle path A."""
        print("Taking path A")
        return {"path": "A", "data": data}

    @listen("path_b")
    def path_b_handler(self, data: Dict[str, Any]):
        """Handle path B."""
        print("Taking path B")
        return {"path": "B", "data": data}