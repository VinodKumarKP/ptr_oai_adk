from crewai.tools import tool

@tool("generate_random_name")
def generate_random_name() -> str:
    """Generate a random name"""
    # Tool logic here
    return "John"

@tool("generate_random_number")
def generate_random_number(lower:int=0, upper:int=100):
    """Generate a random integer between lower and upper bounds."""
    import random
    return random.randint(lower, upper)