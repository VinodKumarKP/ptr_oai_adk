"""Interactive prompts for the CLI."""

import re
import sys


def _ask(prompt: str, default: str = "") -> str:
    """Ask a question, return the answer (or default if blank)."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nAborted.")
        sys.exit(1)
    return answer if answer else default


def _choose(prompt: str, choices: list[str]) -> str:
    """Present a numbered menu and return the chosen value."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}) {choice}")
    while True:
        raw = _ask("Enter number")
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        print(f"  ⚠  Please enter a number between 1 and {len(choices)}.")


def confirm(prompt: str, default: bool = False) -> bool:
    """Ask a yes/no question."""
    default_str = "Y/n" if default else "y/N"
    while True:
        answer = _ask(f"{prompt} ({default_str})").lower()
        if not answer:
            return default
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("  ⚠  Please enter 'y' or 'n'.")


def _slugify(name: str) -> str:
    """Convert a name to a valid Python package slug."""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    name = re.sub(r"^[^a-z]+", "", name)
    return name


def prompt_project_details(
    template: str | None,
    name: str | None,
    author: str,
    email: str | None,
    output_dir: str | None,
    description: str,
) -> tuple[str, str, str, str, str, str, list[str], str | None]:
    """Interactively fill in any missing project details."""

    print("\n✨  Python Project Template Builder\n")

    # Template
    if not template:
        template = _choose("Select a template:", ["agent", "mcp"])
    else:
        print(f"  Template : {template}")

    # Project name
    if not name:
        name = _ask("Project name")
        while not name:
            print("  ⚠  Project name is required.")
            name = _ask("Project name")

    slug = _slugify(name)
    if template == "mcp" and not slug.startswith("ptr_mcp_servers_"):
        slug = f"ptr_mcp_servers_{slug}"
    elif template == "agent" and not slug.startswith("ptr_agent_servers_"):
        slug = f"ptr_agent_servers_{slug}"
    
    if slug != name:
        print(f"  ℹ  Package name will be: {slug!r}")

    # Author
    if not author:
        author = _ask("Author name", default="Your Name")

    # Email
    if not email:
        while True:
            email = _ask("Email address (must be @capgemini.com)", default="user@capgemini.com")
            if email.endswith("@capgemini.com"):
                break
            print("  ⚠  Email must be a @capgemini.com address.")

    # Output Directory
    if not output_dir or output_dir == ".":
        output_dir = _ask("Output directory", default=".")

    # Description
    if not description:
        default_desc = (
            "An AI agent project." if template == "agent" else "An MCP server project."
        )
        description = _ask("Short description", default=default_desc)

    items = []
    framework = None
    if template == "mcp":
        servers_input = _ask("List of MCP servers (comma-separated)", default="default_server")
        items = [s.strip() for s in servers_input.split(",") if s.strip()]
    elif template == "agent":
        agents_input = _ask("List of Agents (comma-separated)", default="default_agent")
        raw_items = [s.strip() for s in agents_input.split(",") if s.strip()]
        # Ensure each agent name ends with _agent
        items = []
        for item in raw_items:
            if not item.endswith("_agent"):
                item = f"{item}_agent"
            items.append(item)

        framework = _choose("Select a framework:", ["langgraph", "crewai", "strands", "openai"])

    print()
    return template, slug, author, email, output_dir, description, items, framework
