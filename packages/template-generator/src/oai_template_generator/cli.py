"""Python Project Template Builder CLI."""

import sys
from pathlib import Path
import click

from oai_template_generator.builder import ProjectBuilder
from oai_template_generator.prompts import prompt_project_details, confirm


@click.group()
def main():
    """🚀 Python Project Template Builder — scaffold agent or MCP projects instantly."""
    pass


@main.command()
def list():
    """List available templates."""
    click.echo("\n📦 Available Templates\n")
    click.echo("  agent  — Autonomous AI agent with tool use, memory, and configurable LLM backend")
    click.echo("  mcp    — Model Context Protocol server with resources, prompts, and tools\n")


@main.command()
@click.argument("template", type=click.Choice(["agent", "mcp"]), required=False)
@click.argument("name", required=False)
@click.option(
    "--output-dir",
    "-o",
    default=".",
    help="Directory to create the project in (default: current directory)",
)
@click.option("--author", "-a", default="", help="Author name")
@click.option("--email", "-e", default="", help="Author email")
@click.option("--description", "-d", default="", help="Project description")
@click.option("--no-git", is_flag=True, help="Skip git initialization")
@click.option("--no-venv", is_flag=True, help="Skip virtual environment creation")
def new(template, name, output_dir, author, email, description, no_git, no_venv):
    """Create a new project from a template."""
    
    # Interactive prompts for missing values
    template, name, author, email, output_dir_str, description, items, framework = prompt_project_details(
        template=template,
        name=name,
        author=author,
        email=email,
        output_dir=output_dir,
        description=description,
    )

    output_dir_path = Path(output_dir_str).resolve()
    project_dir = output_dir_path / name
    
    overwrite = False
    if project_dir.exists():
        if confirm(f"⚠  Directory '{project_dir}' already exists. Overwrite?", default=False):
            overwrite = True
        else:
            click.echo("Aborted.")
            sys.exit(0)

    builder = ProjectBuilder(
        template=template,
        project_name=name,
        author=author,
        email=email,
        description=description,
        output_dir=output_dir_path,
        init_git=not no_git,
        create_venv=not no_venv,
        items=items,
        framework=framework,
        overwrite=overwrite,
    )

    builder.build()


if __name__ == "__main__":
    main()
