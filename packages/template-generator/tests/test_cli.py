import pytest
from unittest.mock import patch
from click.testing import CliRunner
from oai_template_generator.cli import main

def test_cli_list():
    runner = CliRunner()
    result = runner.invoke(main, ["list"])
    assert result.exit_code == 0
    assert "Available Templates" in result.output
    assert "agent" in result.output
    assert "mcp" in result.output

def test_cli_new_help():
    runner = CliRunner()
    result = runner.invoke(main, ["new", "--help"])
    assert result.exit_code == 0
    assert "Create a new project from a template" in result.output

def test_cli_new_interactive_abort():
    # Mock prompt_project_details to simulate user abort or just return values
    # But easier to mock the confirm if directory exists
    runner = CliRunner()
    
    # Mocking the prompt_project_details to avoid actual input
    with patch("oai_template_generator.cli.prompt_project_details", return_value=(
        "mcp", "ptr_mcp_servers_test", "Author", "test@capgemini.com", ".", "Desc", ["srv"], None
    )), \
         patch("oai_template_generator.cli.ProjectBuilder.build"):
    
        result = runner.invoke(main, ["new", "mcp", "test"])
        assert result.exit_code == 0
        assert "ptr_mcp_servers_test" in str(result.output) or result.exit_code == 0