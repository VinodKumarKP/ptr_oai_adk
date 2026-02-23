import pytest
from oai_template_generator.prompts import _slugify, prompt_project_details

def test_slugify():
    assert _slugify("My Project") == "my_project"
    assert _slugify("My-Project") == "my_project"
    assert _slugify("My_Project") == "my_project"
    assert _slugify("  My Project  ") == "my_project"
    assert _slugify("123MyProject") == "myproject"

def test_prompt_project_details_mcp(mocker):
    # Mock inputs for prompt_project_details
    # template, slug, author, email, output_dir, description, items, framework
    
    # Mocking input() calls
    mocker.patch("builtins.input", side_effect=[
        "mcp_test", # Project name
        "Author Name", # Author
        "test@capgemini.com", # Email
        ".", # Output dir
        "Description", # Description
        "server1, server2" # MCP servers
    ])
    
    template, slug, author, email, output_dir, description, items, framework = prompt_project_details(
        template="mcp",
        name=None,
        author=None,
        email=None,
        output_dir=None,
        description=None
    )
    
    assert template == "mcp"
    assert slug == "ptr_mcp_servers_mcp_test"
    assert author == "Author Name"
    assert email == "test@capgemini.com"
    assert items == ["server1_server", "server2_server"]
    assert framework is None

def test_prompt_project_details_agent(mocker):
    # Mocking input() calls
    # template, name, author, email, output_dir, description, items, framework
    mocker.patch("builtins.input", side_effect=[
        "agent_test", # Project name
        "Author Name", # Author
        "test@capgemini.com", # Email
        ".", # Output dir
        "Description", # Description
        "agent1, agent2", # Agents
        "1" # Framework (langgraph)
    ])
    
    template, slug, author, email, output_dir, description, items, framework = prompt_project_details(
        template="agent",
        name=None,
        author=None,
        email=None,
        output_dir=None,
        description=None
    )
    
    assert template == "agent"
    assert slug == "ptr_agent_servers_agent_test"
    assert items == ["agent1_agent", "agent2_agent"]
    assert framework == "langgraph"
