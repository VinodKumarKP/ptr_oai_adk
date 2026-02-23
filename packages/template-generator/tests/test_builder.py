import shutil
import subprocess
from pathlib import Path
import pytest
from oai_template_generator.builder import ProjectBuilder

@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path / "output"

def test_builder_initialization(temp_output_dir):
    builder = ProjectBuilder(
        template="mcp",
        project_name="test_mcp",
        author="Test Author",
        email="test@capgemini.com",
        description="Test Description",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=False
    )
    assert builder.project_name == "test_mcp"
    assert builder.template == "mcp"
    assert builder.project_dir == temp_output_dir / "test_mcp"

def test_mcp_build(temp_output_dir):
    builder = ProjectBuilder(
        template="mcp",
        project_name="ptr_mcp_servers_test",
        author="Test Author",
        email="test@capgemini.com",
        description="Test Description",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=False,
        items=["weather_server"]
    )
    
    builder.build()
    
    assert builder.project_dir.exists()
    assert (builder.project_dir / "pyproject.toml").exists()
    assert (builder.project_dir / "mcp_registry_servers" / "servers" / "weather_server" / "server.py").exists()
    assert (builder.project_dir / "mcp_registry_servers" / "servers_config" / "weather_server.yaml").exists()
    
    # Check if template files were deleted
    assert not (builder.project_dir / "mcp_registry_servers" / "server.template").exists()
    assert not (builder.project_dir / "mcp_registry_servers" / "servers_config" / "template_server.yaml").exists()

def test_agent_build(temp_output_dir):
    builder = ProjectBuilder(
        template="agent",
        project_name="ptr_agent_servers_test",
        author="Test Author",
        email="test@capgemini.com",
        description="Test Description",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=False,
        items=["research_agent"],
        framework="langgraph"
    )
    
    builder.build()
    
    assert builder.project_dir.exists()
    assert (builder.project_dir / "agentic_registry_agents" / "agents" / "research_agent" / "agent.py").exists()
    assert (builder.project_dir / "agentic_registry_agents" / "agents_config" / "research_agent.yaml").exists()
    
    # Check if template files were deleted
    assert not (builder.project_dir / "agentic_registry_agents" / "agents" / "server_template").exists()
    assert not (builder.project_dir / "agentic_registry_agents" / "agents_config" / "template.yaml").exists()
    
    # Check pyproject.toml for framework dependency
    pyproject_content = (builder.project_dir / "pyproject.toml").read_text()
    assert '"oai-langgraph-core",' in pyproject_content

def test_builder_validate_missing_template(temp_output_dir):
    builder = ProjectBuilder(
        template="non_existent",
        project_name="test",
        author="A",
        email="a@capgemini.com",
        description="D",
        output_dir=temp_output_dir
    )
    with pytest.raises(SystemExit):
        builder._validate()

def test_builder_validate_existing_dir(temp_output_dir):
    project_dir = temp_output_dir / "existing"
    project_dir.mkdir(parents=True)
    builder = ProjectBuilder(
        template="mcp",
        project_name="existing",
        author="A",
        email="a@capgemini.com",
        description="D",
        output_dir=temp_output_dir,
        overwrite=False
    )
    with pytest.raises(SystemExit):
        builder._validate()

def test_builder_overwrite(temp_output_dir):
    project_dir = temp_output_dir / "overwrite_me"
    project_dir.mkdir(parents=True)
    (project_dir / "old_file.txt").write_text("old")
    
    builder = ProjectBuilder(
        template="mcp",
        project_name="overwrite_me",
        author="A",
        email="a@capgemini.com",
        description="D",
        output_dir=temp_output_dir,
        overwrite=True,
        init_git=False,
        create_venv=False
    )
    builder.build()
    assert not (project_dir / "old_file.txt").exists()
    assert (project_dir / "pyproject.toml").exists()

def test_git_init(temp_output_dir, mocker):
    mock_run = mocker.patch("subprocess.run")
    builder = ProjectBuilder(
        template="mcp",
        project_name="git_test",
        author="A",
        email="a@capgemini.com",
        description="D",
        output_dir=temp_output_dir,
        init_git=True,
        create_venv=False
    )
    builder.build()
    assert mock_run.call_count >= 3 # init, add, commit

def test_venv_create(temp_output_dir, mocker):
    mock_run = mocker.patch("subprocess.run")
    builder = ProjectBuilder(
        template="mcp",
        project_name="venv_test",
        author="A",
        email="a@capgemini.com",
        description="D",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=True
    )
    builder.build()
    # Check if venv creation was called
    called_args = [call.args[0] for call in mock_run.call_args_list]
    assert any("venv" in args for args in called_args)
