import shutil
import subprocess
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock
from oai_template_generator.builder import ProjectBuilder
from oai_template_generator.constants import TOKEN_AUTHOR

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
    # Prepare items dict
    items = [{
        "name": "weather_server",
        "class_name": "WeatherTools",
        "port": "8080",
        "description": "Weather service",
        "tags": ["weather"],
        "source": "",
        "env": {}
    }]
    
    builder = ProjectBuilder(
        template="mcp",
        project_name="ptr_mcp_servers_test",
        author="Test Author",
        email="test@capgemini.com",
        description="Test Description",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=False,
        items=items
    )
    
    builder.build()
    
    assert builder.project_dir.exists()
    assert (builder.project_dir / "pyproject.toml").exists()
    assert (builder.project_dir / "mcp_registry_servers" / "servers" / "weather_server" / "server.py").exists()
    assert (builder.project_dir / "mcp_registry_servers" / "servers_config" / "weather_server.yaml").exists()
    assert (builder.project_dir / "mcp_registry_servers" / "tools" / "weather_server.py").exists()
    
    # Verify content of generated files
    yaml_content = (builder.project_dir / "mcp_registry_servers" / "servers_config" / "weather_server.yaml").read_text()
    assert "port: 8080" in yaml_content
    
    tools_content = (builder.project_dir / "mcp_registry_servers" / "tools" / "weather_server.py").read_text()
    assert "class WeatherTools" in tools_content

def test_agent_build_simple(temp_output_dir):
    items = [{
        "name": "research_agent",
        "port": "9000",
        "description": "Researcher",
        "instructions": "You research stuff",
        "model_id": "claude-3",
        "region": "us-east-1",
        "use_tools": True,
        "tool_list": ["google_search"],
        "mcp_servers": [{"name": "test_mcp", "type": "stdio", "command": "echo", "args": [], "env": {}}],
        "sub_agents": [],
        "global_kb": [],
        "memory_config": {},
        "use_guardrails": True,
        "tags": ["research"],
        "prompts": [],
        "env": {}
    }]

    builder = ProjectBuilder(
        template="agent",
        project_name="ptr_agent_servers_test",
        author="Test Author",
        email="test@capgemini.com",
        description="Test Description",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=False,
        items=items,
        framework="langgraph"
    )
    
    builder.build()
    
    assert builder.project_dir.exists()
    assert (builder.project_dir / "agentic_registry_agents" / "agents" / "research_agent" / "agent.py").exists()
    assert (builder.project_dir / "agentic_registry_agents" / "agents_config" / "research_agent.yaml").exists()
    assert (builder.project_dir / "agentic_registry_agents" / "utils" / "research_agent_utils.py").exists()
    
    # Verify config content
    yaml_content = (builder.project_dir / "agentic_registry_agents" / "agents_config" / "research_agent.yaml").read_text()
    assert "port: 9000" in yaml_content
    assert "guardrails:" in yaml_content
    assert "research_agent_utils" in yaml_content 
    assert "test_mcp" in yaml_content
    
    # Verify tools file
    utils_content = (builder.project_dir / "agentic_registry_agents" / "utils" / "research_agent_utils.py").read_text()
    assert "def google_search():" in utils_content

def test_builder_validate_missing_template(temp_output_dir):
    builder = ProjectBuilder(
        template="non_existent",
        project_name="test",
        author="A",
        email="a@capgemini.com",
        description="D",
        output_dir=temp_output_dir
    )
    with pytest.raises(FileNotFoundError):
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
        create_venv=False,
        items=[{"name": "test_server", "port": "8001", "description": "d", "tags": [], "source": "", "env": {}}]
    )
    builder.build()
    assert not (project_dir / "old_file.txt").exists()
    assert (project_dir / "pyproject.toml").exists()

def test_git_init(temp_output_dir):
    with patch("subprocess.run") as mock_run:
        builder = ProjectBuilder(
            template="mcp",
            project_name="git_test",
            author="A",
            email="a@capgemini.com",
            description="D",
            output_dir=temp_output_dir,
            init_git=True,
            create_venv=False,
            items=[{"name": "test_server", "port": "8001", "description": "d", "tags": [], "source": "", "env": {}}]
        )
        builder.build()
        assert mock_run.call_count >= 3 # init, add, commit

def test_venv_create(temp_output_dir):
    with patch("subprocess.run") as mock_run:
        builder = ProjectBuilder(
            template="mcp",
            project_name="venv_test",
            author="A",
            email="a@capgemini.com",
            description="D",
            output_dir=temp_output_dir,
            init_git=False,
            create_venv=True,
            items=[{"name": "test_server", "port": "8001", "description": "d", "tags": [], "source": "", "env": {}}]
        )
        builder.build()
        # Check if venv creation was called
        called_args = [call.args[0] for call in mock_run.call_args_list]
        assert any("venv" in args for args in called_args)

# --- Coverage Improvements ---

def test_build_exception_handling(temp_output_dir):
    """Test exception handling during build."""
    builder = ProjectBuilder(
        template="mcp",
        project_name="fail_project",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir
    )
    # Mock _validate to raise exception
    with patch.object(builder, '_validate', side_effect=Exception("Boom")):
        with pytest.raises(SystemExit):
            builder.build()

def test_validate_empty_name(temp_output_dir):
    """Test validation with empty project name."""
    builder = ProjectBuilder(
        template="mcp",
        project_name="",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir
    )
    with pytest.raises(SystemExit):
        builder._validate()

def test_mcp_build_with_env(temp_output_dir):
    """Test MCP build with environment variables."""
    items = [{
        "name": "env_server",
        "env": {"KEY": "VALUE"},
        "tags": ["tag1"],
        "source": "src",
        "class_name": "MyClass",
        "port": 9090,
        "description": "desc"
    }]
    
    builder = ProjectBuilder(
        template="mcp",
        project_name="env_mcp",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir,
        items=items,
        init_git=False,
        create_venv=False
    )
    
    builder.build()
    
    yaml_path = builder.project_dir / "mcp_registry_servers" / "servers_config" / "env_server.yaml"
    assert yaml_path.exists()
    content = yaml_path.read_text()
    assert "KEY: VALUE" in content
    assert "tag1" in content
    assert "src" in content

def test_agent_build_complex(temp_output_dir):
    """Test agent build with complex configuration to cover all branches."""
    
    sub_agent_1 = {
        "name": "sub1",
        "context": [],
        "knowledge_base": [{"name": "kb1", "type": "postgres", "description": "d"}]
    }
    sub_agent_2 = {
        "name": "sub2",
        "context": ["sub1"],
        "knowledge_base": []
    }
    
    items = [{
        "name": "complex_agent",
        "port": 9000,
        "description": "desc",
        "instructions": "inst",
        "model_id": "gpt-4",
        "region": "us-east",
        "sub_agents": [sub_agent_1, sub_agent_2],
        "global_kb": [{"name": "gkb", "type": "s3", "description": "gd"}],
        "memory_config": {"type": "postgres", "collection_name": "mem"},
        "use_guardrails": False,
        "mcp_servers": [{"name": "mcp1", "type": "stdio", "command": "echo", "args": [], "env": {}}],
        "tool_list": [], # No tools covers 'else' branch
        "tags": ["t1"],
        "prompts": ["p1"],
        "entry_agent": "sub1",
        "env": {"E": "V"}
    }]
    
    builder = ProjectBuilder(
        template="agent",
        project_name="complex_agent",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir,
        items=items,
        framework="openai", # Cover openai framework dependency update
        init_git=False,
        create_venv=False
    )
    
    builder.build()
    
    config_path = builder.project_dir / "agentic_registry_agents" / "agents_config" / "complex_agent.yaml"
    assert config_path.exists()
    content = config_path.read_text()
    
    assert "sub1:" in content
    assert "sub2:" in content
    assert "gkb" in content # Global KB
    assert "kb1" in content # Agent KB
    assert "postgres" in content # Memory & KB
    assert "s3" in content # Global KB
    assert "mcp1" in content
    assert "p1" in content
    assert "entry_agent: sub1" in content
    assert "E: V" in content
    assert "t1" in content
    
    # Check pyproject.toml update
    pyproject_content = (builder.project_dir / "pyproject.toml").read_text()
    assert "oai-openai-agent-core" in pyproject_content

def test_binary_file_copy(temp_output_dir):
    """Test that binary files are skipped during rendering."""
    builder = ProjectBuilder(
        template="mcp",
        project_name="binary_test",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=False,
        items=[{"name": "test_server", "port": "8001", "description": "d", "tags": [], "source": "", "env": {}}]
    )
    
    # Mock project_dir to control rglob and avoid file system issues
    mock_project_dir = MagicMock()
    builder.project_dir = mock_project_dir
    
    # Setup files returned by rglob
    binary_file = MagicMock()
    binary_file.is_file.return_value = True
    binary_file.read_text.side_effect = UnicodeDecodeError("utf-8", b"", 0, 1, "fail")
    
    normal_file = MagicMock()
    normal_file.is_file.return_value = True
    # Use the actual constant placeholder to ensure replacement logic works
    normal_file.read_text.return_value = TOKEN_AUTHOR
    
    mock_project_dir.rglob.return_value = [binary_file, normal_file]
    mock_project_dir.exists.return_value = True
    mock_project_dir.relative_to.return_value = Path("rel/path")
    
    # Mock other steps to isolate _render_files
    with patch.object(builder, '_validate'), \
         patch.object(builder, '_copy_template'), \
         patch.object(builder, '_setup_mcp_servers'), \
         patch.object(builder, '_setup_agents'), \
         patch.object(builder, '_update_dependencies'):
        
        builder.build()
        
        # Verify behavior
        # Binary file should be read but not written (exception caught)
        binary_file.read_text.assert_called()
        binary_file.write_text.assert_not_called()
        
        # Normal file should be read and written with replaced content
        normal_file.read_text.assert_called()
        # Should be replaced with "A" (builder.author)
        normal_file.write_text.assert_called_with("A", encoding="utf-8")

def test_git_init_failure(temp_output_dir):
    """Test git init failure handling."""
    builder = ProjectBuilder(
        template="mcp",
        project_name="git_fail",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir,
        init_git=True,
        create_venv=False,
        items=[{"name": "test_server", "port": "8001", "description": "d", "tags": [], "source": "", "env": {}}]
    )
    
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
        # Should not raise exception, just log warning
        builder.build()

def test_venv_create_failure(temp_output_dir):
    """Test venv creation failure handling."""
    builder = ProjectBuilder(
        template="mcp",
        project_name="venv_fail",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir,
        init_git=False,
        create_venv=True,
        items=[{"name": "test_server", "port": "8001", "description": "d", "tags": [], "source": "", "env": {}}]
    )
    
    with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "python")):
        # Should not raise exception
        builder.build()

def test_mcp_build_simple_string_items(temp_output_dir):
    """Test MCP build with items as simple strings (coverage for else branch)."""
    items = ["simple_server"]
    
    builder = ProjectBuilder(
        template="mcp",
        project_name="simple_mcp",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir,
        items=items,
        init_git=False,
        create_venv=False
    )
    
    builder.build()
    
    # Check if yaml was created (copy from template)
    # The file mcp_registry_servers/servers_config/simple_server.yaml should exist
    assert (builder.project_dir / "mcp_registry_servers" / "servers_config" / "simple_server.yaml").exists()
    assert (builder.project_dir / "mcp_registry_servers" / "tools" / "simple_server.py").exists()

def test_agent_build_simple_string_items(temp_output_dir):
    """Test Agent build with items as simple strings (coverage for else branch)."""
    items = ["simple_agent"]
    
    builder = ProjectBuilder(
        template="agent",
        project_name="simple_agent",
        author="A",
        email="a@b.c",
        description="D",
        output_dir=temp_output_dir,
        items=items,
        framework="langgraph",
        init_git=False,
        create_venv=False
    )
    
    builder.build()
    
    # Check if yaml was created
    assert (builder.project_dir / "agentic_registry_agents" / "agents_config" / "simple_agent.yaml").exists()
