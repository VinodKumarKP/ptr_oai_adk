"""Project builder — copies and renders the chosen template."""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


TEMPLATES_DIR = Path(__file__).parent / "templates"

# Placeholder token used inside template files
_TOKEN = "{{PROJECT_NAME}}"
_TOKEN_AUTHOR = "{{AUTHOR}}"
_TOKEN_EMAIL = "{{EMAIL}}"
_TOKEN_DESC = "{{DESCRIPTION}}"
_TOKEN_TITLE = "{{PROJECT_TITLE}}"


class ProjectBuilder:
    def __init__(
        self,
        template: str,
        project_name: str,
        author: str,
        email: str,
        description: str,
        output_dir: Path,
        init_git: bool = True,
        create_venv: bool = True,
        items: list[str] = None,
        framework: str = None,
        overwrite: bool = False,
    ):
        self.template = template
        self.project_name = project_name
        self.author = author
        self.email = email
        self.description = description
        self.output_dir = output_dir
        self.init_git = init_git
        self.create_venv = create_venv
        self.items = items or []
        self.framework = framework
        self.overwrite = overwrite

        self.template_dir = TEMPLATES_DIR / template
        self.project_dir = output_dir / project_name

    # ------------------------------------------------------------------
    def build(self):
        self._validate()
        self._copy_template()
        if self.template == "mcp":
            self._setup_mcp_servers()
        elif self.template == "agent":
            self._setup_agents()
            self._update_dependencies()
        self._render_files()
        if self.init_git:
            self._init_git()
        if self.create_venv:
            self._create_venv()
        self._print_success()

    # ------------------------------------------------------------------
    def _validate(self):
        if not self.template_dir.exists():
            print(f"❌  Template '{self.template}' not found at {self.template_dir}")
            sys.exit(1)

        if self.project_dir.exists() and not self.overwrite:
            print(f"❌  Directory '{self.project_dir}' already exists.")
            sys.exit(1)

        if not self.project_name:
            print("❌  Project name cannot be empty.")
            sys.exit(1)

    def _copy_template(self):
        print(f"📁  Creating project at {self.project_dir} …")
        if self.project_dir.exists() and self.overwrite:
            shutil.rmtree(self.project_dir)
            
        shutil.copytree(self.template_dir, self.project_dir)

        # Rename the inner src package directory if it exists
        placeholder_pkg = self.project_dir / "src" / "project_name"
        if placeholder_pkg.exists():
            placeholder_pkg.rename(self.project_dir / "src" / self.project_name)

    def _setup_mcp_servers(self):
        """Setup specific MCP servers if template is mcp."""
        mcp_registry_servers = self.project_dir / "mcp_registry_servers"
        servers_dir = mcp_registry_servers / "servers"
        servers_config_dir = mcp_registry_servers / "servers_config"
        
        # Template files
        template_yaml = servers_config_dir / "template_server.yaml"
        server_template_file = mcp_registry_servers / "server.template"

        # Original servers template directory (to copy __init__.py etc)
        src_servers_template = self.template_dir / "mcp_registry_servers" / "servers"

        for server_name in self.items:
            target_server_dir = servers_dir / server_name
            
            # 1. Create the server directory and copy base files
            shutil.copytree(src_servers_template, target_server_dir, dirs_exist_ok=True)
            
            # 2. Handle server.py from server.template
            target_server_py = target_server_dir / "server.py"
            if server_template_file.exists():
                content = server_template_file.read_text(encoding="utf-8")
                class_name = "".join(word.capitalize() for word in server_name.split("_")) + "Server"
                content = content.replace("{server_name}", class_name)
                content = content.replace("{tool_file_name}", server_name)
                content = content.replace("{tools_class_name}", class_name.replace("Server", "Tools"))
                target_server_py.write_text(content, encoding="utf-8")

            # 3. Create copy of template_server.yaml
            if template_yaml.exists():
                new_yaml = servers_config_dir / f"{server_name}.yaml"
                shutil.copy2(template_yaml, new_yaml)

        # Remove the template.yaml after creating server-specific configs
        if template_yaml.exists():
            template_yaml.unlink()
            
        # Remove the server.template file from the final project
        if server_template_file.exists():
            server_template_file.unlink()

    def _setup_agents(self):
        """Setup specific Agents if template is agent."""
        agentic_registry_agents = self.project_dir / "agentic_registry_agents"
        agents_dir = agentic_registry_agents / "agents"
        agents_config_dir = agentic_registry_agents / "agents_config"
        
        server_template_dir = agents_dir / "server_template"
        template_yaml = agents_config_dir / "template.yaml"
        
        framework_info = {
            "langgraph": {"core": "langgraph_core", "pkg": "langgraph_agent", "cls": "LangGraphAgent"},
            "crewai": {"core": "crewai_core", "pkg": "crewai_agent", "cls": "CrewAIAgent"},
            "strands": {"core": "aws_strands_core", "pkg": "aws_strands_agent", "cls": "StrandsAgent"},
            "openai": {"core": "openai_core", "pkg": "openai_agent", "cls": "OpenAIAgent"}
        }
        
        info = framework_info.get(self.framework, framework_info["langgraph"])

        for agent_name in self.items:
            target_agent_dir = agents_dir / agent_name
            shutil.copytree(server_template_dir, target_agent_dir, dirs_exist_ok=True)
            
            # Handle templates inside the agent directory
            agent_template_file = target_agent_dir / "agent.template"
            server_template_file = target_agent_dir / "server.template"
            
            agent_class_name = "".join(word.capitalize() for word in agent_name.split("_")) + "Agent"
            
            if agent_template_file.exists():
                content = agent_template_file.read_text(encoding="utf-8")
                content = content.replace("{{core_package_name}}", info["core"])
                content = content.replace("{{agent_package_name}}", info["pkg"])
                content = content.replace("{{agent_class_name}}", info["cls"])
                content = content.replace("{{AgentName}}", agent_class_name)
                (target_agent_dir / "agent.py").write_text(content, encoding="utf-8")
                agent_template_file.unlink()
                
            if server_template_file.exists():
                content = server_template_file.read_text(encoding="utf-8")
                content = content.replace("{{agent_class_file_name}}", agent_name)
                content = content.replace("{{AgentName}}", agent_class_name)
                (target_agent_dir / "server.py").write_text(content, encoding="utf-8")
                server_template_file.unlink()

            # Create agent_name.yaml from template.yaml
            if template_yaml.exists():
                new_yaml = agents_config_dir / f"{agent_name}.yaml"
                shutil.copy2(template_yaml, new_yaml)

        # Remove the template directory from the final project
        if server_template_dir.exists():
            shutil.rmtree(server_template_dir)
            
        # Remove the template.yaml after creating agent-specific configs
        if template_yaml.exists():
            template_yaml.unlink()

    def _update_dependencies(self):
        """Uncomment framework-specific dependencies in pyproject.toml and requirements.txt."""
        if not self.framework:
            return

        framework_map = {
            "langgraph": "oai-langgraph-core",
            "crewai": "oai-crewai-core",
            "strands": "oai-aws-strands-core",
            "openai": "oai-openai-core"
        }
        
        dep_name = framework_map.get(self.framework)
        if not dep_name:
            return

        # Update pyproject.toml
        pyproject_path = self.project_dir / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding="utf-8")
            pattern = rf'#\s*"{dep_name}",'
            if dep_name == "oai-langgraph-core":
                content = re.sub(r'#\s*"oai-langraph-core",', f'"{dep_name}",', content)
            
            content = re.sub(pattern, f'"{dep_name}",', content)
            pyproject_path.write_text(content, encoding="utf-8")

        # Update requirements.txt
        req_path = self.project_dir / "requirements.txt"
        if req_path.exists():
            content = req_path.read_text(encoding="utf-8")
            pattern = rf'#\s*({dep_name}\s*@\s*git\+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit)'
            content = re.sub(pattern, r'\1', content)
            req_path.write_text(content, encoding="utf-8")

    def _render_files(self):
        """Walk every file and replace placeholder tokens."""
        title = self.project_name.replace("_", " ").replace("-", " ").title()

        replacements = {
            _TOKEN: self.project_name,
            _TOKEN_AUTHOR: self.author,
            _TOKEN_EMAIL: self.email,
            _TOKEN_DESC: self.description,
            _TOKEN_TITLE: title,
        }

        for path in self.project_dir.rglob("*"):
            if path.is_file():
                try:
                    text = path.read_text(encoding="utf-8")
                    for token, value in replacements.items():
                        text = text.replace(token, value)
                    path.write_text(text, encoding="utf-8")
                except UnicodeDecodeError:
                    pass  # skip binary files

    def _init_git(self):
        print("🔧  Initialising git repository …")
        try:
            subprocess.run(
                ["git", "init", "-q"],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "add", "."],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
            )
            subprocess.run(
                ["git", "commit", "-q", "-m", "chore: initial project scaffold"],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  ⚠  git not available — skipping.")

    def _create_venv(self):
        print("🐍  Creating virtual environment (.venv) …")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", ".venv"],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            print("  ⚠  Could not create venv — skipping.")

    def _print_success(self):
        venv_activate = (
            "source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows"
        )
        print(f"\n✅  Project '{self.project_name}' created successfully!\n")
        print("  Next steps:\n")
        try:
            rel = self.project_dir.relative_to(Path.cwd())
        except ValueError:
            rel = self.project_dir
        print(f"    cd {rel}")
        print(f"    {venv_activate}")
        print("    pip install -e .[dev]")
        
        if self.template == "mcp":
            print("\n  🛠  MCP Setup Instructions:")
            print("    1. Update 'pyproject.toml' and 'requirements.txt' with any additional dependencies.")
            print("    2. Add necessary utility files in 'mcp_registry_servers/utils/'.")
            print("    3. Test your MCP server using the MCP Inspector or by running:")
            print("       python -m mcp_registry_servers.server")
        elif self.template == "agent":
            print("\n  🤖 Agent Setup Instructions:")
            print(f"    1. Framework selected: {self.framework}")
            print("    2. Update 'pyproject.toml' and 'requirements.txt' with any additional dependencies.")
            print("    3. Update the agent configuration in 'agentic_registry_agents/agents_config/<agent_name>.yaml'.")
            print("    4. Implement your agent logic in 'agentic_registry_agents/agents/<agent_name>/agent.py'.")
            print("    5. Run your agent server:")
            print("       python -m agentic_registry_agents.server")
        print()
