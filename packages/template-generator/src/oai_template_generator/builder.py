"""Project builder — copies and renders the chosen template."""

import logging
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Union

from oai_template_generator.constants import (
    TEMPLATES_DIR,
    TOKEN_PROJECT_NAME,
    TOKEN_AUTHOR,
    TOKEN_EMAIL,
    TOKEN_DESCRIPTION,
    TOKEN_TITLE,
    FRAMEWORK_INFO,
    AGENT_CORE_DEPS,
    AGENT_CORE_SUBDIRS,
    VECTOR_STORE_EXTRAS,
    TOKEN_AGENT_CORE_PYPROJECT,
    TOKEN_AGENT_CORE_REQUIREMENTS,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ProjectBuilder:
    """Handles the scaffolding of new projects based on templates."""

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
        items: Union[List[Dict], List[str], None] = None,
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

    def build(self) -> None:
        """Executes the full build process."""
        try:
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
            
        except Exception as e:
            logger.error(f"\n❌ Build failed: {e}")
            # Clean up partial build if not overwriting existing valid project
            if self.project_dir.exists() and not self.overwrite:
                # Be careful not to delete user's existing work if we didn't start fresh
                pass 
            sys.exit(1)

    def _validate(self) -> None:
        """Validates inputs before starting build."""
        if not self.template_dir.exists():
            raise FileNotFoundError(f"Template '{self.template}' not found at {self.template_dir}")

        if self.project_dir.exists() and not self.overwrite:
            logger.error(f"❌  Directory '{self.project_dir}' already exists.")
            sys.exit(1)

        if not self.project_name:
            logger.error("❌  Project name cannot be empty.")
            sys.exit(1)

    def _copy_template(self) -> None:
        """Copies the base template to the target directory."""
        logger.info(f"📁  Creating project at {self.project_dir} …")
        if self.project_dir.exists() and self.overwrite:
            shutil.rmtree(self.project_dir)
            
        shutil.copytree(self.template_dir, self.project_dir)

        # Rename the inner src package directory if it exists
        placeholder_pkg = self.project_dir / "src" / "project_name"
        if placeholder_pkg.exists():
            placeholder_pkg.rename(self.project_dir / "src" / self.project_name)

    def _setup_mcp_servers(self) -> None:
        """Setup specific MCP servers if template is mcp."""
        mcp_registry_servers = self.project_dir / "mcp_registry_servers"
        servers_dir = mcp_registry_servers / "servers"
        servers_config_dir = mcp_registry_servers / "servers_config"
        tools_dir = mcp_registry_servers / "tools"
        
        # Ensure tools dir exists
        tools_dir.mkdir(exist_ok=True)
        (tools_dir / "__init__.py").touch()
        
        template_yaml = servers_config_dir / "template_server.yaml"
        server_template_file = mcp_registry_servers / "server.template"
        src_servers_template = self.template_dir / "mcp_registry_servers" / "servers"

        for server_config in self.items:
            server_name = server_config["name"] if isinstance(server_config, dict) else server_config
            target_server_dir = servers_dir / server_name
            
            # 1. Copy base server files
            shutil.copytree(src_servers_template, target_server_dir, dirs_exist_ok=True)
            
            # Determine class names
            if isinstance(server_config, dict):
                tools_class_name = server_config.get("class_name")
                if not tools_class_name:
                    tools_class_name = "".join(word.capitalize() for word in server_name.split("_")) 
                    if tools_class_name.endswith("Server"):
                        tools_class_name = tools_class_name[:-6] + "Tools"
                    else:
                        tools_class_name += "Tools"
            else:
                tools_class_name = "".join(word.capitalize() for word in server_name.split("_")) + "Tools"

            tool_file_name = server_name
            
            # 2. Create the Tools class file
            tool_file_path = tools_dir / f"{tool_file_name}.py"
            tool_content = f'''"""Tools for {server_name}."""

class {tools_class_name}:
    """
    Tools for {server_name}.
    """
    
    def example_tool(self, query: str) -> str:
        """
        An example tool function.
        
        Args:
            query (str): The query string.
            
        Returns:
            str: The result.
        """
        return f"Processed {{query}} by {server_name}"

    def status_check(self) -> dict:
        """
        Check status.
        
        Returns:
            dict: Status info.
        """
        return {{"status": "ok", "server": "{server_name}"}}
'''
            tool_file_path.write_text(tool_content, encoding="utf-8")

            # 3. Process server.py from template
            target_server_py = target_server_dir / "server.py"
            if server_template_file.exists():
                content = server_template_file.read_text(encoding="utf-8")
                
                server_class_name = "".join(word.capitalize() for word in server_name.split("_"))
                if not server_class_name.endswith("Server"):
                    server_class_name += "Server"
                
                content = content.replace("{server_name}", server_class_name)
                content = content.replace("{tool_file_name}", tool_file_name)
                content = content.replace("{tools_class_name}", tools_class_name)
                
                target_server_py.write_text(content, encoding="utf-8")

            # 4. Create config yaml
            if isinstance(server_config, dict):
                yaml_content = []
                yaml_content.append(f"port: {server_config['port']}")
                yaml_content.append(f"description: |\n  {server_config['description']}")
                
                if server_config["tags"]:
                    yaml_content.append("tags:")
                    for tag in server_config["tags"]:
                        yaml_content.append(f"  - {tag}")
                
                if server_config["source"]:
                    yaml_content.append(f"source: {server_config['source']}")
                    
                yaml_content.append("env: {}")
                if server_config.get("env"):
                    yaml_content[-1] = "env:"
                    for k, v in server_config["env"].items():
                        yaml_content.append(f"  {k}: {v}")
                
                new_yaml = servers_config_dir / f"{server_name}.yaml"
                new_yaml.write_text("\n".join(yaml_content) + "\n", encoding="utf-8")
            else:
                if template_yaml.exists():
                    new_yaml = servers_config_dir / f"{server_name}.yaml"
                    shutil.copy2(template_yaml, new_yaml)

        # Cleanup templates
        if template_yaml.exists():
            template_yaml.unlink()
        if server_template_file.exists():
            server_template_file.unlink()

    def _setup_agents(self) -> None:
        """Setup specific Agents if template is agent."""
        agentic_registry_agents = self.project_dir / "agentic_registry_agents"
        agents_dir = agentic_registry_agents / "agents"
        agents_config_dir = agentic_registry_agents / "agents_config"
        
        utils_dir = agentic_registry_agents / "utils"
        utils_dir.mkdir(exist_ok=True)
        (utils_dir / "__init__.py").touch()
        
        server_template_dir = agents_dir / "server_template"
        template_yaml = agents_config_dir / "template.yaml"
        
        info = FRAMEWORK_INFO.get(self.framework, FRAMEWORK_INFO["langgraph"])

        for agent_config in self.items:
            agent_name = agent_config["name"] if isinstance(agent_config, dict) else agent_config
            target_agent_dir = agents_dir / agent_name
            shutil.copytree(server_template_dir, target_agent_dir, dirs_exist_ok=True)
            
            # Process templates
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

            # Generate configuration
            if isinstance(agent_config, dict):
                self._generate_agent_utils(agent_config, agent_name, utils_dir)
                self._generate_agent_yaml(agent_config, agent_name, agents_config_dir)
            else:
                # Fallback for strings
                if template_yaml.exists():
                    new_yaml = agents_config_dir / f"{agent_name}.yaml"
                    shutil.copy2(template_yaml, new_yaml)

        # Cleanup
        if server_template_dir.exists():
            shutil.rmtree(server_template_dir)
        if template_yaml.exists():
            template_yaml.unlink()

    def _generate_agent_utils(self, config: Dict, agent_name: str, utils_dir: Path) -> None:
        """Generates utility file for agent tools."""
        tool_list = config.get("tool_list", [])
        if tool_list:
            utils_filename = f"{agent_name}_utils.py"
            utils_content = '"""Tools utility functions."""\n\n'
            for tool in tool_list:
                utils_content += f'''
def {tool}():
    """Dummy implementation for {tool}."""
    return "Executed {tool}"
'''
            (utils_dir / utils_filename).write_text(utils_content, encoding="utf-8")

    def _generate_agent_yaml(self, config: Dict, agent_name: str, config_dir: Path) -> None:
        """Generates agent configuration YAML."""
        yaml_content = []
        yaml_content.append("active: true")
        yaml_content.append(f"name: {agent_name}")
        yaml_content.append(f"description: |\n  {config['description']}")
        yaml_content.append(f"type: {self.framework or 'langgraph'}")
        yaml_content.append("cloud_provider: aws")
        yaml_content.append(f"port: {config['port']}")
        yaml_content.append("")
        
        sub_agents = config.get("sub_agents", [])
        mcp_servers = config.get("mcp_servers", [])
        global_kb = config.get("global_kb", [])
        memory_config = config.get("memory_config", {})
        use_guardrails = config.get("use_guardrails", False)
        tool_list = config.get("tool_list", [])
        pattern = config.get("pattern", "single")
        entry_agent = config.get("entry_agent")
        
        # Helper for KB
        def generate_kb_section(kb_list, indent_level=0):
            lines = []
            indent = " " * indent_level
            for kb in kb_list:
                kb_type = kb.get("type", "chroma")
                lines.append(f"{indent}- name: {kb['name']}")
                lines.append(f"{indent}  description: \"{kb['description']}\"")
                lines.append(f"{indent}  vector_store:")
                lines.append(f"{indent}    type: {kb_type}")
                lines.append(f"{indent}    settings:")
                lines.append(f"{indent}      collection_name: \"{kb['name']}\"")
                lines.append(f"{indent}      persist_directory: \"./rag_db\"")
                
                if kb_type == "postgres":
                    lines.append(f"{indent}      # db_host: your-postgres-host.com")
                    lines.append(f"{indent}      # db_user: user")
                    lines.append(f"{indent}      # db_port: 5432")
                    lines.append(f"{indent}      # db_name: your_db")
                elif kb_type == "s3":
                    lines.append(f"{indent}      # bucket_name: your-bucket")
                    lines.append(f"{indent}      # region: us-east-1")
                    
                lines.append(f"{indent}  embedding:")
                lines.append(f"{indent}    model_id: \"bedrock/amazon.titan-embed-text-v1\"")
                lines.append(f"{indent}    region_name: \"us-west-2\"")
                lines.append(f"{indent}  data_sources:")
                lines.append(f"{indent}    - path: \"docs/sample.pdf\"")
                lines.append(f"{indent}  text_splitter:")
                lines.append(f"{indent}    type: \"recursive_character\"")
                lines.append(f"{indent}    chunk_size: 1000")
                lines.append(f"{indent}    chunk_overlap: 200")
                lines.append(f"{indent}  retrieval_settings:")
                lines.append(f"{indent}    top_k: 5")
                lines.append(f"{indent}    score_threshold: 0.7")
            return lines

        if len(sub_agents) > 1:
            yaml_content.append(f"instructions: |\n  {config['instructions']}")
            yaml_content.append("")
            
            if global_kb:
                yaml_content.append("# Global Knowledge Base")
                yaml_content.append("knowledge_base:")
                yaml_content.extend(generate_kb_section(global_kb, indent_level=2))
                yaml_content.append("")
            
            yaml_content.append("# Multi-agent configuration")
            yaml_content.append("agent_list:")
            for sub in sub_agents:
                sub_name = sub["name"]
                sub_ctx = sub["context"]
                sub_kb = sub.get("knowledge_base", [])
                
                yaml_content.append(f"  - {sub_name}:")
                yaml_content.append(f"      system_prompt: Prompt for {sub_name}")
                
                # Add tools for sub-agent
                if tool_list:
                    yaml_content.append("      tools:")
                    for t in tool_list:
                        yaml_content.append(f"        - {t}")
                else:
                    yaml_content.append("      tools: []")
                    
                if mcp_servers:
                    yaml_content.append("      mcps:")
                    for ms in mcp_servers:
                        yaml_content.append(f"        - {ms['name']}")
                else:
                    yaml_content.append("      mcps: []")
                
                if sub_ctx:
                    yaml_content.append("      context:")
                    for c in sub_ctx:
                        yaml_content.append(f"        - {c}")
                else:
                    yaml_content.append("      # context: [other_agent_name]")
                    
                if sub_kb:
                    yaml_content.append("      knowledge_base:")
                    yaml_content.extend(generate_kb_section(sub_kb, indent_level=8))

            yaml_content.append("")
        else:
            # Single agent
            yaml_content.append(f"instructions: |\n  {config['instructions']}")
            yaml_content.append("")
            
            if global_kb:
                yaml_content.append("# Global Knowledge Base")
                yaml_content.append("knowledge_base:")
                yaml_content.extend(generate_kb_section(global_kb, indent_level=2))
                yaml_content.append("")

            yaml_content.append("# Agent configuration")
            yaml_content.append("agent_list:")
            for sub in sub_agents:
                 sub_name = sub["name"]
                 sub_ctx = sub["context"]
                 sub_kb = sub.get("knowledge_base", [])
                 yaml_content.append(f"  - {sub_name}:")
                 yaml_content.append(f"      system_prompt: {config['instructions'].splitlines()[0] if config['instructions'] else 'Default Prompt'}")
                 
                 if tool_list:
                    yaml_content.append("      tools:")
                    for t in tool_list:
                        yaml_content.append(f"        - {t}")
                 else:
                    yaml_content.append("      tools: []")

                 if mcp_servers:
                    yaml_content.append("      mcps:")
                    for ms in mcp_servers:
                        yaml_content.append(f"        - {ms['name']}")
                 else:
                    yaml_content.append("      mcps: []")
                 
                 if sub_ctx:
                    yaml_content.append("      context:")
                    for c in sub_ctx:
                        yaml_content.append(f"        - {c}")
                 else:
                    yaml_content.append("      # context: [other_agent_name]")
                    
                 if sub_kb:
                    yaml_content.append("      knowledge_base:")
                    yaml_content.extend(generate_kb_section(sub_kb, indent_level=8))
                    
            yaml_content.append("")

        yaml_content.append("# Model configuration")
        yaml_content.append("model:")
        yaml_content.append(f"  model_id: {config['model_id']}")
        yaml_content.append(f"  region_name: {config['region']}")
        yaml_content.append("")

        if tool_list:
            yaml_content.append("# For tools configuration")
            yaml_content.append("tools:")
            utils_name_no_ext = f"{agent_name}_utils"
            yaml_content.append(f"  {utils_name_no_ext}:")
            yaml_content.append(f"    module: {utils_name_no_ext}")
            yaml_content.append(f"    base_path: ./utils")
            yaml_content.append("")
        else:
            yaml_content.append("# tools: {}")
            yaml_content.append("")

        yaml_content.append("# For MCP type agents")
        yaml_content.append("mcps: {}")
        if mcp_servers:
            yaml_content[-1] = "mcps:"
            for srv in mcp_servers:
                srv_name = srv["name"]
                srv_type = srv["type"]
                yaml_content.append(f"  {srv_name}:")
                if srv_type == "stdio":
                    yaml_content.append(f"    command: {srv['command']}")
                    yaml_content.append(f"    args: {srv['args']}")
                    yaml_content.append(f"    env: {srv.get('env', '{}')}")
                else: # remote
                    yaml_content.append(f"    url: {srv['url']}")
                    yaml_content.append(f"    headers: {srv.get('headers', '{}')}")
            yaml_content.append("")
        
        # Memory
        if memory_config:
            mem_type = memory_config.get("type", "chroma")
            mem_coll = memory_config.get("collection_name", "chat_memory")
            
            yaml_content.append("# Memory configuration")
            yaml_content.append("memory:")
            yaml_content.append("  vector_store:")
            yaml_content.append(f"    type: {mem_type}")
            yaml_content.append("    settings:")
            yaml_content.append(f"      collection_name: \"{mem_coll}\"")
            yaml_content.append("      persist_directory: \"./memory_db\"")
            
            if mem_type == "postgres":
                yaml_content.append(f"      # db_host: your-postgres-host.com")
                yaml_content.append(f"      # db_user: user")
                yaml_content.append(f"      # db_port: 5432")
                yaml_content.append(f"      # db_name: your_db")
            elif mem_type == "s3":
                yaml_content.append(f"      # bucket_name: your-bucket")
                yaml_content.append(f"      # region: us-east-1")
                
            yaml_content.append("  embedding:")
            yaml_content.append("    model_id: \"bedrock/amazon.titan-embed-text-v1\"")
            yaml_content.append("    region_name: \"us-west-2\"")
            yaml_content.append("  settings:")
            yaml_content.append("    max_recent_turns: 3")
            yaml_content.append("    max_relevant_turns: 3")
            yaml_content.append("    similarity_threshold: 0.6")
            yaml_content.append("")
        
        # Guardrails
        if use_guardrails:
            yaml_content.append("# Guardrails configuration")
            yaml_content.append("# NOTE: These are sample validators. Update the configuration as per your requirement.")
            yaml_content.append("# For more validators and documentation, check guardrails.ai")
            yaml_content.append("guardrails:")
            yaml_content.append("  enable_agent_validation: false")
            yaml_content.append("  custom_validators_dir: \"custom_guardrails\"")
            yaml_content.append("  validators:")
            yaml_content.append("    - name: competitor_check")
            yaml_content.append("      full_name: guardrails/competitor_check")
            yaml_content.append("      parameters:")
            yaml_content.append("        competitors: [ \"Apple\", \"Samsung\" ]")
            yaml_content.append("      on_fail: \"fix\"")
            yaml_content.append("")
            yaml_content.append("    - name: RestrictToTopic")
            yaml_content.append("      full_name: tryolabs/restricttotopic")
            yaml_content.append("      parameters:")
            yaml_content.append("        valid_topics: [ \"sports\" ]")
            yaml_content.append("")
            yaml_content.append("    - name: DetectPII")
            yaml_content.append("      full_name: guardrails/detect_pii")
            yaml_content.append("      parameters:")
            yaml_content.append("        pii_entities: [\"EMAIL_ADDRESS\", \"PHONE_NUMBER\"]")
            yaml_content.append("")
            yaml_content.append("    - name: profanity_free")
            yaml_content.append("      full_name: guardrails/profanity_free")
            yaml_content.append("")
            yaml_content.append("  input:")
            yaml_content.append("    validators:")
            yaml_content.append("      - ref: competitor_check")
            yaml_content.append("  output:")
            yaml_content.append("    validators:")
            yaml_content.append("      - ref: competitor_check")
            yaml_content.append("      - ref: RestrictToTopic")
            yaml_content.append("")

        # System prompt logic
        if len(sub_agents) > 1:
            yaml_content.append("system_prompt: |\n  " + config['instructions'].replace('\n', '\n  '))
            yaml_content.append("")
        
        if config['tags']:
            yaml_content.append("tags:")
            for tag in config['tags']:
                yaml_content.append(f"  - {tag}")
        
        yaml_content.append("env: {}")
        if config.get("env"):
            yaml_content[-1] = "env:"
            for k, v in config["env"].items():
                yaml_content.append(f"  {k}: {v}")
        
        if config['prompts']:
            yaml_content.append("prompts:")
            for p in config['prompts']:
                yaml_content.append(f"  - \"{p}\"")
        
        yaml_content.append("")
        yaml_content.append("crew_config:")
        yaml_content.append(f"  pattern: {pattern}")
        if entry_agent:
            yaml_content.append(f"  entry_agent: {entry_agent}")

        new_yaml = config_dir / f"{agent_name}.yaml"
        new_yaml.write_text("\n".join(yaml_content) + "\n", encoding="utf-8")

    def _update_dependencies(self) -> None:
        """Uncomment framework-specific dependencies in pyproject.toml and requirements.txt."""
        if not self.framework:
            return

        base_dep = AGENT_CORE_DEPS.get(self.framework)
        if not base_dep:
            return

        extras = set()
        for item in self.items:
            if not isinstance(item, dict):
                continue
            
            if item.get("memory_config"):
                extras.add("vector-required")
                mem_type = item["memory_config"].get("type")
                if mem_type in VECTOR_STORE_EXTRAS:
                    extras.add(VECTOR_STORE_EXTRAS[mem_type])
            
            if item.get("global_kb"):
                extras.add("vector-required")
                for kb in item["global_kb"]:
                    kb_type = kb.get("type")
                    if kb_type in VECTOR_STORE_EXTRAS:
                        extras.add(VECTOR_STORE_EXTRAS[kb_type])
            
            for sub in item.get("sub_agents", []):
                if sub.get("knowledge_base"):
                    extras.add("vector-required")
                    for kb in sub["knowledge_base"]:
                        kb_type = kb.get("type")
                        if kb_type in VECTOR_STORE_EXTRAS:
                            extras.add(VECTOR_STORE_EXTRAS[kb_type])
            
            if item.get("use_guardrails"):
                extras.add("guardrails")

        # Build dependency strings
        extras_str = ",".join(sorted(list(extras)))
        pyproject_dep = f'"{base_dep}[{extras_str}]"' if extras else f'"{base_dep}"'
        
        subdir = AGENT_CORE_SUBDIRS.get(self.framework)
        req_dep = f"{base_dep}[{extras_str}] @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit.git@main#subdirectory={subdir}" if extras else f"{base_dep} @ git+https://github.com/Capgemini-Innersource/ptr_oai_agent_development_kit.git@main#subdirectory={subdir}"

        # Update pyproject.toml
        pyproject_path = self.project_dir / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding="utf-8")
            content = content.replace(f'"{TOKEN_AGENT_CORE_PYPROJECT}"', pyproject_dep)
            pyproject_path.write_text(content, encoding="utf-8")

        # Update requirements.txt
        req_path = self.project_dir / "requirements.txt"
        if req_path.exists():
            content = req_path.read_text(encoding="utf-8")
            content = content.replace(TOKEN_AGENT_CORE_REQUIREMENTS, req_dep)
            req_path.write_text(content, encoding="utf-8")

    def _render_files(self):
        """Walk every file and replace placeholder tokens."""
        title = self.project_name.replace("_", " ").replace("-", " ").title()

        replacements = {
            TOKEN_PROJECT_NAME: self.project_name,
            TOKEN_AUTHOR: self.author,
            TOKEN_EMAIL: self.email,
            TOKEN_DESCRIPTION: self.description,
            TOKEN_TITLE: title,
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
        logger.info("🔧  Initialising git repository …")
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
            logger.warning("  ⚠  git not available — skipping.")

    def _create_venv(self):
        logger.info("🐍  Creating virtual environment (.venv) …")
        try:
            subprocess.run(
                [sys.executable, "-m", "venv", ".venv"],
                cwd=self.project_dir,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            logger.warning("  ⚠  Could not create venv — skipping.")

    def _print_success(self):
        venv_activate = (
            "source .venv/bin/activate  # or .venv\\Scripts\\activate on Windows"
        )
        logger.info(f"\n✅  Project '{self.project_name}' created successfully!\n")
        logger.info("  Next steps:\n")
        try:
            rel = self.project_dir.relative_to(Path.cwd())
        except ValueError:
            rel = self.project_dir
        logger.info(f"    cd {rel}")
        logger.info(f"    {venv_activate}")
        logger.info("    pip install -e .[dev]")
        
        if self.template == "mcp":
            logger.info("\n  🛠  MCP Setup Instructions:")
            logger.info("    1. Update 'pyproject.toml' and 'requirements.txt' with any additional dependencies.")
            logger.info("    2. Add necessary utility files in 'mcp_registry_servers/tools/'.")
            logger.info("    3. Test your MCP server using the MCP Inspector or by running:")
            logger.info("       python -m mcp_registry_servers.server")
        elif self.template == "agent":
            logger.info("\n  🤖 Agent Setup Instructions:")
            logger.info(f"    1. Framework selected: {self.framework}")
            logger.info("    2. Update 'pyproject.toml' and 'requirements.txt' with any additional dependencies.")
            logger.info("    3. Update the agent configuration in 'agentic_registry_agents/agents_config/<agent_name>.yaml'.")
            logger.info("    4. Implement your agent logic in 'agentic_registry_agents/agents/<agent_name>/agent.py'.")
            logger.info("    5. Run your agent server:")
            logger.info("       python -m agentic_registry_agents.server")
        logger.info("")
