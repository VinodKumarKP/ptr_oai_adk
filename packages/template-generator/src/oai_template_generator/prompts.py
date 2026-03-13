"""Interactive prompts for the CLI."""

import re
import sys
from oai_template_generator.constants import FRAMEWORK_PATTERNS


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
) -> tuple[str, str, str, str, str, str, list[dict] | list[str], str | None]:
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
        raw_items = [s.strip() for s in servers_input.split(",") if s.strip()]
        
        for i, item in enumerate(raw_items):
            if not item.endswith("_server"):
                item = f"{item}_server"
            
            print(f"\nConfiguration for '{item}':")
            port = _ask(f"  Port for {item}", default=str(8000 + i))
            
            # Default class name derived from server name (e.g. env_lookup_server -> EnvLookupTools)
            default_cls = "".join(word.capitalize() for word in item.split("_"))
            if default_cls.endswith("Server"):
                default_cls = default_cls[:-6] + "Tools"
            else:
                default_cls += "Tools"
                
            class_name = _ask(f"  Tools Class Name", default=default_cls)
            desc = _ask(f"  Description for {item}", default="MCP server")
            tags = _ask(f"  Tags (comma-separated)", default="mcp")
            source = _ask(f"  Source URL", default="")
            
            env_vars = {}
            if confirm("  Add environment variables?", default=False):
                print("  Enter environment variables (key=value). Empty line to finish.")
                while True:
                    pair = _ask("    KEY=VALUE")
                    if not pair:
                        break
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        env_vars[k.strip()] = v.strip()
            
            items.append({
                "name": item,
                "class_name": class_name,
                "port": port,
                "description": desc,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "source": source,
                "env": env_vars
            })

    elif template == "agent":
        agents_input = _ask("List of Agents (comma-separated)", default="default_agent")
        raw_items = [s.strip() for s in agents_input.split(",") if s.strip()]
        # Ensure each agent name ends with _agent
        temp_items = []
        for item in raw_items:
            if not item.endswith("_agent"):
                item = f"{item}_agent"
            temp_items.append(item)
            
        framework = _choose("Select a framework:", ["langgraph", "crewai", "strands", "openai"])
        
        # Now collect configuration for each agent
        items = []
        for i, item in enumerate(temp_items):
            print(f"\nConfiguration for '{item}':")
            port = _ask(f"  Port for {item}", default=str(8000 + i))
            desc = _ask(f"  Description", default="An AI agent")
            
            # Pattern Selection
            patterns = FRAMEWORK_PATTERNS.get(framework, ["single"])
            pattern = _choose(f"  Select Pattern for '{framework}'", patterns)
            
            # Agent List (Sub-agents)
            sub_agents_input = _ask("  List of sub-agents (comma-separated). Leave empty for single agent", default="")
            raw_sub_names = [s.strip() for s in sub_agents_input.split(",") if s.strip()]
            
            sub_agents = []
            if not raw_sub_names:
                # Default to self if empty - Single Agent
                sub_agents = [{"name": item, "context": [], "knowledge_base": []}]
            else:
                # Multi-agent
                for sub in raw_sub_names:
                    print(f"    Configuration for sub-agent '{sub}':")
                    ctx_input = _ask(f"      Context (comma-separated). Leave empty for placeholder", default="")
                    ctx = [c.strip() for c in ctx_input.split(",") if c.strip()]
                    
                    kb_list = []
                    if confirm(f"      Enable Knowledge Base for '{sub}'?", default=False):
                        kb_name = _ask("        Knowledge Base Name", default="docs_kb")
                        kb_desc = _ask("        Description", default="Document search")
                        kb_type = _choose("        Vector Store Type", ["chroma", "postgres", "s3"])
                        kb_list.append({"name": kb_name, "description": kb_desc, "type": kb_type})
                    
                    sub_agents.append({"name": sub, "context": ctx, "knowledge_base": kb_list})
            
            # Entry Agent for Multi-agent
            entry_agent = None
            if len(sub_agents) > 1:
                # Ask for entry agent, defaulting to the first sub-agent
                default_entry = sub_agents[0]["name"]
                entry_agent = _ask(f"  Entry Agent (starts the interaction)", default=default_entry)
                
                instructions = _ask(f"  System Prompt for Supervisor")
            else:
                instructions = _ask(f"  System Prompt for Agent")

            # Model configuration
            print("  Model Configuration:")
            # Guided Model Selection
            model_options = [
                "anthropic.claude-3-5-sonnet-20240620-v1:0",
                "anthropic.claude-3-sonnet-20240229-v1:0",
                "anthropic.claude-3-haiku-20240307-v1:0",
                "meta.llama3-70b-instruct-v1:0",
                "Custom..."
            ]
            model_choice = _choose("    Select a Model ID", model_options)
            if model_choice == "Custom...":
                model_id = _ask("    Enter Custom Model ID")
            else:
                model_id = model_choice
                
            region = _ask("    AWS Region", default="us-east-1")
            
            # Additional Capabilities
            use_tools = confirm("  Will this agent use tools?", default=False)
            tool_list = []
            if use_tools:
                tools_input = _ask("    List of tools (comma-separated)")
                tool_list = [t.strip() for t in tools_input.split(",") if t.strip()]
            
            # MCP Servers
            mcp_servers = []
            if confirm("  Will this agent use MCP servers?", default=False):
                mcp_input = _ask("    List of MCP servers (comma-separated)")
                mcp_servers = [s.strip() for s in mcp_input.split(",") if s.strip()]
            
            # Global Memory
            memory_config = {}
            if confirm("  Enable Memory (Conversation History)?", default=False):
                print("    Memory Configuration:")
                mem_type = _choose("    Vector Store Type", ["chroma", "postgres", "s3"])
                mem_collection = _ask("    Collection Name", default="chat_memory")
                memory_config = {"type": mem_type, "collection_name": mem_collection}

            # Global Knowledge Base
            global_kb = []
            if confirm("  Enable Global Knowledge Base (shared)?", default=False):
                kb_name = _ask("    Knowledge Base Name", default="global_kb")
                kb_desc = _ask("    Description", default="Global document search")
                kb_type = _choose("    Vector Store Type", ["chroma", "postgres", "s3"])
                global_kb.append({"name": kb_name, "description": kb_desc, "type": kb_type})
            
            # Agent-level KB for single agent case (if not already handled in sub_agents loop above)
            if len(sub_agents) == 1 and sub_agents[0]["name"] == item:
                 if confirm(f"  Enable Knowledge Base for '{item}'?", default=False):
                    kb_name = _ask("    Knowledge Base Name", default="agent_kb")
                    kb_desc = _ask("    Description", default="Agent specific documents")
                    kb_type = _choose("    Vector Store Type", ["chroma", "postgres", "s3"])
                    sub_agents[0]["knowledge_base"].append({"name": kb_name, "description": kb_desc, "type": kb_type})

            # Guardrails
            use_guardrails = confirm("  Enable Guardrails?", default=False)

            tags = _ask(f"  Tags (comma-separated)", default="agent")
            
            # Prompts
            prompts = []
            if confirm("  Add example prompts?", default=True):
                print("    Enter example prompts. Empty line to finish.")
                while True:
                    p = _ask("    Prompt")
                    if not p:
                        break
                    prompts.append(p)
            
            env_vars = {}
            if confirm("  Add environment variables?", default=False):
                print("    Enter environment variables (key=value). Empty line to finish.")
                while True:
                    pair = _ask("    KEY=VALUE")
                    if not pair:
                        break
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        env_vars[k.strip()] = v.strip()

            items.append({
                "name": item,
                "pattern": pattern,
                "port": port,
                "description": desc,
                "instructions": instructions,
                "model_id": model_id,
                "region": region,
                "use_tools": use_tools,
                "tool_list": tool_list,
                "mcp_servers": mcp_servers,
                "sub_agents": sub_agents,
                "entry_agent": entry_agent,
                "global_kb": global_kb,
                "memory_config": memory_config,
                "use_guardrails": use_guardrails,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "prompts": prompts,
                "env": env_vars
            })

    print()
    return template, slug, author, email, output_dir, description, items, framework
