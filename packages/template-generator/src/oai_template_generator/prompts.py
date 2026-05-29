"""Interactive prompts for the CLI."""

import re
import sys
from oai_template_generator.constants import FRAMEWORK_PATTERNS, MODEL_OPTIONS, AWS_REGIONS


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
    """Present a numbered menu and return the chosen value.

    Accepts either the item number (1-N) or the value itself so that
    automated tests and piped inputs can pass the actual string directly.
    """
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        print(f"  {i}) {choice}")
    while True:
        raw = _ask("Enter number or value")
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:
            return raw
        print(f"  ⚠  Please enter a number (1–{len(choices)}) or a valid value.")


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
            email = _ask("Email address", default="user@example.com")
            if "@" in email and "." in email.split("@")[-1]:
                break
            print("  ⚠  Please enter a valid email address.")

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
                structured_output_model = _ask("    Structured Output Model Name (optional)")
                sub_agents = [{"name": item, "context": [], "knowledge_base": [], "structured_output_model": structured_output_model}]
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
                        kb_type = _choose("        Vector Store Type", ["chroma", "postgres", "s3", "pinecone"])
                        kb_list.append({"name": kb_name, "description": kb_desc, "type": kb_type})
                    
                    structured_output_model = _ask("      Structured Output Model Name (optional)")
                    sub_agents.append({"name": sub, "context": ctx, "knowledge_base": kb_list, "structured_output_model": structured_output_model})
            
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
            model_choice = _choose("    Select a Model ID", MODEL_OPTIONS)
            if model_choice == "Custom...":
                model_id = _ask("    Enter Custom Model ID")
            else:
                model_id = model_choice
                
            region = _choose("    Select AWS Region", AWS_REGIONS)
            
            # Additional Capabilities
            use_tools = confirm("  Will this agent use tools?", default=False)
            tool_list = []
            if use_tools:
                tools_input = _ask("    List of tools (comma-separated)")
                tool_list = [t.strip() for t in tools_input.split(",") if t.strip()]
            
            # Skills
            skill_list = []
            use_skill_registry = False
            skill_registry_url = ""
            skill_registry_token = ""
            skill_registry_items = []
            if confirm("  Will this agent use skills?", default=False):
                # Leaving blank triggers Skills Registry mode; a list uses local skills
                skills_input = _ask("    Skill names (comma-separated), or leave blank to configure Skills Registry")
                if skills_input.strip():
                    skill_list = [s.strip() for s in skills_input.split(",") if s.strip()]
                else:
                    use_skill_registry = True
                    skill_registry_url = _ask("    Skills Registry URL", default="${SKILLS_REGISTRY_URL}")
                    skill_registry_token = _ask("    Skills Registry Token (env var)", default="${SKILLS_REGISTRY_TOKEN}")
                    print("    Add skills from registry (empty name to finish):")
                    while True:
                        skill_name = _ask("      Skill name (or Enter to finish)")
                        if not skill_name:
                            break
                        skill_version = _ask(f"      Version for '{skill_name}'", default="latest")
                        skill_registry_items.append({"name": skill_name, "version": skill_version})

            # MCP Servers
            mcp_servers = []
            if confirm("  Will this agent use MCP servers?", default=False):
                mcp_input = _ask("    List of MCP servers (comma-separated)")
                mcp_names = [s.strip() for s in mcp_input.split(",") if s.strip()]
                for mcp_name in mcp_names:
                    print(f"      Configuration for MCP '{mcp_name}':")
                    mcp_type = _choose("      Type", ["stdio", "remote"])
                    mcp_config = {"name": mcp_name, "type": mcp_type}
                    if mcp_type == "stdio":
                        mcp_config["command"] = _ask("        Command", default="python")
                        args_input = _ask("        Args (comma-separated)", default="")
                        mcp_config["args"] = [a.strip() for a in args_input.split(',')] if args_input else []
                        env_input = _ask("        Env Vars (key=value, comma-separated)", default="")
                        env_dict = {}
                        if env_input:
                            for pair in env_input.split(','):
                                if '=' in pair:
                                    key, value = pair.split('=', 1)
                                    env_dict[key.strip()] = value.strip()
                        mcp_config["env"] = env_dict
                    else: # remote
                        mcp_config["url"] = _ask("        URL")
                        headers_input = _ask("        Headers (key=value, comma-separated)", default="")
                        headers_dict = {}
                        if headers_input:
                            for pair in headers_input.split(','):
                                if '=' in pair:
                                    key, value = pair.split('=', 1)
                                    headers_dict[key.strip()] = value.strip()
                        mcp_config["headers"] = headers_dict
                    mcp_servers.append(mcp_config)

            # MCP lazy loading
            use_lazy_loading = False
            if mcp_servers:
                use_lazy_loading = confirm("  Enable MCP Lazy Loading? (recommended for 3+ servers)", default=len(mcp_servers) >= 3)

            # Global Memory
            memory_config = {}
            if confirm("  Enable Memory (Conversation History)?", default=False):
                print("    Memory Configuration:")
                mem_type = _choose("    Vector Store Type", ["chroma", "postgres", "s3"])
                mem_collection = _ask("    Collection Name", default="chat_memory")
                memory_config = {"type": mem_type, "collection_name": mem_collection}

            # Global Knowledge Base
            global_kb = []
            use_kb_registry = False
            kb_registry_url = ""
            kb_registry_token = ""
            kb_registry_items = []
            if confirm("  Enable Knowledge Base?", default=False):
                # Leaving KB name blank triggers KB Registry mode
                kb_name = _ask("    Knowledge Base name (or leave blank to configure KB Registry)")
                if kb_name.strip():
                    kb_desc = _ask("    Description", default="Global document search")
                    kb_type = _choose("    Vector Store Type", ["chroma", "postgres", "s3", "pinecone"])
                    global_kb.append({"name": kb_name, "description": kb_desc, "type": kb_type})
                else:
                    use_kb_registry = True
                    kb_registry_url = _ask("    KB Registry URL", default="${KB_REGISTRY_URL}")
                    kb_registry_token = _ask("    KB Registry Token (env var)", default="${KB_REGISTRY_TOKEN}")
                    print("    Add KB sources from registry (empty name to finish):")
                    while True:
                        reg_name = _ask("      Registry KB name (or Enter to finish)")
                        if not reg_name:
                            break
                        reg_desc = _ask(f"      Description for '{reg_name}'", default="Knowledge base")
                        top_k = _ask(f"      top_k", default="5")
                        score_threshold = _ask(f"      score_threshold", default="0.7")
                        kb_registry_items.append({
                            "registry_name": reg_name,
                            "description": reg_desc,
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                        })
            
            # Agent-level KB for single agent case (if not already handled in sub_agents loop above)
            if len(sub_agents) == 1 and sub_agents[0]["name"] == item:
                 if confirm(f"  Enable Knowledge Base for '{item}'?", default=False):
                    kb_name = _ask("    Knowledge Base Name", default="agent_kb")
                    kb_desc = _ask("    Description", default="Agent specific documents")
                    kb_type = _choose("    Vector Store Type", ["chroma", "postgres", "s3", "pinecone"])
                    sub_agents[0]["knowledge_base"].append({"name": kb_name, "description": kb_desc, "type": kb_type})

            # Guardrails
            use_guardrails = confirm("  Enable Guardrails?", default=False)
            
            # Global Structured Output
            global_structured_output_model = None
            if confirm("  Enable Global Structured Output?", default=False):
                global_structured_output_model = _ask("    Global Structured Output Model Name")

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
                "skill_list": skill_list,
                "mcp_servers": mcp_servers,
                "sub_agents": sub_agents,
                "entry_agent": entry_agent,
                "global_kb": global_kb,
                "memory_config": memory_config,
                "use_guardrails": use_guardrails,
                "global_structured_output_model": global_structured_output_model,
                "tags": [t.strip() for t in tags.split(",") if t.strip()],
                "prompts": prompts,
                "env": env_vars,
                "use_skill_registry": use_skill_registry,
                "skill_registry_url": skill_registry_url,
                "skill_registry_token": skill_registry_token,
                "skill_registry_items": skill_registry_items,
                "use_kb_registry": use_kb_registry,
                "kb_registry_url": kb_registry_url,
                "kb_registry_token": kb_registry_token,
                "kb_registry_items": kb_registry_items,
                "use_lazy_loading": use_lazy_loading,
            })

    print()
    return template, slug, author, email, output_dir, description, items, framework
