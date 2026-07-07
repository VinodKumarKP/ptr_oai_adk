import asyncio
import logging
import os
import threading
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Dict, Any, Optional, Type

from oai_agent_core.components.configuration.model_config import ConfigManager
from oai_agent_core.components.configuration.model_config import config_manager
from oai_agent_core.components.observability.langfuse_observability_manager import LangfuseObservabilityManager
from oai_agent_core.components.observability.tracing import trace_span, traced, configure_tracing
from oai_agent_core.components.observability.metrics import configure_metrics
from oai_agent_core.utils.a2ui_prompt import build_agui_instructions
from oai_agent_core.utils.dotenv_loader import load_dotenv
from oai_agent_core.components.output_parser.output_model_registry import OutputModelRegistry
from oai_agent_core.components.skills.skill_registry import SkillRegistry
from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager
from oai_agent_core.macros import MacroProcessor
from oai_agent_core.services import (
    ConfigResolverService,
    ModelService,
    ConfigValidator,
    SkillService,
    ToolService,
    KnowledgeBaseService,
    MemoryService,
    GuardrailsService,
)


class BaseAgent(ABC):
    """Abstract base class for all agent framework implementations.

    Provides unified interface and shared functionality across different agent frameworks
    (LangChain, CrewAI, AWS Strands, etc.). Handles:
    - Configuration loading and management (YAML-based)
    - LLM initialization and model configuration
    - Tool registry and dynamic tool loading (including MCP)
    - Knowledge base integration for semantic search
    - Conversation memory and context management
    - Input/output validation via guardrails
    - Observability and tracing (Langfuse integration)
    - Macro processing for prompt templating

    **Thread Safety**: Initialization is protected by a lock. Concurrent ainvoke()
    calls are safe. Subclasses should ensure tool execution is thread-safe.

    **Async Model**: Initialization is async-heavy for efficiency. Execution methods
    (ainvoke, astream) are abstract and must be implemented by subclasses.

    **Error Handling**: Non-critical component failures (KB, memory, guardrails) are
    logged but don't prevent agent initialization (graceful degradation).
    """

    def __init__(
            self,
            agent_name: str,
            agent_config: Optional[Dict[str, Any]] = None,
            llm: Any = None,
            session_id: str = "default",
            user_id: str = "default",
            config_root: str = None,
            agent_type: str = None,
            document_loader: Optional[Any] = None,
            vector_store: Optional[Any] = None,
            model_manager: Optional[BaseModelConfigurationManager] = None,
            **kwargs
    ):
        """Initialize the BaseAgent with configuration and optional components.

        **Configuration Loading**: If agent_config is not provided, loads from YAML file
        named {agent_name}.yaml in the config_root directory.

        **LLM Initialization**: Creates LLM from model_manager if not provided directly.
        The model configuration is taken from agent_config['model'].

        **Async Components**: Tool registry, knowledge base, memory store, and guardrails
        are initialized during initialize() call, not in __init__. This allows fast
        instantiation and parallel loading of heavy dependencies.

        Args:
            agent_name: Unique identifier for the agent (used for config file lookup,
                logging, and observability).
            agent_config: Dictionary containing agent configuration. If None, loaded from
                {config_root}/{agent_name}.yaml. Should contain at minimum:
                - 'type': Agent framework type (required)
                - 'model': Model configuration dict with 'name' and optional params
                - 'tools': (optional) Tool configurations
                - 'knowledge_base': (optional) KB configurations
                - 'memory': (optional) Memory store configuration
                - 'guardrails': (optional) Guardrails configuration
            llm: Pre-configured language model instance. If provided, overrides model
                creation from model_manager. Useful for testing or custom LLM setup.
            session_id: Session identifier for tracking multi-turn conversations. Used for
                memory context retrieval and observability. Default "default" groups all
                interactions without session separation.
            user_id: User identifier for tracking and personalization. Default "default".
                Used for memory store queries and observability.
            config_root: Root directory for YAML configuration files. If None, uses
                working directory. Relative paths in configs are resolved from here.
            agent_type: Agent framework type (e.g., 'crewai', 'langchain', 'bedrock').
                Can be overridden by agent_config['type'].
            document_loader: Optional custom document loader for knowledge base. If None,
                uses loader specified in KB config.
            vector_store: Optional custom vector store for knowledge base. If None,
                uses store specified in KB config.
            model_manager: Manager for LLM instantiation. Required if llm is None and
                agent_config['model'] is present.
            **kwargs: Additional keyword arguments are set as instance attributes. Useful
                for passing custom parameters to subclasses.

        Raises:
            ValueError: If agent_config is not provided and cannot be loaded from disk.
        """
        self.agent_name = agent_name
        self.config_manager = ConfigManager(config_root=config_root)
        self.config_root = config_root

        # Load {config_root}/.env into os.environ first (if present) so that
        # config macros (${VAR}), env-based settings, and the tracing/metrics
        # endpoints below all see those values. Existing env vars take precedence.
        load_dotenv(config_root)

        # Auto-configure tracing (if OTEL_EXPORTER_OTLP_ENDPOINT is set and the
        # host hasn't already set up a provider) using the agent name as the
        # service.name shown in Jaeger/Tempo. No-op otherwise.
        configure_tracing(default_service_name=agent_name)

        # Auto-configure Prometheus metrics. Reuses a host-provided MeterProvider
        # (e.g. the HTTP server's /metrics) or, when PROMETHEUS_ENABLED is set,
        # stands up a standalone exposition endpoint. No-op otherwise.
        configure_metrics(default_service_name=agent_name)

        # Initialize logger first for service use
        self.logger = logging.getLogger(__name__)

        # Phase 3.1: Service Layer Integration
        # Initialize service instances for cleaner architecture
        self._config_service = ConfigResolverService(
            config_root=config_root,
            logger=self.logger
        )
        self._config_validator = ConfigValidator(logger=self.logger)
        self._model_service = ModelService(logger=self.logger)
        if model_manager:
            self._model_service.set_model_manager(model_manager)
        
        # Phase 3.3: Initialize Tool and KB services for _load_tools_and_kb_and_memory()
        self._tool_service = ToolService(
            project_root=config_root,
            logger=self.logger
        )
        self._kb_service = KnowledgeBaseService(
            project_root=config_root,
            logger=self.logger
        )
        
        # Phase 3.6: Initialize Memory and Guardrails services
        self._memory_service = MemoryService(
            project_root=config_root,
            logger=self.logger
        )
        self._guardrails_service = GuardrailsService(
            project_root=config_root,
            logger=self.logger
        )

        # Load config from file if not provided
        # Phase 3.1: Use ConfigResolverService for loading
        if agent_config is None:
            try:
                # Try service-based loading first (Phase 3.1)
                agent_config = self._config_service.load_agent_config(agent_name)
            except Exception as e:
                # Fallback to direct ConfigManager (backward compatibility)
                self.logger.debug(f"Service-based config loading failed, using ConfigManager: {e}")
                agent_config = self.config_manager.load_agent_config(agent_name)

        # Phase 3.2: Validate configuration early (informational, not blocking)
        # Only validate if config is complete enough to be worthwhile
        if agent_config and 'type' in agent_config:
            if not self._config_validator.validate_agent_config(agent_config):
                # Log but don't block - graceful degradation
                error_count = len(self._config_validator.get_errors())
                if error_count > 0:
                    self.logger.debug(f"Configuration validation found {error_count} issues (not blocking initialization)")
        
        # Resolve macros in configuration
        agent_config = self._config_service.resolve_macros(agent_config)

        self.agent_config = agent_config

        # Assign LLM if provided, otherwise create from manager
        self.llm = llm
        if self.llm is None:
            self.model_manager = model_manager
            # Check if 'model' key exists in config, not just if value is truthy
            if 'model' in agent_config:
                model_config = agent_config.get('model')
                # Phase 3.1: Use ModelService for LLM creation
                # Skip validation for minimal configs (e.g., in tests)
                should_validate = bool(model_config.get('provider') or model_config.get('name')) if model_config else False
                
                try:
                    if should_validate and self._model_service.validate_model_config(model_config):
                        # Validation passed, create via service
                        self.llm = self._model_service.create_model(model_config)
                    elif not should_validate:
                        # Minimal config (e.g., test mock), skip validation and create directly
                        if self.model_manager:
                            self.llm = self.model_manager.create_model(model_config=model_config)
                    else:
                        # Validation failed but config present, try direct creation
                        self.logger.debug("Model config validation skipped or failed, using direct model_manager")
                        if self.model_manager:
                            self.llm = self.model_manager.create_model(model_config=model_config)
                except Exception as e:
                    # Fallback to direct model_manager (backward compatibility)
                    self.logger.debug(f"Service-based model creation failed, using model_manager: {e}")
                    if self.model_manager:
                        self.llm = self.model_manager.create_model(model_config=model_config)
                    else:
                        self.llm = None
            else:
                self.llm = None  # Handle case where model key not in config

        self.session_id = session_id
        self.user_id = user_id
        self.agent_type = agent_type
        self._initialized = False
        self._initialization_lock = threading.Lock()

        # Initialize any additional attributes from kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.langfuse_manager = LangfuseObservabilityManager(
            agent_name=self.agent_name,
            logger=self.logger,
            framework=self.agent_type
        )
        self.document_loader = document_loader
        self.vector_store = vector_store
        # Following attributes are initialized asynchronously in _load_tools_and_kb_and_memory()
        self.memory_store = None
        self.global_kb_factory = None
        self.tool_registry = None
        self.guardrails_manager = None

        # Initialize skill registry only if agent uses skills
        # (skill registry is optional - not all agents need it)
        self.skill_registry = None
        skills_config = agent_config.get('skills', {})
        if skills_config:
            # Phase 3.1: Use SkillService for registry creation
            skill_service = SkillService(
                project_root=config_root,
                logger=self.logger
            )
            
            # Create skill registry using service
            try:
                self.skill_registry = skill_service.create_skill_registry(skills_config)
                if self.skill_registry:
                    self.logger.debug("Skill registry initialized for this agent (via SkillService)")
            except Exception as e:
                # Fallback to direct SkillRegistry creation (backward compatibility)
                self.logger.debug(f"Service-based skill registry creation failed, using SkillRegistry: {e}")
                
                # Extract skill directory (required if skills configured)
                skill_dir = skills_config.get('skill_dir')

                # Extract remote registry config (optional)
                remote_registry_config = skills_config.get('registry', {})

                # Use skill_dir as cache directory for pulled skills
                skills_cache_dir = skill_dir

                # Initialize skill registry directly
                self.skill_registry = SkillRegistry(
                    logger=self.logger,
                    project_root=config_root,
                    registry_url=remote_registry_config.get('url', os.environ.get('SKILLS_REGISTRY_URL', None)),
                    auth_token=remote_registry_config.get('auth_token', os.environ.get('SKILLS_REGISTRY_AUTH_TOKEN', None)),
                    skills_cache_dir=skills_cache_dir
                )
                self.logger.debug("Skill registry initialized for this agent")

        self.output_model_registry = OutputModelRegistry(
            logger=self.logger,
            project_root=config_root
        )

    def assign_llm(self, model_config: Dict):
        """
        Assign model to the agent.
        :param model_config: Model configuration
        :return:
        """
        self.llm = self.model_manager.create_model(model_config=model_config)

    # ------------------------------------------------------------------
    # Knowledge-base config normalization helpers
    # ------------------------------------------------------------------

    def _enrich_agent_list_kb_configs(self) -> None:
        """Propagate top-level KB registry credentials into every agent's KB entries.

        Called once during :meth:`_load_tools_and_kb_and_memory`.  Walks through
        ``agent_list`` in ``self.agent_config`` and, for any agent whose
        ``knowledge_base`` is a flat list without per-entry credentials, injects
        the top-level ``knowledge_base.registry`` URL and token.

        This means **no builder changes are needed** — by the time the builder
        receives an agent config, each KB entry already has ``registry_url`` and
        ``auth_token`` set.  Per-entry values always win over the top-level block.

        Only runs when ``knowledge_base`` is a dict (new style).  Old-style list
        configs are left untouched for backward compatibility.
        """
        raw_kb = self.agent_config.get('knowledge_base')
        if not isinstance(raw_kb, dict):
            return  # old-style list or absent — nothing to propagate

        registry_config = raw_kb.get('registry', {})
        if not registry_config:
            return  # new-style dict but no registry block — nothing to propagate

        agent_list = self.agent_config.get('agent_list', [])
        for agent_dict in agent_list:
            if not isinstance(agent_dict, dict):
                continue
            for agent_config in agent_dict.values():
                if not isinstance(agent_config, dict):
                    continue
                kb_entries = agent_config.get('knowledge_base', [])
                if kb_entries and isinstance(kb_entries, list):
                    agent_config['knowledge_base'] = (
                        BaseAgent._merge_registry_into_sources(kb_entries, registry_config)
                    )

    @staticmethod
    def _parse_kb_config(raw_config) -> tuple:
        """Normalize the top-level ``knowledge_base`` value.

        Supports two YAML styles:

        **New style** (skills-like, recommended)::

            knowledge_base:
              registry:
                url: http://localhost:8085
                token: dummy-token
              sources:
                - name: insurance
                  description: "..."

        **Old style** (backward-compatible)::

            knowledge_base:
              - name: insurance
                registry_url: http://localhost:8085
                auth_token: dummy-token

        Returns:
            ``(sources_list, registry_config)`` where *sources_list* is the
            flat list of KB entry dicts and *registry_config* is the optional
            ``{url, token}`` dict (empty dict when using old style).
        """
        if isinstance(raw_config, dict):
            return raw_config.get('sources', []), raw_config.get('registry', {})
        if isinstance(raw_config, list):
            return raw_config, {}
        return [], {}

    @staticmethod
    def _merge_registry_into_sources(sources: list, registry_config: dict) -> list:
        """Inject top-level registry URL/token into each KB source entry.

        Per-entry ``registry_url`` / ``auth_token`` values take priority over
        the global registry block, preserving full backward-compatibility.

        Args:
            sources: Flat list of KB entry dicts.
            registry_config: ``{url, token}`` from the top-level
                ``knowledge_base.registry`` block (may be empty).

        Returns:
            New list of entry dicts with registry credentials merged in.
        """
        if not registry_config:
            return sources
        registry_url = registry_config.get('url', '')
        token = registry_config.get('token', '')
        result = []
        for entry in sources:
            e = dict(entry)
            if not e.get('registry_url') and registry_url:
                e['registry_url'] = registry_url
            if not e.get('auth_token') and token:
                e['auth_token'] = token
            result.append(e)
        return result

    # ------------------------------------------------------------------

    async def _load_tools_and_kb_and_memory(self, kb_factory_class: Optional[Type] = None) -> None:
        """Load tools, knowledge base, and memory concurrently.

        Initializes multiple agent subsystems in parallel for efficiency:
        1. Tools and MCP (Model Context Protocol) servers from configuration
        2. Global knowledge base for semantic search and context retrieval
        3. Memory store for conversation history and context management
        4. Guardrails for input/output validation
        5. Environment variables from configuration
        6. Agent skills discovery
        7. Structured output models
        8. System prompt augmentation with macro resolution

        **Concurrency Model**: Tasks run concurrently via asyncio.gather(), but state
        mutations are serialized to prevent race conditions. Thread-safe for multiple
        agents initializing simultaneously.

        Args:
            kb_factory_class: Optional KnowledgeBaseFactory class for KB initialization.
                If None, KB initialization is skipped.

        Raises:
            Exception: Individual component failures are logged but don't fail initialization.
                Graceful degradation allows agent to function without optional components.
        """
        if not hasattr(self, 'tool_registry'):
            self.logger.warning("Tool registry not initialized. Skipping tool loading.")
            return

        # Propagate top-level KB registry credentials into every agent's KB entries
        # so builders receive fully resolved configs with no per-entry credentials needed.
        self._enrich_agent_list_kb_configs()

        async def _load_tools_and_mcp():
            tools_config = self.agent_config.get('tools', {})
            if tools_config:
                with trace_span("agent.load_tools", agent_name=self.agent_name):
                    tool_count = self._tool_service.load_tools(self.tool_registry, tools_config)
                    self.logger.info(f"Loaded {tool_count} global tools into registry")

            mcp_config = self.agent_config.get('mcps', self.agent_config.get('servers', {}))
            if mcp_config:
                with trace_span("agent.load_mcp", agent_name=self.agent_name):
                    mcp_count = self._tool_service.load_mcp_tools(self.tool_registry, mcp_config)
                    self.logger.info(f"Loaded {mcp_count} MCP servers into registry")

        async def _init_global_kb():
            raw_kb = self.agent_config.get('knowledge_base')
            if not raw_kb or not kb_factory_class:
                return
            sources, registry_config = BaseAgent._parse_kb_config(raw_kb)
            if not sources:
                return
            resolved_sources = BaseAgent._merge_registry_into_sources(sources, registry_config)

            with trace_span("agent.load_kb", agent_name=self.agent_name):
                try:
                    self.global_kb_factory = await asyncio.to_thread(
                        self._kb_service.create_knowledge_base_factory,
                        kb_factory_class,
                        resolved_sources,
                        self.llm,
                        self.document_loader,
                        self.vector_store,
                    )
                    self.logger.info("Initialized global knowledge base")
                except ImportError as e:
                    self.logger.warning("Could not initialize global knowledge base: missing dependencies %s", e)
                except Exception as e:
                    self.logger.error("Failed to initialize global knowledge base: %s", e)

        async def _init_memory_store():
            memory_config = self.agent_config.get('memory', {})
            if memory_config:
                with trace_span("agent.load_memory", agent_name=self.agent_name):
                    self.memory_store = self._memory_service.create_memory_store(
                        memory_config=memory_config,
                        llm=self.llm
                    )
                    if self.memory_store:
                        self.logger.info("Initialized memory store")

        async def _init_guardrails():
            guardrails_config = self.agent_config.get('guardrails', {})
            if guardrails_config:
                with trace_span("agent.load_guardrails", agent_name=self.agent_name):
                    self.guardrails_manager = self._guardrails_service.create_guardrails_manager(
                        guardrails_config
                    )
                if not self.guardrails_manager:
                    return
                self.logger.info("Initialized guardrails manager")

                # Apply guardrails prompt to system prompt if validation is enabled
                if self.guardrails_manager and guardrails_config.get('enable_agent_validation', True):
                    if len(self.agent_config.get('agent_list', [])) > 1:
                        self.agent_config[
                            'system_prompt'] = f"{self.agent_config.get('system_prompt', '')}\n{self.guardrails_manager.get_guardrails_prompt()}"
                    elif len(self.agent_config.get('agent_list', [])) == 1:
                        agent_name = list((self.agent_config['agent_list'][0]).keys())[0]
                        self.agent_config['agent_list'][0][agent_name][
                            'system_prompt'] = f"{self.agent_config['agent_list'][0][agent_name].get('system_prompt', '')}\n{self.guardrails_manager.get_guardrails_prompt()}"

        async def _init_environment_vars():
            environ_dict = self.agent_config.get('environment') or self.agent_config.get('env', {})
            resolved_env = self.tool_registry.update_env(environ_dict)
            import os
            for k, v in resolved_env.items():
                if v and len(v) != 0:
                    os.environ[k] = v

        async def _init_agent_skills():
            # Only initialize skills if registry exists (agent uses skills)
            if not self.skill_registry:
                return

            # Get skills configuration
            agent_skills_props = self.agent_config.get("skills", {})
            skill_dir = agent_skills_props.get('skill_dir') if agent_skills_props else None

            if not skill_dir:
                self.logger.debug("No skill_dir configured, skipping skill initialization")
                return

            # Initialize hybrid registry if configured (remote metadata)
            if self.skill_registry.registry_url:
                try:
                    await self.skill_registry.initialize()
                    self.logger.info("Initialized hybrid skill registry with remote metadata")
                except Exception as e:
                    self.logger.warning(f"Failed to initialize hybrid skill registry: {e}")

            # ======= STEP 1: COLLECT ALL REQUIRED SKILLS FROM AGENT CONFIG =======
            agent_list = self.agent_config.get('agent_list', [])
            all_required_skills = set()

            for agent_dict in agent_list:
                if not isinstance(agent_dict, dict):
                    continue

                # Get agent config (agent_dict is {agent_name: agent_config})
                for agent_name, agent_config in agent_dict.items():
                    if not isinstance(agent_config, dict):
                        continue

                    # Get skills required by this agent
                    configured_skills = agent_config.get('skills', [])
                    if isinstance(configured_skills, list):
                        all_required_skills.update(configured_skills)

            self.logger.info(f"Agent requires skills: {all_required_skills if all_required_skills else 'none'}")

            # ======= STEP 2: PULL ALL MISSING SKILLS FROM REGISTRY =======
            if all_required_skills and self.skill_registry.registry_url:
                # skills_cache_dir is resolved against project_root (same as discover_skills)
                pull_target = self.skill_registry.skills_cache_dir
                self.logger.info(f"Pulling missing skills from registry into: {pull_target}")

                with trace_span("agent.pull_skills", level="debug",
                                agent_name=self.agent_name,
                                skill_count=len(all_required_skills)):
                    for skill_name in all_required_skills:
                        try:
                            self.logger.info(f"  • Pulling '{skill_name}'...")
                            success = await self.skill_registry.pull_skill(skill_name)
                            if success:
                                self.logger.info(f"    ✓ Successfully pulled '{skill_name}'")
                            else:
                                self.logger.warning(f"    ⚠ Failed to pull skill '{skill_name}'")
                        except Exception as e:
                            self.logger.warning(f"    ⚠ Error pulling skill '{skill_name}': {e}")

            # ======= STEP 3: DISCOVER ALL SKILLS (LOCAL + PULLED) FROM SAME DIRECTORY =======
            # Both local and pulled skills are in skill_dir, discover them all together
            self.logger.info(f"Discovering all skills from: {skill_dir}")
            with trace_span("agent.discover_skills", level="debug", agent_name=self.agent_name):
                self.skill_registry.discover_skills(skills_dir=skill_dir)
            self.logger.info(f"Discovered {len(self.skill_registry.skills)} skill(s)")

            # Log summary
            if all_required_skills:
                available = set(self.skill_registry.skills.keys())
                loaded = all_required_skills & available
                missing = all_required_skills - available

                self.logger.info(f"Skill loading summary:")
                self.logger.info(f"  • Required: {all_required_skills}")
                self.logger.info(f"  • Loaded: {loaded}")
                if missing:
                    self.logger.warning(f"  • Missing: {missing}")
                else:
                    self.logger.info(f"✓ All {len(loaded)} required skill(s) loaded successfully")

        async def _init_structured_output_models():
            structured_output_models_props = self.agent_config.get("structured_output", {})
            if structured_output_models_props:
                with trace_span("agent.load_structured_output", level="debug",
                                agent_name=self.agent_name):
                    self.output_model_registry.discover_output_models(
                        output_model_dir=structured_output_models_props.get('script_dir')
                    )


        async def _augment_system_prompt_task():
            self.agent_config = self._augment_system_prompt(self.agent_config)


        with trace_span("agent.load_resources", agent_name=self.agent_name,
                        agent_type=self.agent_type):
            await asyncio.gather(_load_tools_and_mcp(),
                                 _init_global_kb(),
                                 _init_memory_store(),
                                 _init_guardrails(),
                                 _init_environment_vars(),
                                 _init_agent_skills(),
                                 _init_structured_output_models(),
                                 _augment_system_prompt_task())

    def _augment_system_prompt(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively process and resolve macros in system prompts.

        This method is non-mutating and returns a new config dictionary with all
        macros in system prompts resolved. It handles both top-level prompts and
        nested agent configurations in agent_list.

        Args:
            config: Agent configuration dictionary potentially containing macro expressions.

        Returns:
            New configuration dictionary with all macros resolved.
        """
        macro_processor = MacroProcessor(project_root=self.config_root, logger=self.logger)

        # Create a deep copy to avoid modifying the original config
        new_config = deepcopy(config)

        # Process top-level system_prompt
        system_prompt = new_config.get('system_prompt')
        if system_prompt and '{{' in system_prompt and '}}' in system_prompt:
            processed_prompt = macro_processor.process(system_prompt)
            if processed_prompt.startswith("Error:"):
                self.logger.warning(f"Macro processing failed for top-level prompt: {processed_prompt}")
            else:
                new_config['system_prompt'] = processed_prompt

        # Process agent_list recursively
        if 'agent_list' in new_config and isinstance(new_config['agent_list'], list):
            new_agent_list = []
            for agent_dict in new_config['agent_list']:
                if isinstance(agent_dict, dict):
                    processed_agent_dict = {}
                    for agent_name, agent_props in agent_dict.items():
                        # Recursively process the nested agent's config
                        processed_agent_dict[agent_name] = self._augment_system_prompt(agent_props)
                    new_agent_list.append(processed_agent_dict)
                else:
                    new_agent_list.append(agent_dict) # Keep non-dict items as is
            new_config['agent_list'] = new_agent_list

        return new_config

    @traced("agent.memory.context", level="debug")
    def _get_conversation_context(self, current_message: str) -> str:
        """Retrieve and format relevant conversation context.

        Uses the memory store to find both recent conversation turns and semantically
        relevant historical turns, formatting them for prompt augmentation.

        Args:
            current_message: Current user message (used for semantic similarity search).

        Returns:
            Formatted conversation context string. Empty string if no memory store or
            if retrieval fails. Format includes recent turns and relevant historical
            turns separated for clarity.
        """
        if not self.memory_store:
            return current_message

        try:
            recent_turns, relevant_turns = self.memory_store.get_relevant_context(
                current_message=current_message,
                session_id=self.session_id,
                user_id=self.user_id,
                include_recent=True
            )

            context = self.memory_store.format_context_for_prompt(
                recent_turns=recent_turns,
                relevant_turns=relevant_turns
            )

            return context

        except Exception as e:
            self.logger.warning("Could not retrieve conversation context: %s", e)
            return ""

    @traced("agent.guardrail.input")
    def _guardrail_input_message(self, message: str) -> str:
        """Apply input validation guardrails to user message.

        Validates incoming user messages against configured guardrails to detect
        and filter potentially harmful, off-topic, or policy-violating content.

        Args:
            message: User input message to validate.

        Returns:
            Validated/filtered message if guardrails enabled, otherwise original message.
            Guardrails may modify or block the message based on configured rules.
        """
        if self.guardrails_manager:
            return self.guardrails_manager.validate_input(message)
        return message

    @traced("agent.guardrail.output")
    def _guardrail_output_message(self, message: str) -> str:
        """Apply output validation guardrails to agent response.

        Validates outgoing agent responses against configured guardrails to ensure
        they comply with policies, are factually grounded, and don't violate safety rules.

        Args:
            message: Agent response message to validate.

        Returns:
            Validated/filtered message if guardrails enabled, otherwise original message.
            Guardrails may modify the message to comply with policies.
        """
        if self.guardrails_manager:
            return self.guardrails_manager.validate_output(message)
        return message

    @traced("agent.augment_message", level="debug")
    def _augment_message(self, message: str, original_query: str = None) -> str:
        """Augment message with knowledge base and conversation context.

        Enriches the message with:
        1. Relevant documents from global knowledge base (semantic search)
        2. Recent and relevant conversation history from memory store
        3. Structured output model schema (if configured)

        This creates a more grounded context for the LLM by combining historical
        conversation context with domain-specific knowledge.

        Args:
            message: Message/prompt to augment (typically the formatted user query).
            original_query: Original user query for semantic search. If None, uses message
                instead. Useful when message has been transformed/formatted.

        Returns:
            Augmented message string with knowledge base results and conversation context
            appended. Returns original message if KB or memory retrieval fails.
        """
        query = original_query if original_query else message
        augmented_message = message

        # Augment with global knowledge base if available
        if self.global_kb_factory:
            try:
                kb_result = self.global_kb_factory.search_custom_knowledge_base(query)
                augmented_message = "%s\n\nRelevant Context from Knowledge Base:\n%s" % (augmented_message, kb_result)
            except Exception as e:
                self.logger.warning("Failed to search knowledge base: %s", e)

        # Add conversation context
        if self.memory_store:
            conversation_context = self._get_conversation_context(current_message=query)
            if conversation_context:
                augmented_message = f"{conversation_context}\n\n{augmented_message}"

        structured_output_model = self.agent_config.get('crew_config', {}).get('structured_output_model', None)
        if structured_output_model:
            augmented_message=f"{augmented_message}{self.output_model_registry.get_system_prompt(structured_output_model)}"

        # A2UI (agent-driven UI): when crew_config.agui_config.enabled is set,
        # instruct the model to emit A2UI v0.9 JSON blocks that the A2A server
        # layer converts into renderable UI parts. The component reference is
        # generated from agui_config.catalog_path when a custom catalog is used.
        agui_instructions = build_agui_instructions(self.agent_config, self.config_root)
        if agui_instructions:
            augmented_message = f"{augmented_message}\n\n{agui_instructions}"

        return augmented_message

    def get_config_value(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'model.temperature').

        Args:
            key_path: Dot-separated path to the configuration key.
            default: Default value to return if key is not found.

        Returns:
            The configuration value or default.
        """
        return self.config_manager.get_config_value(self.agent_config, key_path, default)

    def set_config_value(self, key_path: str, value: Any) -> None:
        """Set a configuration value using dot notation.

        Args:
            key_path: Dot-separated path to the configuration key.
            value: Value to set.
        """
        self.config_manager.set_config_value(self.agent_config, key_path, value)
        # Update agent_type if it was changed
        if key_path == 'type':
            self.agent_type = value

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Merge configuration updates into agent config.

        Performs a deep merge to preserve existing nested configuration while
        updating specified fields. Updates are applied to agent_config only
        (not persisted to disk).

        Args:
            updates: Dictionary of configuration updates to merge. Nested dicts
                are merged recursively, not replaced wholesale.

        Example:
            agent.update_config({"model": {"temperature": 0.8}})
        """
        self.agent_config = self.config_manager.merge_configs(self.agent_config, updates)
        # Update agent_type if it was changed
        if 'type' in updates:
            self.agent_type = updates['type']

    def validate_config(self) -> None:
        """Validate agent configuration comprehensively.

        Performs validation of essential configuration fields and dependencies:
        1. **Required fields**: 'type' must be present and valid
        2. **Model configuration**: Either 'llm' or 'model' config must exist
        3. **Tool setup**: If tools configured, validates tool_registry is available
        4. **Knowledge base**: If KB configured, validates necessary dependencies
        5. **Memory setup**: If memory configured, validates vector_store config
        6. **Guardrails**: If guardrails enabled, validates guardrails_manager exists

        Validation is lenient for optional components—missing optional dependencies
        are logged as warnings, not errors, allowing graceful degradation.

        Raises:
            ValueError: If required configuration is missing or invalid.
            TypeError: If configuration values have incorrect types.

        Example:
            ```python
            agent = MyAgent(config)
            try:
                agent.validate_config()
            except ValueError as e:
                print(f"Configuration error: {e}")
            ```
        """
        errors = []
        warnings = []

        # 1. Validate required 'type' field
        if 'type' not in self.agent_config:
            errors.append("Missing required field 'type' in configuration")
        else:
            valid_types = [
                'custom', 'langchain', 'crewai', 'bedrock', 'mcp',
                'remote', 'multi-agent', 'langgraph', 'openai'
            ]
            agent_type = self.agent_config.get('type')
            if agent_type not in valid_types:
                errors.append(
                    "'type' must be one of %s, got '%s'" % (valid_types, agent_type)
                )

        # 2. Validate model configuration
        if self.llm is None and 'model' not in self.agent_config:
            errors.append(
                "Either 'llm' parameter or 'model' configuration required. "
                "Neither provided."
            )
        elif 'model' in self.agent_config:
            model_config = self.agent_config['model']
            if not isinstance(model_config, dict):
                errors.append("'model' configuration must be a dictionary")
            elif 'name' not in model_config:
                errors.append("'model' configuration missing required 'name' field")

        # 3. Validate tool configuration
        tools_config = self.agent_config.get('tools', {})
        if tools_config:
            if not isinstance(tools_config, dict):
                errors.append("'tools' configuration must be a dictionary")
            # Tool registry is optional, warn if not present
            if not hasattr(self, 'tool_registry') or self.tool_registry is None:
                warnings.append("Tools configured but tool_registry not initialized")

        # 4. Validate knowledge base configuration
        kb_config = self.agent_config.get('knowledge_base')
        if kb_config:
            if isinstance(kb_config, dict):
                # New style: {registry: {url, token}, sources: [...]}
                registry = kb_config.get('registry', {})
                if registry and not registry.get('url'):
                    warnings.append(
                        "'knowledge_base.registry.url' is empty — set it or KB_REGISTRY_URL env var"
                    )
                sources = kb_config.get('sources', [])
                if not isinstance(sources, list):
                    errors.append("'knowledge_base.sources' must be a list")
            elif isinstance(kb_config, list):
                # Old style (or agent-level flat list) — backward-compatible
                for idx, kb in enumerate(kb_config):
                    if not isinstance(kb, dict):
                        errors.append("Knowledge base config item %d must be a dictionary" % idx)
                        continue
                    # Only require vector_store for fully inline (non-registry) configs
                    is_registry_kb = kb.get('registry_url') or kb.get('registry_name')
                    if not is_registry_kb and 'vector_store' not in kb:
                        errors.append(
                            "Knowledge base config item %d missing 'vector_store' configuration" % idx
                        )
            else:
                errors.append(
                    "'knowledge_base' must be a list (inline sources) or a dict with "
                    "'registry' and 'sources' keys (registry style)"
                )

        # 5. Validate memory configuration
        memory_config = self.agent_config.get('memory', {})
        if memory_config:
            if not isinstance(memory_config, dict):
                errors.append("'memory' configuration must be a dictionary")
            else:
                if 'vector_store' not in memory_config:
                    errors.append("'memory' configuration missing required 'vector_store'")
                if 'embedding' not in memory_config:
                    warnings.append("'memory' configuration should include 'embedding' config")

        # 6. Validate guardrails configuration
        guardrails_config = self.agent_config.get('guardrails', {})
        if guardrails_config:
            if not isinstance(guardrails_config, dict):
                errors.append("'guardrails' configuration must be a dictionary")
            if guardrails_config.get('enable_agent_validation', True):
                if not hasattr(self, 'guardrails_manager') or self.guardrails_manager is None:
                    warnings.append("Guardrails enabled but guardrails_manager not initialized")

        # 7. Validate skills configuration (optional)
        # Skills can include skill_dir and optional remote registry
        skills_config = self.agent_config.get('skills', {})
        if skills_config:
            if not isinstance(skills_config, dict):
                errors.append("'skills' configuration must be a dictionary")
            else:
                # Validate skill_dir if present
                if 'skill_dir' in skills_config:
                    skill_dir = skills_config.get('skill_dir')
                    if not isinstance(skill_dir, str):
                        errors.append("'skills.skill_dir' must be a string path")

                # Validate registry configuration if present
                if 'registry' in skills_config:
                    registry_config = skills_config.get('registry', {})
                    if not isinstance(registry_config, dict):
                        errors.append("'skills.registry' configuration must be a dictionary")
                    elif 'url' in registry_config:
                        registry_url = registry_config.get('url')
                        if not isinstance(registry_url, str):
                            errors.append("'skills.registry.url' must be a string URL")

        # 8. Validate structured output configuration
        structured_output_config = self.agent_config.get('structured_output', {})
        if structured_output_config:
            if not isinstance(structured_output_config, dict):
                errors.append("'structured_output' configuration must be a dictionary")
            elif 'script_dir' in structured_output_config:
                script_dir = structured_output_config.get('script_dir')
                if not isinstance(script_dir, str):
                    errors.append("'structured_output.script_dir' must be a string path")

        # Log warnings
        for warning in warnings:
            self.logger.warning("Configuration warning: %s", warning)

        # Raise errors if any
        if errors:
            error_msg = "Configuration validation failed for agent '%s':\n  - %s" % (
                self.agent_name,
                "\n  - ".join(errors)
            )
            raise ValueError(error_msg)

    @staticmethod
    def load_agent_config(agent_name: str) -> Dict[str, Any]:
        """Load agent configuration from YAML file (static method for backward compatibility).

        Args:
            agent_name: Name of the agent to load configuration for.

        Returns:
            Dictionary containing agent configuration.
        """
        return config_manager.load_agent_config(agent_name)

    @property
    def is_initialized(self) -> bool:
        """Check if agent is initialized.

        Returns:
            True if initialized, False otherwise.
        """
        return self._initialized

    @abstractmethod
    async def initialize(self):
        """Initialize agent resources and prepare for execution.

        Subclasses should implement initialization of framework-specific components,
        including loading tools into the agent, setting up the execution pipeline,
        and validating configuration. This is called automatically on first ainvoke()
        if not called explicitly.

        Must set self._initialized = True when complete to prevent re-initialization.

        Raises:
            Exception: Should raise if required resources cannot be initialized.
        """
        raise NotImplementedError("Subclasses must implement initialize method")

    @abstractmethod
    async def astream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Stream agent response asynchronously, yielding chunks as they're generated.

        Enables real-time response delivery for better user experience. Each chunk
        should be a dict with at least a 'content' field containing the text chunk.

        Args:
            user_message: Input message from the user.
            config: Optional configuration overrides for this request (e.g., temperature,
                tool selection). Merges with agent_config for this invocation only.

        Yields:
            Response chunks as dicts. Expected format:
            {"content": str, "final": bool, ...}
            where 'final'=True indicates the last chunk.

        Raises:
            Exception: Should be caught and yielded as error dict to maintain stream contract.
        """
        raise NotImplementedError("Subclasses must implement astream method")

    @abstractmethod
    async def ainvoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Execute agent and return complete response asynchronously.

        Preferred method for non-blocking agent execution. Includes guardrails,
        memory augmentation, and knowledge base retrieval as configured.

        Args:
            user_message: Input message from the user.
            config: Optional configuration overrides for this request. Takes precedence
                over agent_config but doesn't persist.

        Returns:
            Agent response. Format is framework-specific but should include:
            - 'content' or 'message': Response text
            - 'final': True
            - Optional: tool calls, reasoning steps, etc.

        Raises:
            Exception: Implementation-specific errors should be raised (not caught).
        """
        raise NotImplementedError("Subclasses must implement ainvoke method")

    @abstractmethod
    def invoke(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Execute agent synchronously and return complete response.

        Convenience wrapper for async code. Typically uses asyncio.run() or similar
        to block until ainvoke() completes. Suitable for sync codebases or simple scripts.

        Args:
            user_message: Input message from the user.
            config: Optional configuration overrides for this request.

        Returns:
            Agent response (same format as ainvoke).

        Raises:
            Exception: Implementation-specific errors.
        """
        raise NotImplementedError("Subclasses must implement invoke method")

    @abstractmethod
    async def stream(self, user_message: str, config: Optional[Dict[str, Any]] = None):
        """Stream agent response synchronously (blocking).

        Primarily for frameworks that don't support true async streaming. This is
        marked as async for API consistency but may block.

        Args:
            user_message: Input message from the user.
            config: Optional configuration overrides for this request.

        Yields:
            Response chunks (same format as astream).

        Raises:
            NotImplementedError: Some frameworks may not support streaming.
        """
        raise NotImplementedError("Subclasses must implement stream method")

    async def _ensure_initialized(self) -> None:
        """Ensure the agent is initialized before use."""
        if not self._initialized:
            await self.initialize()

    def __del__(self):
        """Cleanup resources when the agent is destroyed."""
        if hasattr(self, 'langfuse_manager') and self.langfuse_manager.is_enabled:
            self.langfuse_manager.flush()
