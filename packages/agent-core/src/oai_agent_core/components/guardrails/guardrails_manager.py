import importlib
import json
import logging
import os
import subprocess
import threading
from functools import lru_cache
from typing import Optional, Dict, Any

import sys
import time

try:
    from guardrails import Guard
except ImportError:
    Guard = None

from oai_agent_core.utils.dynamic_class_loader import DynamicClassLoader


class GuardrailError(Exception):
    """Base exception for GuardrailManager errors."""
    pass


class GuardrailConfigurationError(GuardrailError):
    """Raised when there is an error in the guardrail configuration."""
    pass


class GuardrailInitializationError(GuardrailError):
    """Raised when a guard or validator fails to initialize."""
    pass


class GuardrailManager:
    """
    Production-ready manager for input and output guardrails.

    Implements the Multiton pattern to ensure only one instance exists per
    unique configuration, optimizing resource usage and initialization time.
    """

    _instances: Dict[str, 'GuardrailManager'] = {}
    _lock = threading.Lock()

    def __new__(
            cls,
            project_root: str,
            guardrails_config: Optional[Dict[str, Any]],
            logger: Optional[logging.Logger] = None
    ):
        """
        Implements the Multiton pattern. Returns an existing instance if the
        configuration is the same, otherwise creates a new one.
        """
        # Create a stable key from the configuration
        config_str = json.dumps(guardrails_config, sort_keys=True) if guardrails_config else "{}"
        config_key = f"{project_root}:{config_str}"

        with cls._lock:
            if config_key not in cls._instances:
                instance = super(GuardrailManager, cls).__new__(cls)
                # Initialize the instance-specific attributes
                instance._initialized = False
                cls._instances[config_key] = instance
            return cls._instances[config_key]

    def __init__(
            self,
            project_root: str,
            guardrails_config: Optional[Dict[str, Any]],
            logger: Optional[logging.Logger] = None
    ):
        """
        Initializes the GuardrailManager. Only runs once per unique configuration.
        """
        if getattr(self, "_initialized", False):
            return

        self.project_root = project_root
        self.guardrails_config = guardrails_config or {}
        self.logger = logger or logging.getLogger(__name__)

        if not self.guardrails_config:
            self.logger.info("No guardrail configuration provided. Validation will be skipped.")
            self.input_guard = None
            self.output_guard = None
            self._initialized = True
            return

        # Configure custom validators directory
        self._setup_custom_validators()

        # Pre-process reusable validators
        self.reusable_validators = {
            v["name"]: v for v in self.guardrails_config.get("validators", [])
        }

        # Initialize guards
        try:
            self.input_guard = self._init_guard("input")
            self.output_guard = self._init_guard("output")
        except Exception as e:
            self.logger.critical(f"Failed to initialize GuardrailManager: {e}")
            raise GuardrailInitializationError(f"Critical failure during guard initialization: {e}")

        self._initialized = True

    def _setup_custom_validators(self):
        """Adds the custom validators directory to sys.path."""
        custom_dir = self.guardrails_config.get("custom_validators_dir")
        if custom_dir:
            full_custom_path = os.path.abspath(os.path.join(self.project_root, custom_dir))
            if os.path.exists(full_custom_path):
                if full_custom_path not in sys.path:
                    sys.path.insert(0, full_custom_path)
                self.logger.info(f"Custom validators directory added to sys.path: {full_custom_path}")
            else:
                self.logger.warning(f"Custom validators directory not found: {full_custom_path}")

    @staticmethod
    @lru_cache(maxsize=128)
    def _snake_to_pascal(snake_str: str) -> str:
        """Converts snake_case to PascalCase with caching."""
        return "".join(x.capitalize() for x in snake_str.lower().split("_"))

    def _is_validator_importable(self, class_name: str) -> bool:
        """
        Checks whether a validator class is already importable from guardrails.hub.
        This avoids redundant subprocess installs for validators already present
        in the environment.
        """
        try:
            importlib.invalidate_caches()
            module = importlib.import_module("guardrails.hub")
            return hasattr(module, class_name)
        except ImportError:
            return False

    def _install_validator(self, full_name: str, class_name: str) -> bool:
        """
        Installs a validator from the Guardrails Hub only if it is not already
        importable. Uses exponential backoff when verifying the import after
        installation, instead of a fixed sleep.

        Args:
            full_name:   The validator identifier (e.g. 'toxic_language' or
                         'guardrails/toxic_language' or 'hub://guardrails/toxic_language').
            class_name:  The PascalCase class name used to confirm the import
                         succeeded (e.g. 'ToxicLanguage').

        Returns:
            True if the validator is importable after this call, False otherwise.
        """
        # ── 1. Skip install if already available ──────────────────────────────
        if self._is_validator_importable(class_name):
            self.logger.info(f"{class_name} is already installed. Skipping hub install.")
            return True

        # ── 2. Normalise the hub URI ───────────────────────────────────────────
        install_name = full_name
        if not install_name.startswith("hub://"):
            if not install_name.startswith("guardrails/"):
                install_name = f"guardrails/{install_name}"
            install_name = f"hub://{install_name}"

        self.logger.info(f"Installing validator from hub: {install_name}")

        # ── 3. Resolve the guardrails CLI binary ───────────────────────────────
        python_bin_dir = os.path.dirname(sys.executable)
        guardrails_bin = os.path.join(python_bin_dir, "guardrails")
        cmd = guardrails_bin if os.path.exists(guardrails_bin) else "guardrails"

        # ── 4. Run the install ─────────────────────────────────────────────────
        try:
            subprocess.check_call(
                [cmd, "hub", "install", install_name],
                timeout=60,
            )
            self.logger.info(f"Hub install command succeeded for {install_name}.")
        except subprocess.TimeoutExpired:
            self.logger.error(f"Installation of {install_name} timed out after 60 s.")
            return False
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Hub install command failed for {install_name}: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error installing {install_name}: {e}")
            return False

        # ── 5. Invalidate import caches so the new package is discoverable ─────
        importlib.invalidate_caches()
        if "guardrails.hub" in sys.modules:
            try:
                importlib.reload(sys.modules["guardrails.hub"])
            except Exception as e:
                self.logger.warning(f"Could not reload guardrails.hub: {e}")

        # ── 6. Verify with exponential backoff (replaces arbitrary sleep) ──────
        max_attempts = 5
        base_delay = 0.5  # seconds
        max_delay = 16.0  # seconds cap

        for attempt in range(1, max_attempts + 1):
            if self._is_validator_importable(class_name):
                self.logger.info(
                    f"{class_name} became importable after attempt {attempt}."
                )
                return True

            if attempt < max_attempts:
                delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                self.logger.debug(
                    f"{class_name} not yet importable. "
                    f"Retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})."
                )
                time.sleep(delay)

        self.logger.error(
            f"{class_name} still not importable after {max_attempts} attempts. "
            f"Installation may have failed silently."
        )
        return False

    def _init_guard(self, stage: str) -> Optional[Guard]:
        """Initializes a Guard instance for a specific stage. Returns None if no validators."""
        stage_config = self.guardrails_config.get(stage, {})
        validator_configs = stage_config.get("validators", [])

        if not validator_configs:
            self.logger.info(f"No validators configured for {stage} stage. Skipping initialization.")
            return None

        guard_rail_instances = []
        for config in validator_configs:
            if "ref" in config:
                ref_name = config["ref"]
                if ref_name in self.reusable_validators:
                    base_config = self.reusable_validators[ref_name].copy()
                    base_config.update(config)
                    config = base_config
                else:
                    self.logger.error(f"Reference '{ref_name}' not found.")
                    continue

            full_name = config.get("full_name") or config.get("name")
            class_name = config.get("class_name") or config.get("name")
            if not full_name:
                self.logger.error(f"Validator config missing 'name' or 'full_name': {config}")
                continue

            module_path = config.get("module")
            is_hub_validator = not bool(module_path)

            if is_hub_validator:
                validator_name = full_name.split("/")[-1]
                class_name = class_name or self._snake_to_pascal(validator_name)
                module_path = "guardrails.hub"
            else:
                class_name = full_name

            params = config.get("parameters", {})
            params["on_fail"] = config.get("on_fail", "noop")

            try:
                rail_class = DynamicClassLoader.dynamic_import(module_path, class_name)
                guard_rail_instances.append(rail_class(**params))
                self.logger.info(f"Loaded {class_name} for {stage}.")
            except Exception as e:
                if is_hub_validator:
                    self.logger.warning(
                        f"Failed to load {class_name}: {e}. Attempting hub install..."
                    )
                    installed = self._install_validator(full_name, class_name)
                    if installed:
                        try:
                            rail_class = DynamicClassLoader.dynamic_import(module_path, class_name)
                            guard_rail_instances.append(rail_class(**params))
                            self.logger.info(f"Loaded {class_name} for {stage} after install.")
                        except Exception as retry_e:
                            self.logger.error(
                                f"Final failure loading {class_name} even after successful install: {retry_e}"
                            )
                    else:
                        self.logger.error(
                            f"Skipping {class_name}: install did not make it importable."
                        )
                else:
                    self.logger.error(f"Failed to load custom validator {class_name}: {e}")

        return Guard().use(*guard_rail_instances) if guard_rail_instances else None

    def validate_input(self, text: str) -> str:
        """Validates input text. Returns validated text or error message."""
        return self._run_validation(self.input_guard, text, "input")

    def validate_output(self, text: str) -> str:
        """Validates output text. Returns validated text or error message."""
        return self._run_validation(self.output_guard, text, "output")

    def _run_validation(self, guard: Optional[Guard], text: str, stage: str) -> str:
        """Internal helper to run validation with error handling. Skips if guard is None."""
        if guard is None:
            self.logger.debug(f"Skipping {stage} validation: No guard initialized.")
            return text

        self.logger.debug(f"Starting {stage} validation.")
        try:
            response = guard.validate(text)
            if not response.validation_passed:
                self.logger.warning(f"{stage.capitalize()} validation failed.")
            return response.validated_output
        except Exception as e:
            self.logger.error(f"Error during {stage} validation: {e}")
            return f"Validation Error ({stage}): {str(e)}"

    def get_guardrails_prompt(self, stage: str = "output") -> str:
        """
        Builds a prompt fragment describing the active guardrail rules for the
        given stage. Intended to be injected into an agent's system prompt so the
        LLM is aware of the constraints it must satisfy *before* a response is
        validated — reducing avoidable validation failures at inference time.

        Args:
            stage: Which stage's validators to describe.
                   Accepts "input", "output" (default), or "both".

        Returns:
            A formatted string of numbered guardrail instructions, or an empty
            string if no validators are configured for the requested stage.
        """
        on_fail_instructions = {
            "exception": "If this check fails, do not provide any response and show an appropriate error.",
            "fix": "If this check fails, the agent should attempt to fix the response by masking or appropriate ways",
            "reask": "If this check fails, you will be re-prompted to correct the specific part of your response.",
            "filter": "If this check fails, the failing content will be removed from your final output.",
            "refrain": "If this check fails, the failing content will be replaced with a placeholder or omitted.",
            "noop": "If this check fails, no corrective action will be taken, but the failure will be logged."
        }

        instructions = []
        stages_to_process = ["input", "output"] if stage == "both" else [stage]

        for s in stages_to_process:
            stage_config = self.guardrails_config.get(s, {})
            validator_configs = stage_config.get("validators", [])

            for config in validator_configs:
                # Resolve reference if 'ref' is present
                if "ref" in config:
                    ref_name = config["ref"]
                    if ref_name in self.reusable_validators:
                        base_config = self.reusable_validators[ref_name].copy()
                        base_config.update(config)
                        config = base_config

                # Use the explicit instruction if provided
                instruction = config.get("instruction")

                # If no instruction, fallback to name/params (optional, but you requested to use instruction)
                if not instruction:
                    name = config.get("name") or config.get("full_name", "").split("/")[-1]
                    params = config.get("parameters", {})
                    instruction = f"Validator: {name}"
                    if params:
                        instruction += f", Parameters: {params}"

                on_fail = config.get("on_fail", "noop")
                strategy_instruction = on_fail_instructions.get(on_fail, on_fail_instructions["noop"])

                full_instruction = f"{instruction} {strategy_instruction}"

                if full_instruction not in instructions:
                    instructions.append(full_instruction)

        if not instructions:
            return ""

        prompt = "Please adhere to the following guardrail guidelines in your responses:\n"
        for i, instruction in enumerate(instructions, 1):
            prompt += f"{i}. {instruction}\n"

        prompt += "\nFocus on producing responses that satisfy these guidelines naturally, without referencing or explaining these validation rules to the user."
        return prompt
