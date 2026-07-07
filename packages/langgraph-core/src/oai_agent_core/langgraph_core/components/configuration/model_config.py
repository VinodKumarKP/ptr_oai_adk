from typing import Optional, Dict, Any

from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager


def _import_chat_litellm():
    """Import ``ChatLiteLLM`` without eagerly loading langchain_litellm's
    unused OCR document-loader and embeddings submodules.

    ``langchain_litellm/__init__.py`` eagerly imports ``.document_loaders``
    (the OCR loader) and ``.embeddings``. Those transitively pull
    ``langchain_text_splitters`` -> ``nltk`` (~2s of import cost) which a chat
    model never needs — but every agent construction paid it. We install lazy
    proxy submodules so the package initializes cheaply; each proxy attribute
    materializes the *real* submodule on first actual use, so OCR/embeddings
    still work for any code path that genuinely needs them. On any unexpected
    error we strip the proxies and fall back to a normal (slower but correct)
    import, so correctness can never regress.
    """
    import importlib
    import sys
    import types

    # Fast path: package already imported — nothing to optimize.
    existing = sys.modules.get("langchain_litellm")
    if existing is not None:
        return existing.ChatLiteLLM

    class _LazyAttr:
        """Placeholder that resolves to the real class on first use."""

        def __init__(self, module_name: str, attr_name: str):
            self._module_name = module_name
            self._attr_name = attr_name
            self._real = None

        def _resolve(self):
            if self._real is None:
                sys.modules.pop(self._module_name, None)
                real_module = importlib.import_module(self._module_name)
                self._real = getattr(real_module, self._attr_name)
            return self._real

        def __call__(self, *args, **kwargs):
            return self._resolve()(*args, **kwargs)

        def __getattr__(self, item):
            return getattr(self._resolve(), item)

    def _make_stub(module_name: str, attr_names) -> types.ModuleType:
        stub = types.ModuleType(module_name)
        stub.__spec__ = None
        stub.__path__ = []  # behave like a package
        for attr in attr_names:
            setattr(stub, attr, _LazyAttr(module_name, attr))
        return stub

    installed = []
    stubs = {
        "langchain_litellm.document_loaders": ("LiteLLMOCRLoader",),
        "langchain_litellm.embeddings": ("LiteLLMEmbeddings", "LiteLLMEmbeddingsRouter"),
    }
    for module_name, attr_names in stubs.items():
        if module_name not in sys.modules:
            sys.modules[module_name] = _make_stub(module_name, attr_names)
            installed.append(module_name)

    try:
        from langchain_litellm import ChatLiteLLM
        return ChatLiteLLM
    except Exception:
        # Something about the lazy stubbing didn't agree with this
        # langchain_litellm version — remove our stubs and import normally.
        for module_name in installed:
            sys.modules.pop(module_name, None)
        sys.modules.pop("langchain_litellm", None)
        from langchain_litellm import ChatLiteLLM
        return ChatLiteLLM


class LangChainModelConfigurationManager(BaseModelConfigurationManager):
    """Model configuration manager for LangChain using ChatLiteLLM.

    This implementation uses LangChain's ChatLiteLLM wrapper around LiteLLM
    to provide a LangChain-compatible chat model interface.

    Supports additional LLM parameters like top_p, top_k, presence_penalty, etc.
    """

    def create_model(self, model_config: Optional[Dict[str, Any]] = None):
        """Create ChatLiteLLM instance from configuration.

        Args:
            model_config: Optional model configuration (uses defaults if None)
                Additional params can include: top_p, top_k, presence_penalty,
                frequency_penalty, stop, streaming, timeout, etc.

        Returns:
            Configured ChatLiteLLM instance

        Example:
            >>> manager = LangChainModelConfigurationManager()
            >>> model = manager.create_model({
            ...     'temperature': 0.9,
            ...     'params': {'top_p': 0.95, 'presence_penalty': 0.5}
            ... })
        """
        try:
            ChatLiteLLM = _import_chat_litellm()
        except ImportError:
            raise ImportError(
                "Please install langchain-litellm: "
                "pip install langchain-litellm"
            )

        # Use default config if none provided
        if not model_config:
            config = self.default_config
        else:
            config = self._merge_with_defaults(model_config)

        # Validate configuration
        self._validate_config(config)

        try:
            # Extract standard parameters
            model_params = {
                'model': config['model_id'],
                'temperature': config['params']['temperature'],
                'max_tokens': config['params']['max_tokens']
            }

            # Add any additional LLM parameters from config
            additional_params = {
                k: v for k, v in config['params'].items()
                if k not in ['temperature', 'max_tokens']
            }
            model_params.update(additional_params)

            # ChatLiteLLM expects model and standard LLM parameters
            model = ChatLiteLLM(**model_params)

            self.logger.debug(
                f"Created ChatLiteLLM: {config['model_id']} "
                f"with params: {model_params}"
            )

            return model

        except Exception as e:
            self.logger.error(f"Failed to create ChatLiteLLM: {e}")
            raise