from .main import AgentHTTPServer, main

try:
    from ._version import version as __version__
except ImportError:
    # Fallback if package is not installed/built correctly
    __version__ = "0.0.0"
