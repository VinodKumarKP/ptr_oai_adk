import os


class Config:
    """Configuration for the Agent Server."""

    def __init__(self):
        """Initialize configuration from environment variables."""
        self.agent_auth_enabled = os.environ.get('AGENT_AUTH_ENABLED', 'true').lower() == 'true'
        self.force_auth = os.environ.get('FORCE_AUTH', 'false').lower() == 'true'
