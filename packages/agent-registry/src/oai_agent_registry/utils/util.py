import logging

import requests

logger = logging.getLogger(__name__)


def get_public_ip() -> str:
    """
    Best-effort determination of the host machine's public IP address.
    Falls back to localhost.
    """
    try:
        response = requests.get('https://api.ipify.org', timeout=5)
        if response.ok:
            return response.text
    except requests.RequestException as e:
        logger.warning(f"Could not determine public IP: {e}. Falling back to 127.0.0.1.")
    return '127.0.0.1'


def get_private_ip() -> str:
    """
    Best-effort determination of the host machine's private IP address.
    Falls back to localhost.
    """
    try:
        import socket

        # Get the hostname of the machine
        hostname = socket.gethostname()

        # Resolve the hostname to an IP address
        private_ip = socket.gethostbyname(hostname)
        return private_ip
    except:
        logger.warning(f"Could not determine private IP. Falling back to 127.0.0.1.")
    return '127.0.0.1'
