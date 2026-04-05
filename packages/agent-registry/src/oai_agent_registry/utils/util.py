import requests
import logging

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
