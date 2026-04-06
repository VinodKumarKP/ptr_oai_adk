import socket
import requests
import logging

logger = logging.getLogger(__name__)

def get_local_ip() -> str:
    """Best-effort determination of the host machine's IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        ip_addr = s.getsockname()[0]
    except Exception:
        ip_addr = '127.0.0.1'
    finally:
        s.close()
    return ip_addr

def get_public_ip() -> str:
    """Best-effort determination of the host machine's public IP address."""
    try:
        response = requests.get('https://api.ipify.org', timeout=5)
        if response.ok:
            return response.text
    except requests.RequestException as e:
        logger.warning(f"Could not determine public IP: {e}. Falling back to 127.0.0.1.")
    return '127.0.0.1'
