import requests
import pytest
from unittest.mock import patch, MagicMock
from oai_platform_core.networking import get_public_ip, get_private_ip, get_local_ip


@patch("oai_platform_core.networking.requests.get")
def test_get_public_ip_success(mock_get):
    """get_public_ip returns the text from the ipify response."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.text = "123.45.67.89"
    mock_get.return_value = mock_response

    ip = get_public_ip()
    assert ip == "123.45.67.89"
    mock_get.assert_called_once_with("https://api.ipify.org", timeout=5)


@patch("oai_platform_core.networking.requests.get")
def test_get_public_ip_custom_timeout(mock_get):
    """get_public_ip forwards the timeout parameter."""
    mock_response = MagicMock()
    mock_response.ok = True
    mock_response.text = "1.2.3.4"
    mock_get.return_value = mock_response

    ip = get_public_ip(timeout=10)
    assert ip == "1.2.3.4"
    mock_get.assert_called_once_with("https://api.ipify.org", timeout=10)


@patch("oai_platform_core.networking.requests.get")
def test_get_public_ip_non_ok_response_falls_back(mock_get):
    """When response.ok is False the fallback IP is returned."""
    mock_response = MagicMock()
    mock_response.ok = False
    mock_get.return_value = mock_response

    ip = get_public_ip()
    assert ip == "127.0.0.1"


@patch("oai_platform_core.networking.requests.get")
def test_get_public_ip_failure_falls_back(mock_get):
    """Network errors are silently swallowed; the fallback IP is returned."""
    mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

    ip = get_public_ip()
    assert ip == "127.0.0.1"


@patch("oai_platform_core.networking.requests.get")
def test_get_public_ip_timeout_falls_back(mock_get):
    """Timeout errors are silently swallowed; the fallback IP is returned."""
    mock_get.side_effect = requests.exceptions.Timeout("timed out")

    ip = get_public_ip()
    assert ip == "127.0.0.1"


@patch("oai_platform_core.networking.socket.socket")
def test_get_private_ip_success(mock_socket_cls):
    """get_private_ip returns the IP from getsockname()[0]."""
    mock_sock = MagicMock()
    mock_sock.getsockname.return_value = ("192.168.1.100", 0)
    mock_socket_cls.return_value = mock_sock

    ip = get_private_ip()
    assert ip == "192.168.1.100"
    mock_sock.connect.assert_called_once_with(("8.8.8.8", 1))
    mock_sock.close.assert_called_once()


@patch("oai_platform_core.networking.socket.socket")
def test_get_private_ip_failure_falls_back(mock_socket_cls):
    """Any socket error is swallowed; 127.0.0.1 is returned."""
    mock_socket_cls.side_effect = OSError("Socket error")

    ip = get_private_ip()
    assert ip == "127.0.0.1"


@patch("oai_platform_core.networking.socket.socket")
def test_get_private_ip_connect_failure_falls_back(mock_socket_cls):
    """connect() failure is swallowed; socket is still closed."""
    mock_sock = MagicMock()
    mock_sock.connect.side_effect = OSError("unreachable")
    mock_socket_cls.return_value = mock_sock

    ip = get_private_ip()
    assert ip == "127.0.0.1"
    mock_sock.close.assert_called_once()


@patch("oai_platform_core.networking.socket.socket")
def test_get_local_ip_is_alias_for_get_private_ip(mock_socket_cls):
    """get_local_ip and get_private_ip return the same result."""
    mock_sock = MagicMock()
    mock_sock.getsockname.return_value = ("10.0.0.5", 0)
    mock_socket_cls.return_value = mock_sock

    assert get_local_ip() == get_private_ip()
