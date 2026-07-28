"""Backend selection for TokenManager: TOKEN_CACHE_BACKEND and the fast probe."""

import socket
import threading
import time

import pytest

from oai_platform_core.security.token_manager import TokenManager


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    for var in ("TOKEN_CACHE_BACKEND", "REDIS_HOST", "REDIS_PORT",
                "REDIS_CONNECT_TIMEOUT", "CACHE_DB_PATH"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CACHE_DB_PATH", str(tmp_path / "cache"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class TestBackendPreference:
    def test_diskcache_skips_redis_entirely(self, monkeypatch):
        monkeypatch.setenv("TOKEN_CACHE_BACKEND", "diskcache")

        def explode(*args, **kwargs):  # pragma: no cover - must not run
            raise AssertionError("Redis must not be probed for diskcache backend")

        monkeypatch.setattr(socket, "create_connection", explode)
        assert TokenManager(db_path_name="t.db").backend == "diskcache"

    def test_auto_falls_back_when_redis_absent(self, monkeypatch):
        monkeypatch.setenv("REDIS_PORT", str(_free_port()))
        assert TokenManager(db_path_name="t.db").backend == "diskcache"

    def test_redis_mode_raises_instead_of_degrading(self, monkeypatch):
        monkeypatch.setenv("TOKEN_CACHE_BACKEND", "redis")
        monkeypatch.setenv("REDIS_PORT", str(_free_port()))
        with pytest.raises(RuntimeError, match="TOKEN_CACHE_BACKEND=redis"):
            TokenManager(db_path_name="t.db")


class TestProbeIsFast:
    def test_unreachable_redis_fails_fast(self, monkeypatch):
        """A refused connection must not cost redis-py's retry/backoff time.

        redis-py retries a refused connection with backoff (measured seconds);
        the reachability probe uses a plain socket so an absent Redis costs
        milliseconds. Generous bound — this guards against the retry storm
        regressing, not against slow CI.
        """
        monkeypatch.setenv("REDIS_PORT", str(_free_port()))
        started = time.perf_counter()
        TokenManager(db_path_name="t.db")
        elapsed = time.perf_counter() - started
        assert elapsed < 1.0, f"probe took {elapsed:.2f}s; retry storm is back"

    def test_reachable_listener_selects_redis(self, monkeypatch):
        """A listener answering PING is used — the probe doesn't break Redis."""
        port = _free_port()
        server = socket.socket()
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", port))
        server.listen(5)

        def serve():
            while True:
                try:
                    conn, _ = server.accept()
                except OSError:
                    return
                threading.Thread(target=handle, args=(conn,), daemon=True).start()

        def handle(conn):
            try:
                while conn.recv(4096):
                    conn.sendall(b"+PONG\r\n")
            except OSError:
                pass

        threading.Thread(target=serve, daemon=True).start()
        try:
            monkeypatch.setenv("REDIS_HOST", "127.0.0.1")
            monkeypatch.setenv("REDIS_PORT", str(port))
            assert TokenManager(db_path_name="t.db").backend == "redis"
        finally:
            server.close()
