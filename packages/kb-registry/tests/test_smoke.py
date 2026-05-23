"""
Smoke tests for KB Registry.

These tests run without external services (Chroma, Postgres, S3).
They verify that the FastAPI app starts and health / root endpoints respond.
"""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client with auth disabled."""
    import os
    os.environ["KB_AUTH_ENABLED"] = "false"
    os.environ["REGISTRY_DB_LOGGING_ENABLED"] = "false"

    from oai_kb_registry.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    resp = client.get("/api/v1/kb-registry/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_root(client):
    resp = client.get("/api/v1/kb-registry/")
    assert resp.status_code == 200
    data = resp.json()
    assert "endpoints" in data


def test_list_knowledge_bases_empty(client):
    resp = client.get("/api/v1/kb-registry/knowledge-bases")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_unknown_kb_404(client):
    resp = client.get("/api/v1/kb-registry/knowledge-bases/does-not-exist")
    assert resp.status_code == 404
