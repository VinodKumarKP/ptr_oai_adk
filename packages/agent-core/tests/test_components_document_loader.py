import pytest
import os
from unittest.mock import MagicMock, patch
from oai_agent_core.components.loaders.document_loader import DocumentLoader

@pytest.fixture
def mock_embedding():
    return MagicMock()

@pytest.fixture
def loader(mock_embedding):
    # We rely on conftest.py to mock BaseDocumentLoader
    # We just need to patch Chroma to avoid real init
    with patch('oai_agent_core.components.loaders.document_loader.ChromaVectorStore') as MockChroma:
        loader = DocumentLoader(
            embedding=mock_embedding,
            persist_directory="/tmp/db"
        )
        yield loader

def test_init_success(loader, mock_embedding):
    assert loader.embedding == mock_embedding
    assert loader.persist_directory == "/tmp/db"
    assert loader.vector_store is not None

def test_init_with_provided_vector_store(mock_embedding):
    mock_store = MagicMock()
    mock_store.count.return_value = 0
    
    loader = DocumentLoader(
        embedding=mock_embedding,
        vector_store=mock_store
    )
    assert loader.vector_store == mock_store

def test_reinitialize_database_exists(loader):
    loader.vector_store.reinitialize_database.return_value = False
    assert loader.reinitialize_database() is False

def test_reinitialize_database_not_exists(loader):
    loader.vector_store.reinitialize_database.return_value = True
    assert loader.reinitialize_database() is True

def test_reinitialize_database_empty(loader):
    loader.vector_store.reinitialize_database.return_value = True
    assert loader.reinitialize_database() is True
