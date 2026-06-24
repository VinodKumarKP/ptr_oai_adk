"""KB Registry: Neo4j knowledge-graph provider support (no live DB required).

The neo4j driver is imported lazily inside Neo4jGraphStore._connect, so these
tests patch _connect and never need a real database.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from oai_kb_registry.models import KBRegistration, VectorDBType, DeploymentMode
from oai_kb_registry.services.provider_factory import VectorStoreProviderFactory
from oai_agent_core.components.vector_store.neo4j_graph_store import Neo4jGraphStore


# --- model: neo4j is always external -------------------------------------

def test_neo4j_registration_forced_external():
    reg = KBRegistration(
        name="insurance_graph",
        vector_db_type=VectorDBType.NEO4J,
        deployment_mode=DeploymentMode.BUILTIN,  # should be overridden
    )
    assert reg.deployment_mode == DeploymentMode.EXTERNAL


def test_pinecone_still_forced_external():
    reg = KBRegistration(
        name="p", vector_db_type=VectorDBType.PINECONE,
        deployment_mode=DeploymentMode.BUILTIN,
    )
    assert reg.deployment_mode == DeploymentMode.EXTERNAL


def test_neo4j_text2cypher_without_llm_rejected_at_registration():
    from oai_kb_registry.models import VectorDBConfig
    with pytest.raises(ValueError, match="neo4j_llm_model_id"):
        KBRegistration(
            name="g",
            vector_db_type=VectorDBType.NEO4J,
            vector_db_config=VectorDBConfig(
                neo4j_url="bolt://x", neo4j_retrieval_mode="text2cypher",
            ),
        )


def test_neo4j_text2cypher_with_llm_allowed_at_registration():
    from oai_kb_registry.models import VectorDBConfig
    reg = KBRegistration(
        name="g",
        vector_db_type=VectorDBType.NEO4J,
        vector_db_config=VectorDBConfig(
            neo4j_url="bolt://x",
            neo4j_retrieval_mode="text2cypher",
            neo4j_llm_model_id="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        ),
    )
    assert reg.vector_db_config.neo4j_llm_model_id.startswith("bedrock/")


def test_neo4j_traversal_allowed_at_registration():
    from oai_kb_registry.models import VectorDBConfig
    reg = KBRegistration(
        name="g",
        vector_db_type=VectorDBType.NEO4J,
        vector_db_config=VectorDBConfig(
            neo4j_url="bolt://x", neo4j_retrieval_mode="traversal",
            neo4j_entry_strategy="fulltext",
        ),
    )
    assert reg.vector_db_config.neo4j_retrieval_mode == "traversal"


def test_factory_text2cypher_without_llm_rejected():
    configs = {"neo4j_url": "bolt://x", "neo4j_retrieval_mode": "text2cypher"}
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        with pytest.raises(ValueError, match="neo4j_llm_model_id"):
            VectorStoreProviderFactory.create_vector_store(
                kb_name="g", vector_db_type="neo4j_graph",
                deployment_mode="external", configs=configs,
            )


def test_factory_text2cypher_with_llm_builds_store():
    configs = {
        "neo4j_url": "bolt://x",
        "neo4j_retrieval_mode": "text2cypher",
        "neo4j_llm_model_id": "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
        "neo4j_llm_region": "us-west-2",
    }
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        store = VectorStoreProviderFactory.create_vector_store(
            kb_name="g", vector_db_type="neo4j_graph",
            deployment_mode="external", configs=configs,
        )
    assert isinstance(store, Neo4jGraphStore)
    assert store._mode == "text2cypher"
    assert store._llm == "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0"
    assert store._llm_region == "us-west-2"


# --- provider factory: build the graph store from configs ----------------

def test_factory_builds_traversal_graph_store():
    configs = {
        "neo4j_url": "bolt://localhost:7687",
        "neo4j_username": "neo4j",
        "neo4j_password": "pw",
        "neo4j_database": "neo4j",
        "neo4j_retrieval_mode": "traversal",
        "neo4j_entry_strategy": "fulltext",
        "neo4j_fulltext_index": "entityNames",
        "neo4j_max_hops": "3",
        "neo4j_rel_limit": "25",
    }
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        store = VectorStoreProviderFactory.create_vector_store(
            kb_name="insurance_graph",
            vector_db_type="neo4j_graph",
            deployment_mode="external",
            configs=configs,
        )
    assert isinstance(store, Neo4jGraphStore)
    assert store._entry_strategy == "fulltext"
    assert store._fulltext_index == "entityNames"
    assert store._max_hops == 3
    assert store._rel_limit == 25
    # traversal needs no embeddings
    assert store.embedding_function is None


def test_factory_requires_url():
    with pytest.raises(ValueError, match="neo4j_url is required"):
        VectorStoreProviderFactory.create_vector_store(
            kb_name="g", vector_db_type="neo4j_graph",
            deployment_mode="external", configs={},
        )


def test_factory_hybrid_vector_entry_parses_json_and_builds_embeddings():
    configs = {
        "neo4j_url": "bolt://localhost:7687",
        "neo4j_entry_strategy": "vector",
        "neo4j_vector_entry": json.dumps(
            {"type": "postgres", "settings": {"collection_name": "kg_nodes"}}
        ),
    }
    fake_emb = object()
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()), \
         patch(
             "oai_kb_registry.services.provider_factory._make_litellm_embeddings",
             return_value=fake_emb,
         ) as mock_emb:
        store = VectorStoreProviderFactory.create_vector_store(
            kb_name="g", vector_db_type="neo4j_graph",
            deployment_mode="external",
            embedding_model_id="bedrock/amazon.titan-embed-text-v1",
            configs=configs,
        )
    assert isinstance(store, Neo4jGraphStore)
    assert store._entry_strategy == "vector"
    assert store._vector_entry_config == {
        "type": "postgres", "settings": {"collection_name": "kg_nodes"}
    }
    mock_emb.assert_called_once()                 # embeddings created for hybrid
    assert store.embedding_function is fake_emb


def test_factory_vector_entry_requires_json_config():
    configs = {"neo4j_url": "bolt://x", "neo4j_entry_strategy": "vector"}
    with patch.object(Neo4jGraphStore, "_connect", return_value=MagicMock()):
        with pytest.raises(ValueError, match="neo4j_vector_entry"):
            VectorStoreProviderFactory.create_vector_store(
                kb_name="g", vector_db_type="neo4j_graph",
                deployment_mode="external", configs=configs,
            )
