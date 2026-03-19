import pytest
from unittest.mock import MagicMock, patch
import sys
from pathlib import Path

# Forcefully add the agent-core src directory to the Python path
# This is a workaround for local development and testing in a monorepo
agent_core_path = Path(__file__).parent.parent.parent / "agent-core/src"
sys.path.insert(0, str(agent_core_path))

from oai_agent_core.aws_strands_core.builders.orchestration_builder import OrchestrationBuilder


@pytest.fixture
def builder():
    return OrchestrationBuilder(logger=MagicMock(), llm=MagicMock(), structured_output_model_registry=MagicMock())


def test_build_orchestration_single_agent(builder):
    agent_map = {'a1': MagicMock()}
    result = builder.build_orchestration(agent_map, {}, {
        'pattern': 'sequential'
    })
    assert result == agent_map['a1']


def test_build_orchestration_swarm(builder):
    agent_map = {'a1': MagicMock(), 'a2': MagicMock()}

    with patch('oai_agent_core.aws_strands_core.builders.orchestration_builder.Swarm') as MockSwarm:
        builder.build_orchestration(agent_map, {},
                                    {'pattern': 'swarm',
                                     'entry_agent': 'a1'
                                     })

        MockSwarm.assert_called_once()
        call_args = MockSwarm.call_args[1]
        assert len(call_args['nodes']) == 2
        assert call_args['entry_point'] == agent_map['a1']


def test_build_orchestration_graph(builder):
    agent_map = {'a1': MagicMock(), 'a2': MagicMock()}
    context_map = {'a2': ['a1']}

    mock_graph_builder = MagicMock()
    with patch('oai_agent_core.aws_strands_core.builders.orchestration_builder.GraphBuilder',
               return_value=mock_graph_builder):
        builder.build_orchestration(agent_map, context_map, {
            'pattern': 'graph'
        })

        assert mock_graph_builder.add_node.call_count == 2
        mock_graph_builder.add_edge.assert_called_with('a1', 'a2')
        mock_graph_builder.build.assert_called_once()


def test_build_orchestration_agent_as_tool(builder):
    agent_map = {'a1': MagicMock(), 'a2': MagicMock()}

    with patch('oai_agent_core.aws_strands_core.builders.orchestration_builder.Agent') as MockAgent:
        builder.build_orchestration(agent_map, {}, {
            'pattern': 'agent-as-tool'
        })

        MockAgent.assert_called_once()
        call_args = MockAgent.call_args[1]
        assert call_args['name'] == 'Supervisor'
        assert len(call_args['tools']) == 2


def test_build_orchestration_agent_as_tool_no_llm():
    builder_no_llm = OrchestrationBuilder(logger=MagicMock())
    agent_map = {'a1': MagicMock()}

    with pytest.raises(ValueError, match="needs an LLM"):
        builder_no_llm.build_orchestration(agent_map, {}, {
            'pattern': 'agent-as-tool'
        })


def test_build_orchestration_invalid_pattern(builder):
    agent_map = {'a1': MagicMock(), 'a2': MagicMock()}
    with pytest.raises(ValueError, match="Unknown orchestration pattern"):
        builder.build_orchestration(agent_map, {}, {
            'pattern': 'invalid'
        })


def test_build_orchestration_no_agents(builder):
    with pytest.raises(ValueError, match="Cannot build orchestration with no agents"):
        builder.build_orchestration({}, {}, {
            'pattern': 'graph'
        })


def test_validate_graph_config_valid(builder):
    agent_map = {'a1': MagicMock(), 'a2': MagicMock()}
    context_map = {'a2': ['a1']}

    is_valid, warnings = builder.validate_graph_config(agent_map, context_map)
    assert is_valid is True
    assert len(warnings) == 0


def test_validate_graph_config_missing_dep(builder):
    agent_map = {'a1': MagicMock()}
    context_map = {'a1': ['unknown']}

    is_valid, warnings = builder.validate_graph_config(agent_map, context_map)
    assert is_valid is False
    assert "depends on unknown agent" in warnings[0]


def test_validate_graph_config_isolated(builder):
    agent_map = {'a1': MagicMock(), 'a2': MagicMock()}
    context_map = {}  # No connections

    is_valid, warnings = builder.validate_graph_config(agent_map, context_map)
    assert is_valid is False
    assert "is isolated" in warnings[0]


def test_get_orchestration_info(builder):
    mock_orch = MagicMock()
    mock_orch.nodes = [1, 2]
    mock_orch.entry_point = "entry"

    info = builder.get_orchestration_info(mock_orch, 'swarm', 2)
    assert info['pattern'] == 'swarm'
    assert info['agent_count'] == 2
    assert info['node_count'] == 2
    assert info['has_entry_point'] is True
