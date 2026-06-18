import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from oai_agent_core.openai_core.builders.agent_builder import AgentBuilder


@pytest.fixture
def mock_deps():
    mock_manager = MagicMock()
    mock_manager.get_model_info = MagicMock(return_value={'provider': 'openai'})
    return {
        'model_manager': MagicMock(),
        'tool_registry': MagicMock(),
        'llm': MagicMock(),
        'config_root': '/tmp',
        'skill_registry': MagicMock(),
        'structured_output_model_registry': MagicMock()
    }


@pytest.fixture
def builder(mock_deps):
    return AgentBuilder(**mock_deps)


def test_init(builder, mock_deps):
    assert builder.llm == mock_deps['llm']
    assert builder.tool_registry == mock_deps['tool_registry']
    assert builder.config_root == '/tmp'


def test_create_agent_instance_applies_model_settings(builder):
    # Regression: configured sampling params must be passed to ModelSettings
    # (alongside include_usage), not silently dropped.
    builder.tool_registry.enable_lazy_loading = False
    builder.model_manager.build_model_settings = MagicMock(
        return_value={'temperature': 0.3, 'top_p': 0.9}
    )
    builder.structured_output_model_registry.get_model = MagicMock(return_value=None)

    config = {
        'system_prompt': 'prompt',
        'model': {'model_id': 'gpt-4', 'temperature': 0.3},
    }

    with patch('oai_agent_core.openai_core.builders.agent_builder.Agent') as mock_agent_cls, \
         patch('oai_agent_core.openai_core.builders.agent_builder.ModelSettings') as mock_ms:
        builder._create_agent_instance('agent1', config, [])

        builder.model_manager.build_model_settings.assert_called_once_with(config['model'])
        mock_ms.assert_called_once()
        kwargs = mock_ms.call_args.kwargs
        assert kwargs['include_usage'] is True
        assert kwargs['temperature'] == 0.3
        assert kwargs['top_p'] == 0.9
        mock_agent_cls.assert_called_once()

def test_create_agent_instance_model_settings_failure_is_safe(builder):
    # If settings derivation fails, the agent must still build with include_usage.
    builder.tool_registry.enable_lazy_loading = False
    builder.model_manager.build_model_settings = MagicMock(side_effect=Exception("boom"))
    builder.structured_output_model_registry.get_model = MagicMock(return_value=None)

    config = {'system_prompt': 'prompt', 'model': {'model_id': 'gpt-4'}}

    with patch('oai_agent_core.openai_core.builders.agent_builder.Agent'), \
         patch('oai_agent_core.openai_core.builders.agent_builder.ModelSettings') as mock_ms:
        builder._create_agent_instance('agent1', config, [])

        kwargs = mock_ms.call_args.kwargs
        assert kwargs == {'include_usage': True}

def test_create_single_agent(builder):
    async def run():
        config = {
            'system_prompt': 'test prompt',
            'tools': ['t1']
        }

        builder.tool_registry.get_tools_for_agent.return_value = ['tool_obj']
        builder.tool_registry.get_mcp_configs.return_value = {}
        builder.tool_registry.load_mcp_tools_from_config = AsyncMock()

        with patch('oai_agent_core.openai_core.builders.agent_builder.Agent') as mock_agent_cls:
            mock_agent = MagicMock()
            mock_agent_cls.return_value = mock_agent
            mock_agent_cls.model_manager.get_model = MagicMock(return_value={'provider': 'openai'})

            agent = await builder.create_single_agent("test", config)

            assert agent == mock_agent
            builder.tool_registry.get_tools_for_agent.assert_called_with(['t1'])
            mock_agent_cls.assert_called_once()

    asyncio.run(run())


def test_create_single_agent_with_mcp(builder):
    async def run():
        config = {
            'system_prompt': 'test',
            'mcps': {'s1': {}}
        }

        builder.tool_registry.load_mcp_tools_from_config = AsyncMock()

        with patch('oai_agent_core.openai_core.builders.agent_builder.Agent'):
            await builder.create_single_agent("test", config)

            builder.tool_registry.load_mcp_tools_from_config.assert_called()

    asyncio.run(run())


def test_create_agent_as_tool(builder):
    mock_agent = MagicMock()

    with patch('oai_agent_core.openai_core.builders.agent_builder.function_tool') as mock_decorator:
        # The decorator returns the function it decorates (or a wrapper)
        def side_effect(func):
            func.name = "tool_name"
            func.description = "desc"
            return func

        mock_decorator.side_effect = side_effect

        tool = builder._create_agent_as_tool(mock_agent, "tool_name", "desc")

        assert tool.name == "tool_name"
        assert tool.description == "desc"
        assert asyncio.iscoroutinefunction(tool)


def test_create_multi_agent_system_handoff(builder):
    async def run():
        configs = [{'a1': {}}, {'a2': {}}]

        # Mock ConfigManager
        with patch('oai_agent_core.components.configuration.model_config.ConfigManager') as mock_cm_cls:
            mock_cm = MagicMock()
            mock_cm.ruamel_to_native.return_value = {'a1': {}, 'a2': {}}
            mock_cm_cls.return_value = mock_cm

            # Mock create_single_agent
            builder.create_single_agent = AsyncMock(side_effect=[MagicMock(), MagicMock()])

            with patch('oai_agent_core.openai_core.builders.agent_builder.Agent') as mock_supervisor_cls:
                supervisor, agents = await builder.create_multi_agent_system(
                    configs, crew_config={"pattern": "handoff"}
                )

                assert len(agents) == 2
                assert supervisor is not None
                mock_supervisor_cls.assert_called_once()

    asyncio.run(run())


def test_create_multi_agent_system_supervisor(builder):
    async def run():
        configs = [{'a1': {}}, {'a2': {}}]

        with patch('oai_agent_core.components.configuration.model_config.ConfigManager') as mock_cm_cls:
            mock_cm = MagicMock()
            mock_cm.ruamel_to_native.return_value = {'a1': {}, 'a2': {}}
            mock_cm_cls.return_value = mock_cm

            builder.create_single_agent = AsyncMock(side_effect=[MagicMock(), MagicMock()])

            with patch('oai_agent_core.openai_core.builders.agent_builder.Agent') as mock_supervisor_cls:
                mock_supervisor = MagicMock()
                mock_supervisor_cls.return_value = mock_supervisor

                supervisor, agents = await builder.create_multi_agent_system(
                    configs, crew_config={"pattern": "supervisor"}
                )

                assert len(agents) == 2

    asyncio.run(run())


def test_create_multi_agent_system_agent_as_tool(builder):
    async def run():
        configs = [{'a1': {}}]

        with patch('oai_agent_core.components.configuration.model_config.ConfigManager') as mock_cm_cls:
            mock_cm = MagicMock()
            mock_cm.ruamel_to_native.return_value = {'a1': {}}
            mock_cm_cls.return_value = mock_cm

            builder.create_single_agent = AsyncMock(return_value=MagicMock())
            builder._create_agent_as_tool = MagicMock(return_value="tool_obj")

            with patch('oai_agent_core.openai_core.builders.agent_builder.Agent') as mock_supervisor_cls:
                supervisor, agents = await builder.create_multi_agent_system(
                    configs, crew_config={"pattern": "agent-as-tool"}
                )

                assert len(agents) == 1
                # Supervisor should be created with tools
                call_kwargs = mock_supervisor_cls.call_args[1]
                assert call_kwargs['tools'] == ["tool_obj"]

    asyncio.run(run())
