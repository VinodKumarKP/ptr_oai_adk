from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from oai_agent_evaluator.runner import RegressionRunner
from oai_agent_evaluator.scenario import TestScenario


@pytest.fixture
def mock_evaluator_cls():
    with patch('oai_agent_evaluator.runner.AgentEvaluator') as mock:
        yield mock


@pytest.fixture
def mock_loader_cls():
    with patch('oai_agent_evaluator.runner.ScenarioLoader') as mock:
        yield mock


@pytest.fixture
def mock_reporter_cls():
    with patch('oai_agent_evaluator.runner.HtmlReporter') as mock:
        yield mock


@pytest.mark.asyncio
async def test_run_async_success(mock_evaluator_cls, mock_loader_cls, mock_reporter_cls):
    # Setup
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")

    # Mock scenarios
    mock_loader_cls.load_from_file.return_value = [
        TestScenario(name="S1", description="D", input_message="I", expected_output="O")
    ]

    # Mock evaluation result with all required fields
    mock_instance = mock_evaluator_cls.return_value
    mock_instance.run_suite = AsyncMock(return_value=[{
        'passed': True,
        'score': 10,
        'scenario': 'S1',
        'input': 'I',
        'actual_output': 'O',
        'expected_output': 'O',
        'explanation': 'Good'
    }])

    # Run
    with patch('os.path.exists', return_value=True), patch('os.path.isfile', return_value=True):
        success = await runner.run_async("test.yaml")

    assert success is True
    mock_loader_cls.load_from_file.assert_called_once()
    mock_instance.run_suite.assert_called_once()
    mock_reporter_cls.return_value.generate_report.assert_called_once()


@pytest.mark.asyncio
async def test_run_async_failure(mock_evaluator_cls, mock_loader_cls, mock_reporter_cls):
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")

    mock_loader_cls.load_from_file.return_value = [
        TestScenario(name="S1", description="D", input_message="I", expected_output="O")
    ]

    mock_instance = mock_evaluator_cls.return_value
    mock_instance.run_suite = AsyncMock(return_value=[{
        'passed': False,
        'score': 0,
        'scenario': 'S1',
        'input': 'I',
        'actual_output': 'X',
        'expected_output': 'O',
        'explanation': 'Bad'
    }])

    with patch('os.path.exists', return_value=True), patch('os.path.isfile', return_value=True):
        success = await runner.run_async("test.yaml")

    assert success is False


def test_load_agent_class_dynamic():
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")

    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        mock_class = MagicMock()
        setattr(mock_module, 'MyAgent', mock_class)
        mock_import.return_value = mock_module

        cls = runner._load_agent_class('mod.MyAgent')
        assert cls == mock_class


def test_load_agent_class_error():
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")
    with patch('importlib.import_module', side_effect=ImportError):
        with pytest.raises(ImportError):
            runner._load_agent_class('bad.mod')


@pytest.mark.asyncio
async def test_run_async_multi_agent(mock_evaluator_cls, mock_loader_cls, mock_reporter_cls):
    # Setup scenarios with different agent classes
    s1 = TestScenario(name="S1", description="D", input_message="I", expected_output="O")
    s2 = TestScenario(name="S2", description="D", input_message="I", expected_output="O", agent_class="mod.OtherAgent")

    mock_loader_cls.load_from_file.return_value = [s1, s2]

    # Mock evaluator run with complete result structure
    eval_instance = mock_evaluator_cls.return_value
    eval_instance.run_suite = AsyncMock(return_value=[{
        'passed': True,
        'score': 10,
        'scenario': 'S1',
        'input': 'I',
        'actual_output': 'O',
        'expected_output': 'O',
        'explanation': 'Good'
    }])

    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")

    # Mock dynamic loading
    with patch.object(runner, '_load_agent_class', return_value=MagicMock()) as mock_load:
        with patch('os.path.exists', return_value=True), patch('os.path.isfile', return_value=True):
            await runner.run_async("test.yaml")

            # Should load the other agent class
            mock_load.assert_called_with("mod.OtherAgent")

            # Should create 2 evaluators (one for default, one for other)
            assert mock_evaluator_cls.call_count == 2


def test_run_sync_success():
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")
    # Mock run_async to return True
    with patch.object(runner, 'run_async', return_value=True) as mock_async:
        # Patch asyncio.run to just return the result of the coroutine (which is our mock return value)
        with patch('asyncio.run', side_effect=lambda x: x):
            with patch('sys.exit') as mock_exit:
                runner.run("path")
                mock_exit.assert_called_with(0)
