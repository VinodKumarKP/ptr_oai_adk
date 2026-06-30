from unittest.mock import MagicMock, patch, AsyncMock
import json
import logging

import pytest

from oai_agent_evaluator.runner import RegressionRunner
from oai_agent_evaluator.scenario import TestScenario
from oai_agent_evaluator.evaluator import AgentEvaluator


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
    class DummyAgentClass:
        pass
    s1 = TestScenario(name="S1", description="D", input_message="I", expected_output="O")
    s2 = TestScenario(name="S2", description="D", input_message="I", expected_output="O", agent_class="mod.OtherAgent")
    s3 = TestScenario(name="S3", description="D", input_message="I", expected_output="O", agent_class=DummyAgentClass)

    mock_loader_cls.load_from_file.return_value = [s1, s2, s3]

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

            # Should create 3 evaluators (one for default, one for string path, one for class object)
            assert mock_evaluator_cls.call_count == 3


def test_run_sync_success():
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")
    # Mock run_async to return True
    with patch('asyncio.run', return_value=True):
        with patch('oai_agent_evaluator.runner.sys.exit') as mock_exit:
            runner.run("path")
            mock_exit.assert_called_with(0)


@pytest.mark.asyncio
async def test_concurrent_failures(mock_agent_class):
    """Test that suite handles partial failures with concurrent execution."""
    runner = RegressionRunner(
        agent_class=mock_agent_class,
        project_root="/tmp",
        max_concurrency=2
    )

    # Create scenarios that will have mixed pass/fail results
    scenarios = [
        TestScenario(
            name="Passing Scenario",
            description="Should pass",
            input_message="Input1",
            expected_output="Output1",
            metrics=["correctness"]
        ),
        TestScenario(
            name="Failing Scenario",
            description="Should fail",
            input_message="Input2",
            expected_output="Output2",
            metrics=["correctness"]
        )
    ]

    # Mock judge responses
    pass_response = {'correctness': {'score': 8.5, 'explanation': 'Good'}}
    fail_response = {'correctness': {'score': 5.0, 'explanation': 'Poor'}}

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    # Return same response for both invocations
    mock_subject.ainvoke.side_effect = [
        {'content': [{'text': 'Response1'}]},
        {'content': [{'text': 'Response2'}]}
    ]

    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    # Return different judge responses
    mock_judge.ainvoke.side_effect = [
        {'content': [{'text': json.dumps(pass_response)}]},
        {'content': [{'text': json.dumps(fail_response)}]}
    ]

    # Side effect: judge, subject, judge (reused), subject (reused)
    mock_agent_class.side_effect = [mock_judge, mock_subject]

    # Manually call run_suite instead of full runner to avoid filesystem
    evaluator = AgentEvaluator(
        agent_class=mock_agent_class,
        project_root="/tmp",
        max_concurrency=2
    )

    results = await evaluator.run_suite(scenarios)

    # Check results
    assert len(results) == 2
    assert results[0]['passed'] is True
    assert results[1]['passed'] is False
    assert results[0]['score'] == 8.5
    assert results[1]['score'] == 5.0


def test_runner_logger_and_missing_paths(mock_loader_cls):
    """Test logger initialization and non-existent paths."""
    # When logger is None, check standard logger creation
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp", logger=None)
    assert runner.logger is not None
    assert runner.logger.name == "RegressionRunner"

    # When logger is explicitly provided
    custom_logger = logging.getLogger("Custom")
    runner_custom = RegressionRunner(agent_class=MagicMock(), project_root="/tmp", logger=custom_logger)
    assert runner_custom.logger == custom_logger

    # Path not found warning
    with patch('os.path.exists', return_value=False):
        async def run():
            return await runner.run_async("non_existent_file.yaml")
        import asyncio
        res = asyncio.run(run())
        assert res is False


@pytest.mark.asyncio
async def test_runner_load_from_directory_and_defaults(mock_evaluator_cls, mock_loader_cls, mock_reporter_cls):
    """Test loader handles directories and runner falls back to defaults."""
    # 1. Directory loading path
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")
    mock_loader_cls.load_from_directory.return_value = [
        TestScenario(name="S1", description="D", input_message="I", expected_output="O")
    ]
    mock_evaluator_cls.return_value.run_suite = AsyncMock(return_value=[{
        'passed': True,
        'score': 10,
        'scenario': 'S1',
        'input': 'I',
        'actual_output': 'O',
        'expected_output': 'O',
        'explanation': 'Good'
    }])

    with patch('os.path.exists', return_value=True), patch('os.path.isdir', return_value=True):
        success = await runner.run_async("some_dir")
    assert success is True
    mock_loader_cls.load_from_directory.assert_called_once()

    # 2. No scenarios path, falls back to default_scenarios
    scenario_default = TestScenario(name="S_Default", description="D", input_message="I", expected_output="O")
    runner_default = RegressionRunner(
        agent_class=MagicMock(),
        project_root="/tmp",
        default_scenarios=[scenario_default]
    )
    mock_evaluator_cls.return_value.run_suite = AsyncMock(return_value=[{
        'passed': True,
        'score': 10,
        'scenario': 'S_Default',
        'input': 'I',
        'actual_output': 'O',
        'expected_output': 'O',
        'explanation': 'Good'
    }])
    success_default = await runner_default.run_async(None)
    assert success_default is True


@pytest.mark.asyncio
async def test_dynamic_class_loading_exception_fallback(mock_evaluator_cls, mock_loader_cls, mock_reporter_cls):
    """Test dynamic class loading exception fallback to default class."""
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")
    
    scenario = TestScenario(
        name="S1",
        description="D",
        input_message="I",
        expected_output="O",
        agent_class="invalid.PythonPath"
    )
    mock_loader_cls.load_from_file.return_value = [scenario]
    
    mock_evaluator_cls.return_value.run_suite = AsyncMock(return_value=[{
        'passed': True,
        'score': 10,
        'scenario': 'S1',
        'input': 'I',
        'actual_output': 'O',
        'expected_output': 'O',
        'explanation': 'Good'
    }])

    with patch('os.path.exists', return_value=True), patch('os.path.isfile', return_value=True):
        success = await runner.run_async("test.yaml")
    
    assert success is True
    # Evaluator should be created with runner's default agent_class
    mock_evaluator_cls.assert_called_with(
        agent_class=runner.agent_class,
        project_root="/tmp",
        judge_model_id="gpt-4o",
        logger=runner.logger,
        max_concurrency=1,
        pass_threshold=7.0
    )


def test_run_sync_failures_and_interrupt():
    """Test synchronous entry point failures and KeyboardInterrupt handling."""
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")
    
    # 1. success = False exits with 1
    with patch('asyncio.run', return_value=False):
        with patch('oai_agent_evaluator.runner.sys.exit', side_effect=SystemExit) as mock_exit:
            with pytest.raises(SystemExit):
                runner.run("path")
            mock_exit.assert_called_with(1)

    # 2. KeyboardInterrupt exits with 130
    with patch('asyncio.run', side_effect=KeyboardInterrupt):
        with patch('oai_agent_evaluator.runner.sys.exit', side_effect=SystemExit) as mock_exit:
            with pytest.raises(SystemExit):
                runner.run("path")
            mock_exit.assert_called_with(130)


def test_report_results_with_error():
    """Test report results rendering when error key is present."""
    runner = RegressionRunner(agent_class=MagicMock(), project_root="/tmp")
    
    results = [
        {
            'passed': True,
            'score': 9.0,
            'scenario': 'S1',
            'error': 'Some non-fatal warning or error info'
        }
    ]
    # Should print results and return True (since passed is True)
    success = runner._report_results(results)
    assert success is True

