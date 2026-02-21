from unittest.mock import patch, MagicMock

import pytest

from oai_agent_evaluator.cli import main, load_agent_class


def test_load_agent_class_success():
    with patch('importlib.import_module') as mock_import:
        mock_module = MagicMock()
        mock_class = MagicMock()
        setattr(mock_module, 'MyAgent', mock_class)
        mock_import.return_value = mock_module

        cls = load_agent_class('my_module.MyAgent')
        assert cls == mock_class
        mock_import.assert_called_with('my_module')


def test_load_agent_class_failure():
    with patch('importlib.import_module', side_effect=ImportError("No module")):
        with pytest.raises(SystemExit):
            load_agent_class('bad.module.Agent')


def test_main_success():
    test_args = [
        'oai-agent-evaluator',
        'tests/scenarios.yaml',
        '--agent-class', 'mod.Agent',
        '--project-root', '/tmp'
    ]

    with patch('sys.argv', test_args):
        with patch('oai_agent_evaluator.cli.load_agent_class') as mock_load:
            with patch('oai_agent_evaluator.cli.RegressionRunner') as mock_runner_cls:
                mock_runner = mock_runner_cls.return_value

                main()

                mock_load.assert_called_with('mod.Agent')
                mock_runner_cls.assert_called_once()
                mock_runner.run.assert_called_with('tests/scenarios.yaml')
