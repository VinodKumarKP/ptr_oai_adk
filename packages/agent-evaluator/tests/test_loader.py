import yaml
import os
import tempfile

from oai_agent_evaluator.loader import ScenarioLoader


def test_load_from_file_single(tmp_path):
    file_path = tmp_path / "test.yaml"
    data = {
        'name': 'Test 1',
        'description': 'Desc',
        'input_message': 'Input',
        'expected_output': 'Output'
    }
    file_path.write_text(yaml.dump(data))

    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert len(scenarios) == 1
    assert scenarios[0].name == 'Test 1'
    assert scenarios[0].agent_name == 'test'  # Defaults to filename


def test_load_from_file_list(tmp_path):
    file_path = tmp_path / "test_list.yaml"
    data = {
        'agent_config': {'model': 'gpt-4'},
        'scenarios': [
            {
                'name': 'S1',
                'input_message': 'I1',
                'expected_output': 'O1'
            },
            {
                'name': 'S2',
                'input_message': 'I2',
                'expected_output': 'O2',
                'metrics': ['safety']
            }
        ]
    }
    file_path.write_text(yaml.dump(data))

    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert len(scenarios) == 2
    assert scenarios[0].agent_config['model'] == 'gpt-4'
    assert scenarios[1].metrics == ['safety']
    assert scenarios[0].metrics == ['correctness', 'relevance', 'safety']  # Default


def test_load_external_config(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: gpt-3.5")

    file_path = tmp_path / "test_ext.yaml"
    data = {
        'agent_config': 'config.yaml',
        'scenarios': [{'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'}]
    }
    file_path.write_text(yaml.dump(data))

    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert scenarios[0].agent_config['model'] == 'gpt-3.5'


def test_pass_threshold_global(tmp_path):
    """Test global pass_threshold configuration."""
    file_path = tmp_path / "test_threshold.yaml"
    data = {
        'pass_threshold': 8.5,
        'scenarios': [
            {'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'},
            {'name': 'S2', 'input_message': 'I2', 'expected_output': 'O2', 'pass_threshold': 9.0}
        ]
    }
    file_path.write_text(yaml.dump(data))

    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert len(scenarios) == 2
    assert scenarios[0].pass_threshold == 8.5  # Global
    assert scenarios[1].pass_threshold == 9.0  # Per-scenario override


def test_env_var_expansion(tmp_path):
    """Test environment variable expansion in config file paths."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: gpt-4o")

    # Set environment variable
    os.environ['TEST_CONFIG_DIR'] = str(tmp_path)

    file_path = tmp_path / "test_env.yaml"
    data = {
        'agent_config': '$TEST_CONFIG_DIR/config.yaml',
        'scenarios': [{'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'}]
    }
    file_path.write_text(yaml.dump(data))

    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert scenarios[0].agent_config['model'] == 'gpt-4o'

    # Cleanup
    del os.environ['TEST_CONFIG_DIR']


def test_absolute_path_config(tmp_path):
    """Test absolute path support in config file references."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model: claude-3")

    file_path = tmp_path / "test_abs.yaml"
    data = {
        'agent_config': str(config_path),  # Absolute path
        'scenarios': [{'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'}]
    }
    file_path.write_text(yaml.dump(data))

    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert scenarios[0].agent_config['model'] == 'claude-3'
