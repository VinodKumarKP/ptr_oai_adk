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


import pytest
import re

def test_loader_schema_validations(tmp_path):
    """Verify all loader validation errors for scenarios."""
    def verify_validation_error(bad_scenario, expected_err, error_type=ValueError):
        file_path = tmp_path / "bad.yaml"
        file_path.write_text(yaml.dump({'scenarios': [bad_scenario]}))
        with pytest.raises(error_type, match=re.escape(expected_err)):
            ScenarioLoader.load_from_file(str(file_path))

    # name validations
    verify_validation_error({'input_message': 'I'}, "Scenario missing required field 'name'")
    verify_validation_error({'name': 123, 'input_message': 'I'}, "Scenario 'name' must be non-empty string", ValueError)
    verify_validation_error({'name': ' ', 'input_message': 'I'}, "Scenario 'name' must be non-empty string")

    # input_message validations
    verify_validation_error({'name': 'S'}, "missing required field 'input_message'")
    verify_validation_error({'name': 'S', 'input_message': 123}, "input_message' must be non-empty string", ValueError)
    verify_validation_error({'name': 'S', 'input_message': ' '}, "input_message' must be non-empty string")

    # evaluation fields validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 123}, "expected_output' must be string or None", TypeError)
    verify_validation_error({'name': 'S', 'input_message': 'I', 'evaluation_criteria': 123}, "evaluation_criteria' must be string or None", TypeError)

    # metrics validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'metrics': 'not-list'}, "metrics' must be a list", TypeError)
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'metrics': []}, "metrics' cannot be empty list")
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'metrics': [123]}, "metrics must be non-empty strings")
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'metrics': [' ']}, "metrics must be non-empty strings")

    # agent_config validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'agent_config': 123}, "agent_config' must be dict or string path", TypeError)

    # config_overrides validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'config_overrides': 123}, "config_overrides' must be dict or None", TypeError)

    # agent_class validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'agent_class': 123}, "agent_class' must be string path", TypeError)
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'agent_class': 'invalid path!'}, "agent_class' invalid Python path")

    # agent_model_config validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'agent_model_config': 123}, "agent_model_config' must be dict, list, or None", TypeError)
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'agent_model_config': []}, "agent_model_config' list cannot be empty")

    # judge_model_id validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'judge_model_id': 123}, "judge_model_id' must be non-empty string or None", ValueError)
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'judge_model_id': ' '}, "judge_model_id' must be non-empty string or None")

    # pass_threshold validations
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'pass_threshold': 'not-num'}, "pass_threshold' must be number or None", TypeError)
    verify_validation_error({'name': 'S', 'input_message': 'I', 'expected_output': 'O', 'pass_threshold': 11.0}, "pass_threshold' must be between 0 and 10")


def test_loader_load_from_directory(tmp_path):
    """Test loading multiple scenarios from a directory."""
    # Write some yaml files
    (tmp_path / "s1.yaml").write_text(yaml.dump({'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'}))
    (tmp_path / "s2.yaml").write_text(yaml.dump({'name': 'S2', 'input_message': 'I2', 'expected_output': 'O2'}))
    (tmp_path / "ignored.txt").write_text("some text")

    scenarios = ScenarioLoader.load_from_directory(str(tmp_path))
    assert len(scenarios) == 2
    names = {s.name for s in scenarios}
    assert names == {'S1', 'S2'}


def test_loader_edge_cases(tmp_path):
    """Verify various loader boundary scenarios and configs."""
    # 1. External agent config not found
    file_path = tmp_path / "missing_config_ref.yaml"
    data = {
        'agent_config': 'non_existent_config.yaml',
        'scenarios': [{'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'}]
    }
    file_path.write_text(yaml.dump(data))
    with pytest.raises(FileNotFoundError, match="Agent config file not found"):
        ScenarioLoader.load_from_file(str(file_path))

    # 2. Document is a list directly
    file_path = tmp_path / "list_doc.yaml"
    list_data = [
        {'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'},
        {'name': 'S2', 'input_message': 'I2', 'expected_output': 'O2'}
    ]
    file_path.write_text(yaml.dump(list_data))
    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert len(scenarios) == 2

    # 3. Scenarios points to a single dict instead of a list
    file_path = tmp_path / "single_dict_scenarios.yaml"
    single_dict_data = {
        'scenarios': {'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'}
    }
    file_path.write_text(yaml.dump(single_dict_data))
    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert len(scenarios) == 1
    assert scenarios[0].name == 'S1'

    # 4. Multi-document YAML with some empty docs, and non-dict items
    file_path = tmp_path / "multi_doc.yaml"
    multi_doc_text = """
---
# empty doc
---
name: S1
input_message: I1
expected_output: O1
---
scenarios:
  - "not a dict item"
  - name: S2
    input_message: I2
    expected_output: O2
"""
    file_path.write_text(multi_doc_text)
    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert len(scenarios) == 2
    assert {s.name for s in scenarios} == {'S1', 'S2'}

    # 5. Global configurations validation and matrix testing
    file_path = tmp_path / "globals.yaml"
    globals_data = {
        'agent_name': 'GlobalAgent',
        'agent_class': 'my_module.MyClass',
        'metrics': ['safety'],
        'judge_model_id': 'claude-3-opus',
        'agent_model_config': [
            {'model_id': 'gpt-4'},
            {'model_id': 'gpt-3.5'}
        ],
        'scenarios': [
            {'name': 'S1', 'input_message': 'I1', 'expected_output': 'O1'}
        ]
    }
    file_path.write_text(yaml.dump(globals_data))
    scenarios = ScenarioLoader.load_from_file(str(file_path))
    assert len(scenarios) == 2  # Matrix test generated 2 instances
    assert scenarios[0].agent_name == 'GlobalAgent'
    assert scenarios[0].agent_class == 'my_module.MyClass'
    assert scenarios[0].metrics == ['safety']
    assert scenarios[0].judge_model_id == 'claude-3-opus'
    assert scenarios[0].agent_model_config == {'model_id': 'gpt-4'}
    assert scenarios[1].agent_model_config == {'model_id': 'gpt-3.5'}

