import pytest
import os
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from oai_agent_core.components.configuration.model_config import ConfigManager

@pytest.fixture
def config_manager():
    return ConfigManager(config_root="/tmp/config")

def test_init(config_manager):
    assert config_manager.config_root == Path("/tmp/config")
    assert config_manager.agent_config_dir == Path("/tmp/config/agents_config")

def test_ruamel_to_native(config_manager):
    # Mock ruamel objects if possible, or just test with dicts/lists
    data = {'a': 1, 'b': [2, 3]}
    result = config_manager.ruamel_to_native(data)
    assert result == data

def test_load_agent_config_cached(config_manager):
    config_manager._agent_config_cache['test_agent'] = {'type': 'cached'}
    config = config_manager.load_agent_config('test_agent')
    assert config == {'type': 'cached'}

def test_load_agent_config_file_not_found(config_manager):
    with patch.object(Path, 'exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            config_manager.load_agent_config('test_agent')

def test_load_agent_config_success(config_manager):
    yaml_content = "type: test_agent\nmodel:\n  temperature: 0.5"
    with patch.object(Path, 'exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml_content)):
        
        config = config_manager.load_agent_config('test_agent')
        assert config['type'] == 'test_agent'
        assert config['model']['temperature'] == 0.5
        assert 'test_agent' in config_manager._agent_config_cache

def test_load_global_config_success(config_manager):
    yaml_content = "model:\n  temperature: 0.1"
    with patch.object(Path, 'exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml_content)):
        
        config = config_manager.load_global_config('aws')
        assert config['model']['temperature'] == 0.1

def test_merge_configs(config_manager):
    base = {'a': 1, 'b': {'c': 2}}
    override = {'b': {'d': 3}, 'e': 4}
    merged = config_manager.merge_configs(base, override)
    assert merged == {'a': 1, 'b': {'c': 2, 'd': 3}, 'e': 4}

def test_validate_config_value_valid(config_manager):
    assert config_manager.validate_config_value('temperature', 0.5) == 0.5
    assert config_manager.validate_config_value('max_tokens', 100) == 100

def test_validate_config_value_invalid(config_manager):
    with pytest.raises(ValueError):
        config_manager.validate_config_value('temperature', 3.0)
    with pytest.raises(ValueError):
        config_manager.validate_config_value('max_tokens', -1)

def test_get_config_value(config_manager):
    config = {'a': {'b': 1}}
    assert config_manager.get_config_value(config, 'a.b') == 1
    assert config_manager.get_config_value(config, 'a.c', default=2) == 2

def test_set_config_value(config_manager):
    config = {'a': {'b': 1}}
    config_manager.set_config_value(config, 'a.b', 2)
    assert config['a']['b'] == 2
    
    config_manager.set_config_value(config, 'x.y', 3)
    assert config['x']['y'] == 3

def test_save_agent_config(config_manager):
    config = {'type': 'test'}
    with patch.object(Path, 'mkdir'), \
         patch('builtins.open', mock_open()) as mock_file:
        
        config_manager.save_agent_config('test_agent', config)
        
        mock_file.assert_called_once()
        # Verify yaml dump was called (indirectly via write)
        # Since we use self.yaml.dump(config, f), f.write will be called
        assert mock_file().write.called

def test_list_available_agents(config_manager):
    with patch.object(Path, 'exists', return_value=True), \
         patch.object(Path, 'glob', return_value=[Path('a.yaml'), Path('b.yaml')]):
        
        agents = config_manager.list_available_agents()
        assert sorted(agents) == ['a', 'b']

def test_load_server_config(config_manager):
    with patch('glob.glob', return_value=['/tmp/config/agents_config/agent1.yaml']), \
         patch.object(config_manager, '_load_individual_yaml_config', return_value={'name': 'agent1'}):
        
        configs = config_manager.load_server_config()
        assert 'agent1' in configs
        assert configs['agent1']['name'] == 'agent1'

def test_load_individual_yaml_config(config_manager):
    yaml_content = "name: agent1"
    with patch('builtins.open', mock_open(read_data=yaml_content)):
        config = config_manager._load_individual_yaml_config("file.yaml")
        assert config['name'] == 'agent1'

def test_load_individual_json_config(config_manager):
    json_content = '{"name": "agent1"}'
    with patch('builtins.open', mock_open(read_data=json_content)):
        config = config_manager._load_individual_json_config("file.json")
        assert config['name'] == 'agent1'

def test_load_config_from_path_yaml(config_manager):
    yaml_content = "name: agent1"
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=yaml_content)):
        
        config = config_manager.load_config_from_path("file.yaml")
        assert config['name'] == 'agent1'

def test_load_config_from_path_json(config_manager):
    json_content = '{"name": "agent1"}'
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', mock_open(read_data=json_content)):
        
        config = config_manager.load_config_from_path("file.json")
        assert config['name'] == 'agent1'
