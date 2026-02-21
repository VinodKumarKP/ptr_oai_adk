import pytest
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from ruamel.yaml import comments
from oai_agent_core.components.configuration.model_config import ConfigManager


class TestConfigManagerCore:
    """Core functionality tests for ConfigManager"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager(config_root="/tmp/test")

    def test_init_with_config_root(self):
        cm = ConfigManager(config_root="/custom/path")
        assert cm.config_root == Path("/custom/path")
        assert cm.agent_config_dir == Path("/custom/path/agents_config")

    def test_init_without_config_root(self):
        cm = ConfigManager()
        assert cm.config_root is None

    def test_ruamel_to_native_dict(self, config_manager):
        regular_dict = {'key': 'value', 'nested': {'inner': 'data'}}
        result = config_manager.ruamel_to_native(regular_dict)
        assert result == regular_dict

    def test_ruamel_to_native_list(self, config_manager):
        regular_list = ['item1', 'item2', {'key': 'value'}]
        result = config_manager.ruamel_to_native(regular_list)
        assert result == regular_list

    def test_ruamel_to_native_primitive(self, config_manager):
        assert config_manager.ruamel_to_native("string") == "string"
        assert config_manager.ruamel_to_native(123) == 123
        assert config_manager.ruamel_to_native(True) is True


class TestConfigValidation:
    """Test configuration validation methods"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager()

    def test_validate_temperature_valid(self, config_manager):
        assert config_manager.validate_config_value('temperature', 0.0) == 0.0
        assert config_manager.validate_config_value('temperature', 1.0) == 1.0
        assert config_manager.validate_config_value('temperature', 2.0) == 2.0

    def test_validate_temperature_invalid(self, config_manager):
        with pytest.raises(ValueError, match="Temperature must be a number between 0 and 2"):
            config_manager.validate_config_value('temperature', -0.1)
        
        with pytest.raises(ValueError, match="Temperature must be a number between 0 and 2"):
            config_manager.validate_config_value('temperature', 2.1)
        
        with pytest.raises(ValueError, match="Temperature must be a number between 0 and 2"):
            config_manager.validate_config_value('temperature', "invalid")

    def test_validate_max_tokens_valid(self, config_manager):
        assert config_manager.validate_config_value('max_tokens', 1) == 1
        assert config_manager.validate_config_value('max_tokens', 1000) == 1000

    def test_validate_max_tokens_invalid(self, config_manager):
        with pytest.raises(ValueError, match="Max tokens must be a positive integer"):
            config_manager.validate_config_value('max_tokens', 0)
        
        with pytest.raises(ValueError, match="Max tokens must be a positive integer"):
            config_manager.validate_config_value('max_tokens', -1)
        
        with pytest.raises(ValueError, match="Max tokens must be a positive integer"):
            config_manager.validate_config_value('max_tokens', 1.5)

    def test_validate_agent_type_valid(self, config_manager):
        assert config_manager.validate_config_value('type', 'valid_type', 'agent') == 'valid_type'

    def test_validate_agent_type_invalid(self, config_manager):
        with pytest.raises(ValueError, match="Agent type must be a non-empty string"):
            config_manager.validate_config_value('type', '', 'agent')
        
        with pytest.raises(ValueError, match="Agent type must be a non-empty string"):
            config_manager.validate_config_value('type', 123, 'agent')

    def test_validate_unknown_key(self, config_manager):
        # Should return value unchanged for unknown keys
        assert config_manager.validate_config_value('unknown_key', 'value') == 'value'

    def test_validate_agent_config_valid(self, config_manager):
        config = {'type': 'valid_agent'}
        # Should not raise an exception
        config_manager._validate_agent_config(config, 'test_agent')

    def test_validate_agent_config_missing_type(self, config_manager):
        config = {'model': {}}
        with pytest.raises(ValueError, match="Missing required field 'type'"):
            config_manager._validate_agent_config(config, 'test_agent')

    def test_validate_agent_config_empty_type(self, config_manager):
        config = {'type': ''}
        with pytest.raises(ValueError, match="Invalid agent type"):
            config_manager._validate_agent_config(config, 'test_agent')

    def test_validate_agent_config_non_string_type(self, config_manager):
        config = {'type': 123}
        with pytest.raises(ValueError, match="Invalid agent type"):
            config_manager._validate_agent_config(config, 'test_agent')


class TestConfigOperations:
    """Test configuration manipulation methods"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager()

    def test_get_config_value_simple(self, config_manager):
        config = {'key': 'value'}
        assert config_manager.get_config_value(config, 'key') == 'value'

    def test_get_config_value_nested(self, config_manager):
        config = {'a': {'b': {'c': 'deep_value'}}}
        assert config_manager.get_config_value(config, 'a.b.c') == 'deep_value'

    def test_get_config_value_missing_with_default(self, config_manager):
        config = {'a': {'b': 'value'}}
        assert config_manager.get_config_value(config, 'a.missing', 'default') == 'default'

    def test_get_config_value_type_error(self, config_manager):
        config = {'a': 'string_not_dict'}
        assert config_manager.get_config_value(config, 'a.b', 'default') == 'default'

    def test_set_config_value_simple(self, config_manager):
        config = {}
        config_manager.set_config_value(config, 'key', 'value')
        assert config['key'] == 'value'

    def test_set_config_value_nested_new(self, config_manager):
        config = {}
        config_manager.set_config_value(config, 'a.b.c', 'value')
        assert config['a']['b']['c'] == 'value'

    def test_set_config_value_nested_existing(self, config_manager):
        config = {'a': {'b': 'old_value'}}
        config_manager.set_config_value(config, 'a.b', 'new_value')
        assert config['a']['b'] == 'new_value'

    def test_merge_configs_simple(self, config_manager):
        base = {'a': 1, 'b': 2}
        override = {'b': 3, 'c': 4}
        result = config_manager.merge_configs(base, override)
        assert result == {'a': 1, 'b': 3, 'c': 4}

    def test_merge_configs_nested(self, config_manager):
        base = {'a': {'x': 1, 'y': 2}, 'b': 3}
        override = {'a': {'y': 20, 'z': 30}, 'c': 4}
        result = config_manager.merge_configs(base, override)
        expected = {'a': {'x': 1, 'y': 20, 'z': 30}, 'b': 3, 'c': 4}
        assert result == expected

    def test_merge_configs_override_with_non_dict(self, config_manager):
        base = {'a': {'nested': 'value'}}
        override = {'a': 'string_replacement'}
        result = config_manager.merge_configs(base, override)
        assert result == {'a': 'string_replacement'}


class TestCacheManagement:
    """Test configuration caching functionality"""
    
    @pytest.fixture
    def config_manager(self):
        cm = ConfigManager()
        # Pre-populate caches for testing
        cm._agent_config_cache['agent1'] = {'type': 'test1'}
        cm._agent_config_cache['agent2'] = {'type': 'test2'}
        cm._global_config_cache['aws'] = {'model': {'temperature': 0.1}}
        cm._global_config_cache['azure'] = {'model': {'temperature': 0.2}}
        return cm

    def test_clear_cache_all(self, config_manager):
        config_manager.clear_cache()
        assert len(config_manager._agent_config_cache) == 0
        assert len(config_manager._global_config_cache) == 0

    def test_clear_cache_agent_only(self, config_manager):
        config_manager.clear_cache('agent')
        assert len(config_manager._agent_config_cache) == 0
        assert len(config_manager._global_config_cache) == 2

    def test_clear_cache_global_only(self, config_manager):
        config_manager.clear_cache('global')
        assert len(config_manager._agent_config_cache) == 2
        assert len(config_manager._global_config_cache) == 0


class TestDefaultConfigurations:
    """Test default configuration methods"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager()

    def test_get_default_global_config(self, config_manager):
        default = config_manager._get_default_global_config()
        assert 'model' in default
        assert default['model']['temperature'] == 0.0
        assert default['model']['max_tokens'] == 1000

    def test_apply_global_config_defaults(self, config_manager):
        config = {'model': {'temperature': 0.5}}
        result = config_manager._apply_global_config_defaults(config)
        assert result['model']['temperature'] == 0.5  # Override preserved
        assert result['model']['max_tokens'] == 1000   # Default applied


class TestFileOperations:
    """Test file loading and saving operations with mocking"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager(config_root="/tmp/test")

    def test_load_agent_config_from_cache(self, config_manager):
        # Pre-populate cache
        config_manager._agent_config_cache['cached_agent'] = {'type': 'cached', 'data': 'test'}
        
        result = config_manager.load_agent_config('cached_agent')
        assert result == {'type': 'cached', 'data': 'test'}
        # Verify it's a copy, not the same object
        result['new_key'] = 'new_value'
        assert 'new_key' not in config_manager._agent_config_cache['cached_agent']

    def test_load_agent_config_file_not_found_abort_true(self, config_manager):
        with patch.object(Path, 'exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Agent configuration file not found"):
                config_manager.load_agent_config('nonexistent')

    def test_load_agent_config_file_not_found_abort_false(self, config_manager):
        with patch.object(Path, 'exists', return_value=False):
            result = config_manager.load_agent_config('nonexistent', abort_if_not_found=False)
            assert result == {}

    def test_load_global_config_from_cache(self, config_manager):
        # Pre-populate cache
        config_manager._global_config_cache['test_provider'] = {'model': {'temperature': 0.3}}
        
        result = config_manager.load_global_config('test_provider')
        assert result == {'model': {'temperature': 0.3}}

    def test_load_global_config_file_not_found(self, config_manager):
        with patch.object(Path, 'exists', return_value=False):
            result = config_manager.load_global_config('nonexistent')
            expected = config_manager._get_default_global_config()
            assert result == expected

    def test_list_available_agents_no_directory(self, config_manager):
        with patch.object(Path, 'exists', return_value=False):
            result = config_manager.list_available_agents()
            assert result == []

    def test_list_available_providers_no_directory(self, config_manager):
        with patch.object(Path, 'exists', return_value=False):
            result = config_manager.list_available_providers()
            assert result == []

    def test_load_config_from_path_nonexistent(self, config_manager):
        with patch('os.path.exists', return_value=False):
            result = config_manager.load_config_from_path('nonexistent.yaml')
            assert result == {}

    def test_load_individual_json_config_file_not_found(self, config_manager):
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with pytest.raises(Exception, match="Configuration file not found"):
                config_manager._load_individual_json_config('missing.json')

    def test_load_individual_yaml_config_file_not_found(self, config_manager):
        with patch('builtins.open', side_effect=FileNotFoundError()):
            with pytest.raises(Exception, match="Configuration file not found"):
                config_manager._load_individual_yaml_config('missing.yaml')

    def test_load_individual_json_config_invalid_json(self, config_manager):
        with patch('builtins.open', mock_open(read_data='invalid json')):
            with pytest.raises(Exception, match="Invalid JSON"):
                config_manager._load_individual_json_config('invalid.json')


class TestSuccessfulFileOperations:
    """Test successful file operations with proper mocking"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager(config_root="/tmp/test")

    def test_load_individual_yaml_config_success(self, config_manager):
        yaml_content = "name: test\nvalue: 123"
        with patch('builtins.open', mock_open(read_data=yaml_content)), \
             patch.object(config_manager.yaml, 'load', return_value={'name': 'test', 'value': 123}):
            result = config_manager._load_individual_yaml_config('test.yaml')
            assert result == {'name': 'test', 'value': 123}

    def test_load_individual_json_config_success(self, config_manager):
        json_content = '{"name": "test", "value": 123}'
        with patch('builtins.open', mock_open(read_data=json_content)):
            result = config_manager._load_individual_json_config('test.json')
            assert result == {'name': 'test', 'value': 123}

    def test_load_config_from_path_yaml(self, config_manager):
        yaml_content = "key: value"
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=yaml_content)), \
             patch.object(config_manager.yaml, 'load', return_value={'key': 'value'}):
            result = config_manager.load_config_from_path('test.yaml')
            assert result == {'key': 'value'}

    def test_load_config_from_path_json(self, config_manager):
        json_content = '{"key": "value"}'
        with patch('os.path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=json_content)):
            result = config_manager.load_config_from_path('test.json')
            assert result == {'key': 'value'}

    def test_save_agent_config_success(self, config_manager):
        config = {'type': 'test_agent'}
        with patch.object(Path, 'mkdir'), \
             patch('builtins.open', mock_open()) as mock_file, \
             patch.object(config_manager.yaml, 'dump') as mock_dump:
            
            config_manager.save_agent_config('test', config)
            
            # Verify the file was opened for writing
            mock_file.assert_called_once()
            # Verify yaml.dump was called
            mock_dump.assert_called_once()
            # Verify config was cached
            assert config_manager._agent_config_cache['test'] == config

    def test_save_agent_config_write_error(self, config_manager):
        config = {'type': 'test_agent'}
        with patch.object(Path, 'mkdir'), \
             patch('builtins.open', side_effect=IOError("Write error")):
            with pytest.raises(RuntimeError, match="Failed to save configuration"):
                config_manager.save_agent_config('test', config)


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error conditions"""
    
    @pytest.fixture
    def config_manager(self):
        return ConfigManager()

    def test_load_agent_config_empty_yaml_result(self, config_manager):
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="# empty yaml")), \
             patch.object(config_manager.yaml, 'load', return_value=None):
            with pytest.raises(RuntimeError, match="Failed to load configuration"):
                config_manager.load_agent_config('empty')

    def test_load_agent_config_yaml_exception(self, config_manager):
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid: yaml:")), \
             patch.object(config_manager.yaml, 'load', side_effect=Exception("YAML parse error")):
            with pytest.raises(RuntimeError, match="Failed to load configuration"):
                config_manager.load_agent_config('invalid')

    def test_load_global_config_no_cache(self, config_manager):
        yaml_content = "model:\n  temperature: 0.3"
        with patch.object(Path, 'exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=yaml_content)), \
             patch.object(config_manager.yaml, 'load', return_value={'model': {'temperature': 0.3}}):
            result = config_manager.load_global_config('aws', use_cache=False)
            assert result['model']['temperature'] == 0.3
            # Verify it wasn't cached
            assert 'aws' not in config_manager._global_config_cache

    def test_load_server_config_mixed_formats(self, config_manager):
        files = ['/path/agent1.yaml', '/path/agent2.json']
        with patch('glob.glob', side_effect=[files[:1], files[1:], []]), \
             patch.object(config_manager, '_load_individual_yaml_config', return_value={'yaml': True}), \
             patch.object(config_manager, '_load_individual_json_config', return_value={'json': True}):
            
            result = config_manager.load_server_config()
            assert 'agent1' in result
            assert 'agent2' in result
            assert result['agent1']['yaml'] is True
            assert result['agent2']['json'] is True


# Test the global config_manager instance
def test_global_config_manager_instance():
    from oai_agent_core.components.configuration.model_config import config_manager
    assert isinstance(config_manager, ConfigManager)
    assert config_manager.config_root is None

def test_load_config_from_path_exception_handling():
    config_manager = ConfigManager()
    with patch('os.path.exists', return_value=True), \
         patch('builtins.open', side_effect=Exception("File error")):
        with pytest.raises(Exception, match="Error loading config file"):
            config_manager.load_config_from_path("test.yaml")

def test_load_individual_yaml_config_exception():
    config_manager = ConfigManager()
    with patch('builtins.open', side_effect=Exception("YAML load error")):
        with pytest.raises(Exception, match="Error loading server configuration"):
            config_manager._load_individual_yaml_config("test.yaml")

def test_load_global_config_yaml_specific_exception():
    config_manager = ConfigManager()
    # Test the yaml-specific exception path
    class YamlError(Exception):
        pass
    
    with patch.object(Path, 'exists', return_value=True), \
         patch('builtins.open', mock_open(read_data="test")), \
         patch.object(config_manager.yaml, 'load', side_effect=YamlError("yaml error")):
        # This should trigger the yaml-specific error handling
        result = config_manager.load_global_config('aws')
        expected = config_manager._get_default_global_config()
        assert result == expected