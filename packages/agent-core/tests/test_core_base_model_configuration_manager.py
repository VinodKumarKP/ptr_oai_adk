import pytest
from unittest.mock import MagicMock
from oai_agent_core.core.base_model_configuration_manager import BaseModelConfigurationManager

class ConcreteModelManager(BaseModelConfigurationManager):
    def create_model(self, model_config=None):
        return "model"

@pytest.fixture
def manager():
    return ConcreteModelManager()

def test_init(manager):
    assert manager.default_config['model_id'] == BaseModelConfigurationManager.DEFAULT_MODEL_ID
    assert manager.default_config['params']['temperature'] == BaseModelConfigurationManager.DEFAULT_TEMPERATURE

def test_init_with_override():
    override = {'model_id': 'custom', 'params': {'temperature': 0.5}}
    manager = ConcreteModelManager(default_config=override)
    assert manager.default_config['model_id'] == 'custom'
    assert manager.default_config['params']['temperature'] == 0.5

def test_build_default_config_legacy_params():
    override = {'temperature': 0.7, 'max_tokens': 100}
    manager = ConcreteModelManager(default_config=override)
    assert manager.default_config['params']['temperature'] == 0.7
    assert manager.default_config['params']['max_tokens'] == 100

def test_get_default_config(manager):
    config = manager.get_default_config()
    assert config == manager.default_config
    assert config is not manager.default_config # Should be a copy

def test_merge_with_defaults(manager):
    config = {'model_id': 'new', 'params': {'temperature': 0.9}}
    merged = manager._merge_with_defaults(config)
    assert merged['model_id'] == 'new'
    assert merged['params']['temperature'] == 0.9
    assert merged['params']['max_tokens'] == manager.default_config['params']['max_tokens']

def test_validate_config_valid(manager):
    config = {
        'model_id': 'model',
        'params': {'temperature': 0.5, 'max_tokens': 100}
    }
    manager._validate_config(config) # Should not raise

def test_validate_config_invalid_temp(manager):
    config = {
        'model_id': 'model',
        'params': {'temperature': 2.0, 'max_tokens': 100}
    }
    with pytest.raises(ValueError, match="Temperature must be between 0 and 1"):
        manager._validate_config(config)

def test_validate_config_invalid_tokens(manager):
    config = {
        'model_id': 'model',
        'params': {'temperature': 0.5, 'max_tokens': -1}
    }
    with pytest.raises(ValueError, match="max_tokens must be a positive integer"):
        manager._validate_config(config)

def test_update_default_config(manager):
    manager.update_default_config(temperature=0.8, model_id='updated')
    assert manager.default_config['params']['temperature'] == 0.8
    assert manager.default_config['model_id'] == 'updated'

def test_get_model_info(manager):
    info = manager.get_model_info()
    assert info['model_id'] == manager.default_config['model_id']
    assert 'provider' in info
    assert 'model_family' in info

def test_extract_provider(manager):
    assert manager._extract_provider('gpt-4') == 'openai'
    assert manager._extract_provider('claude-3') == 'anthropic'
    assert manager._extract_provider('bedrock/titan') == 'aws-bedrock'
    assert manager._extract_provider('unknown') == 'unknown'

def test_extract_model_family(manager):
    assert manager._extract_model_family('gpt-4') == 'gpt'
    assert manager._extract_model_family('claude-3') == 'claude'
    assert manager._extract_model_family('unknown') == 'unknown'

def test_list_supported_parameters(manager):
    params = manager.list_supported_parameters()
    assert 'temperature' in params
    assert 'max_tokens' in params
