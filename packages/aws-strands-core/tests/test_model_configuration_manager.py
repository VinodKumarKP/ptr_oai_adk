import pytest
from unittest.mock import patch
from oai_agent_core.aws_strands_core.components.configuration.model_config import StrandsModelConfigurationManager

def test_create_model():
    manager = StrandsModelConfigurationManager()
    config = {
        'model_id': 'gpt-4',
        'params': {'temperature': 0.7, 'max_tokens': 100}
    }
    
    with patch('strands.models.litellm.LiteLLMModel') as MockModel:
        model = manager.create_model(config)
        
        MockModel.assert_called_once_with(
            model_id='gpt-4',
            params={'temperature': 0.7, 'max_tokens': 100}
        )
        assert model == MockModel.return_value

def test_create_model_default_config():
    default_config = {
        'model_id': 'claude-3',
        'params': {'temperature': 0.5, 'max_tokens': 200}
    }
    manager = StrandsModelConfigurationManager(default_config=default_config)
    
    with patch('strands.models.litellm.LiteLLMModel') as MockModel:
        manager.create_model()
        
        MockModel.assert_called_once_with(
            model_id='claude-3',
            params={'temperature': 0.5, 'max_tokens': 200}
        )

def test_create_model_error():
    manager = StrandsModelConfigurationManager()
    config = {'model_id': 'gpt-4', 'params': {}}
    
    with patch('strands.models.litellm.LiteLLMModel', side_effect=Exception("Error")):
        with pytest.raises(Exception, match="Error"):
            manager.create_model(config)
