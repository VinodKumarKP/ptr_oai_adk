import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.crewai_core.components.configuration.model_config import CrewAIModelConfigurationManager

@pytest.fixture
def manager():
    return CrewAIModelConfigurationManager()

def test_create_model_default(manager):
    # Mock crewai.LLM
    mock_crewai = MagicMock()
    mock_crewai.LLM = MagicMock()
    
    # Mock the default_config property on the instance
    # Since manager is an instance of CrewAIModelConfigurationManager which inherits from MockBaseModelConfigurationManager (from conftest)
    # And MockBaseModelConfigurationManager has a default_config property returning {}
    # We can just set it on the instance if it wasn't a property, but it is.
    # So we use patch.object on the class or instance.
    
    with patch.object(CrewAIModelConfigurationManager, 'default_config', new_callable=lambda: {'model_id': 'default', 'params': {'temperature': 0.1, 'max_tokens': 100}}):
        with patch.dict('sys.modules', {'crewai': mock_crewai}):
            model = manager.create_model()
            
            mock_crewai.LLM.assert_called_with(
                model='default',
                temperature=0.1,
                max_tokens=100
            )
            assert model == mock_crewai.LLM.return_value

def test_create_model_custom(manager):
    config = {
        'model_id': 'custom-model',
        'params': {'temperature': 0.9, 'max_tokens': 200}
    }
    
    mock_crewai = MagicMock()
    mock_crewai.LLM = MagicMock()
    
    with patch.dict('sys.modules', {'crewai': mock_crewai}):
        model = manager.create_model(config)
        
        mock_crewai.LLM.assert_called_with(
            model='custom-model',
            temperature=0.9,
            max_tokens=200
        )

def test_create_model_import_error(manager):
    # Simulate ImportError when importing crewai
    original_import = __import__
    def mock_import(name, *args, **kwargs):
        if name == 'crewai':
            raise ImportError("No module named 'crewai'")
        return original_import(name, *args, **kwargs)
            
    with patch('builtins.__import__', side_effect=mock_import):
        with pytest.raises(ImportError, match="Please install crewai"):
            manager.create_model()

def test_create_model_exception(manager):
    mock_crewai = MagicMock()
    mock_crewai.LLM.side_effect = Exception("Error")
    
    # We need to ensure default_config is available if no config passed, or pass a config
    # The test calls create_model() without args, so it uses default_config.
    # The mock in conftest returns {}, which causes KeyError when accessing 'model_id'.
    # But here we want to test exception from LLM init.
    
    with patch.object(CrewAIModelConfigurationManager, 'default_config', new_callable=lambda: {'model_id': 'default', 'params': {'temperature': 0.1, 'max_tokens': 100}}):
        with patch.dict('sys.modules', {'crewai': mock_crewai}):
            # The error message from create_model is "Failed to create CrewAI LLM: Error"
            # So we should match "Failed to create CrewAI LLM" or just "Error" if it re-raises?
            # The code logs error and then raises. So it re-raises the original exception?
            # No, `raise` without arguments re-raises the active exception.
            
            with pytest.raises(Exception, match="Error"):
                manager.create_model()
