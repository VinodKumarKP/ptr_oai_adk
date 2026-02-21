import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.utils.prompt_analyzer import PromptAnalyzer

@pytest.fixture
def mock_llm():
    return MagicMock()

@pytest.fixture
def analyzer(mock_llm):
    return PromptAnalyzer(llm=mock_llm)

def test_init(analyzer, mock_llm):
    assert analyzer.llm == mock_llm
    assert analyzer.logger is not None

def test_analyze_no_llm():
    analyzer = PromptAnalyzer(llm=None)
    result = analyzer.analyze("query")
    assert result == ["query"]

def test_analyze_langchain_llm(analyzer, mock_llm):
    # Mock LangChain style invoke
    mock_response = MagicMock()
    mock_response.content = "q1, q2"
    mock_llm.invoke.return_value = mock_response
    
    result = analyzer.analyze("query")
    
    assert result == ["q1", "q2"]
    mock_llm.invoke.assert_called_once()

def test_analyze_litellm_fallback(analyzer, mock_llm):
    # Remove invoke method to trigger fallback
    del mock_llm.invoke
    
    # Mock config for model_id
    mock_llm.config = {'model_id': 'gpt-4'}
    
    with patch('litellm.completion') as mock_completion:
        mock_completion.return_value = {
            'choices': [{'message': {'content': 'q1, q2'}}]
        }
        
        result = analyzer.analyze("query")
        
        assert result == ["q1", "q2"]
        mock_completion.assert_called_once()

def test_analyze_error(analyzer, mock_llm):
    mock_llm.invoke.side_effect = Exception("Error")
    
    result = analyzer.analyze("query")
    
    assert result == ["query"]
