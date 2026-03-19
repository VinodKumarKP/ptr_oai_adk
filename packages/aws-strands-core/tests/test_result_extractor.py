import pytest
from unittest.mock import MagicMock
from oai_agent_core.aws_strands_core.processing.result_extractor import ResultExtractor, GraphResult, SwarmResult, AgentResult

@pytest.fixture
def extractor():
    return ResultExtractor()

def test_extract_text_graph_result(extractor):
    mock_result = MagicMock(spec=GraphResult)
    mock_execution = MagicMock()
    mock_execution.result.result.message.content = "graph content"
    mock_execution.result.result.structured_output = None
    mock_result.execution_order = [mock_execution]
    
    text = extractor.extract_text(mock_result)
    assert text == "graph content"

def test_extract_text_graph_result_fallback(extractor):
    mock_result = MagicMock(spec=GraphResult)
    mock_execution = MagicMock()
    
    # Configure the mock to return None for nested paths, forcing fallback
    # The code tries:
    # 1. result.result.message -> None
    # 2. result.message -> None
    # 3. message -> None
    # 4. result -> "direct result"
    
    # We can achieve this by configuring the mock attributes
    # Note: MagicMock creates attributes on access, so we need to be careful.
    
    # Let's use side_effect on _safe_get to control exactly what is returned for each path
    # This avoids fighting with MagicMock structure
    
    original_safe_get = extractor._safe_get
    
    def side_effect(obj, path, default=None):
        if path == 'result.result.structured_output':
            return None
        if path == 'result':
            return "direct result"
        if path in ['result.result.message', 'result.message', 'message']:
            return None
        return original_safe_get(obj, path, default)
        
    extractor._safe_get = side_effect
    
    mock_result.execution_order = [mock_execution]
    
    text = extractor.extract_text(mock_result)
    assert text == "direct result"

def test_extract_text_swarm_result(extractor):
    mock_result = MagicMock(spec=SwarmResult)
    mock_node = MagicMock()
    mock_node.executor.messages = [{'content': [{'text': 'swarm content'}]}]
    mock_result.node_history = [mock_node]
    
    text = extractor.extract_text(mock_result)
    assert text == "swarm content"

def test_extract_text_agent_result(extractor):
    mock_result = MagicMock(spec=AgentResult)
    # Ensure message attribute exists and is a mock
    mock_result.message = MagicMock()
    mock_result.message.content = "agent content"
    
    text = extractor.extract_text(mock_result)
    assert text == "agent content"

def test_extract_text_dict(extractor):
    result = {'content': 'dict content'}
    text = extractor.extract_text(result)
    assert text == "dict content"

def test_extract_text_dict_output(extractor):
    result = {'output': 'output content'}
    text = extractor.extract_text(result)
    assert text == "output content"

def test_extract_text_dict_nested_result(extractor):
    result = {'result': {'content': 'nested content'}}
    text = extractor.extract_text(result)
    assert text == "nested content"

def test_extract_text_exception(extractor):
    # Force an exception during extraction
    class BadObj:
        @property
        def message(self):
            raise Exception("Access Error")
            
    result = BadObj()
    text = extractor.extract_text(result)
    assert "BadObj" in text

def test_safe_get(extractor):
    obj = {'a': {'b': 'c'}}
    assert extractor._safe_get(obj, 'a.b') == 'c'
    assert extractor._safe_get(obj, 'a.x') is None
    assert extractor._safe_get(obj, 'a.x', 'default') == 'default'
    
    obj2 = MagicMock()
    obj2.a.b = 'c'
    assert extractor._safe_get(obj2, 'a.b') == 'c'

def test_extract_from_content_list_mixed(extractor):
    content = [
        {'text': 'part1'},
        MagicMock(text='part2'),
        'part3',
        123 # Should be ignored
    ]
    text = extractor._extract_from_content(content)
    assert text == "part1part2part3"

def test_extract_token_usage(extractor):
    # Test accumulated_usage
    r1 = MagicMock()
    r1.accumulated_usage = {'tokens': 10}
    assert extractor.extract_token_usage(r1) == {'tokens': 10}
    
    # Test metrics.accumulated_usage
    r2 = MagicMock()
    del r2.accumulated_usage # Ensure it doesn't have it
    r2.metrics.accumulated_usage = {'tokens': 20}
    assert extractor.extract_token_usage(r2) == {'tokens': 20}
    
    # Test usage
    r3 = MagicMock()
    del r3.accumulated_usage
    del r3.metrics
    r3.usage = {'tokens': 30}
    assert extractor.extract_token_usage(r3) == {'tokens': 30}
    
    # Test message.usage
    r4 = MagicMock()
    del r4.accumulated_usage
    del r4.metrics
    del r4.usage
    r4.message.usage = {'tokens': 40}
    assert extractor.extract_token_usage(r4) == {'tokens': 40}

def test_format_response_raw(extractor):
    result = "simple result"
    response = extractor.format_response(result, "sess", "model", include_raw=True)
    assert response['raw_result'] == "simple result"

def test_serialize_result(extractor):
    obj = {'a': [1, 2], 'b': 'c'}
    serialized = extractor._serialize_result(obj)
    assert serialized == obj
    
    class CustomObj:
        def __init__(self):
            self.x = 1
            self._y = 2
            
    serialized_custom = extractor._serialize_result(CustomObj())
    assert serialized_custom == {'x': 1}

def test_extract_execution_metadata_graph(extractor):
    mock_result = MagicMock(spec=GraphResult)
    step1 = MagicMock()
    step1.node_id = "node1"
    mock_result.execution_order = [step1]
    
    meta = extractor.extract_execution_metadata(mock_result)
    assert meta['execution_steps'] == 1
    assert meta['execution_path'] == ['node1']

def test_extract_execution_metadata_swarm(extractor):
    mock_result = MagicMock(spec=SwarmResult)
    node1 = MagicMock()
    node1.node_id = "agent1"
    mock_result.node_history = [node1]
    mock_result.results = ['res1']
    
    meta = extractor.extract_execution_metadata(mock_result)
    assert meta['node_count'] == 1
    assert meta['agents_involved'] == ['agent1']
    assert meta['result_count'] == 1

def test_format_streaming_chunk(extractor):
    chunk = extractor.format_streaming_chunk("content", agent="agent1", extra="val")
    assert chunk['content'] == "content"
    assert chunk['agent'] == "agent1"
    assert chunk['extra'] == "val"
    assert chunk['final'] is False

def test_extract_results_list(extractor):
    # Graph
    g_res = MagicMock(spec=GraphResult)
    step = MagicMock()
    step.result = "res1"
    g_res.execution_order = [step]
    assert extractor.extract_results_list(g_res) == ["res1"]
    
    # Swarm
    s_res = MagicMock(spec=SwarmResult)
    s_res.results = ["res2"]
    assert extractor.extract_results_list(s_res) == ["res2"]

def test_repr(extractor):
    assert repr(extractor) == "ResultExtractor()"
