from unittest.mock import AsyncMock
import json

import pytest

from oai_agent_evaluator.evaluator import AgentEvaluator
from oai_agent_evaluator.scenario import TestScenario


@pytest.mark.asyncio
async def test_evaluate_scenario_success(mock_agent_class, mock_judge_response):
    # Setup scenario
    scenario = TestScenario(
        name="Test 1",
        description="Desc",
        input_message="Input",
        expected_output="Output",
        metrics=["correctness", "relevance"]
    )

    # Setup evaluator
    evaluator = AgentEvaluator(
        agent_class=mock_agent_class,
        project_root="/tmp"
    )

    # Mock judge agent behavior
    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = mock_judge_response

    # We need to ensure the evaluator uses our mock judge instance
    # Since _initialize_judge_agent creates a NEW instance of agent_class,
    # we rely on mock_agent_class returning a mock that behaves correctly.
    # However, to control the judge specifically, we can patch the method or 
    # rely on the fact that mock_agent_class returns a mock.

    # Let's customize the mock_agent_class to return different mocks for judge vs subject
    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Actual response'}]}

    mock_judge_instance = AsyncMock()
    mock_judge_instance.initialize = AsyncMock()
    mock_judge_instance.ainvoke.return_value = mock_judge_response

    mock_agent_class.side_effect = [mock_judge_instance, mock_subject]

    # Run evaluation
    result = await evaluator.evaluate_scenario(scenario)

    assert result['passed'] is True
    assert result['score'] == 10.0
    assert 'metrics' in result
    assert result['metrics']['correctness']['score'] == 10

    # Verify calls
    assert mock_agent_class.call_count == 2  # 1 for judge, 1 for subject
    mock_subject.ainvoke.assert_called_once()
    mock_judge_instance.ainvoke.assert_called_once()


@pytest.mark.asyncio
async def test_evaluate_scenario_json_failure(mock_agent_class):
    scenario = TestScenario(
        name="Test 1",
        description="Desc",
        input_message="Input",
        expected_output="Output"
    )

    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")

    # Mock judge returning invalid JSON
    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = {'content': [{'text': 'Invalid JSON'}]}

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Response'}]}

    mock_agent_class.side_effect = [mock_judge, mock_subject]

    result = await evaluator.evaluate_scenario(scenario)

    # Should handle error gracefully (score 0)
    assert result['passed'] is False
    assert result['score'] == 0.0
    assert "Evaluation failed" in result['metrics']['correctness']['explanation']


@pytest.mark.asyncio
async def test_caching(mock_agent_class, mock_judge_response):
    scenario1 = TestScenario(name="S1", description="D", input_message="I", expected_output="O", agent_config={'a': 1})
    scenario2 = TestScenario(name="S2", description="D", input_message="I", expected_output="O", agent_config={'a': 1})

    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")

    # Setup mocks
    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = mock_judge_response

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Response'}]}

    # Sequence: Judge init -> Subject init (S1) -> Subject reuse (S2)
    # So side_effect should be [Judge, Subject]
    mock_agent_class.side_effect = [mock_judge, mock_subject]

    await evaluator.evaluate_scenario(scenario1)
    await evaluator.evaluate_scenario(scenario2)

    # Agent class should be instantiated only twice (1 judge + 1 subject reused)
    assert mock_agent_class.call_count == 2


@pytest.mark.asyncio
async def test_evaluate_scenario_metrics(mock_agent_class):
    scenario = TestScenario(
        name="Test", description="D", input_message="I", expected_output="O",
        metrics=["correctness", "relevance", "safety", "custom"]
    )

    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")

    # Mock judge response with all metrics
    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = {
        'content': [{
            'text': '''{
                "correctness": {"score": 10, "explanation": "Good"},
                "relevance": {"score": 8, "explanation": "Okay"},
                "safety": {"score": 10, "explanation": "Safe"},
                "custom": {"score": 5, "explanation": "Custom"}
            }'''
        }]
    }

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Response'}]}

    mock_agent_class.side_effect = [mock_judge, mock_subject]

    result = await evaluator.evaluate_scenario(scenario)

    assert result['metrics']['correctness']['score'] == 10
    assert result['metrics']['relevance']['score'] == 8
    assert result['metrics']['safety']['score'] == 10
    assert result['metrics']['custom']['score'] == 5

    # Average: (10+8+10+5)/4 = 8.25
    assert result['score'] == 8.25


@pytest.mark.asyncio
async def test_evaluate_scenario_judge_init_once(mock_agent_class):
    scenario = TestScenario(name="T", description="D", input_message="I", expected_output="O")
    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")

    # Mock judge
    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = {'content': [{'text': '{"correctness": {"score": 10}}'}]}

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'R'}]}

    mock_agent_class.side_effect = [mock_judge, mock_subject]

    # Run twice
    await evaluator.evaluate_scenario(scenario)
    await evaluator.evaluate_scenario(scenario)

    # Judge should be initialized only once (first call in side_effect)
    # Subject initialized once (second call in side_effect) due to caching
    assert mock_agent_class.call_count == 2
    mock_judge.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_run_suite_parallel(mock_agent_class, mock_judge_response):
    # Create multiple scenarios
    scenarios = [
        TestScenario(name=f"S{i}", description="D", input_message="I", expected_output="O")
        for i in range(5)
    ]

    # Set concurrency to 2
    evaluator = AgentEvaluator(
        agent_class=mock_agent_class, 
        project_root="/tmp",
        max_concurrency=2
    )

    # Mock judge
    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = mock_judge_response

    # Mock subject
    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Response'}]}

    # Side effect: 1 judge + 1 subject (reused)
    mock_agent_class.side_effect = [mock_judge, mock_subject]

    results = await evaluator.run_suite(scenarios)

    assert len(results) == 5
    # Judge initialized once, Subject initialized once (cached)
    assert mock_agent_class.call_count == 2


@pytest.mark.asyncio
async def test_pass_threshold_custom(mock_agent_class, mock_judge_response):
    """Test custom pass_threshold configuration."""
    scenario = TestScenario(
        name="Test with custom threshold",
        description="Desc",
        input_message="Input",
        expected_output="Output",
        metrics=["correctness"],
        pass_threshold=8.0
    )

    evaluator = AgentEvaluator(
        agent_class=mock_agent_class,
        project_root="/tmp",
        pass_threshold=7.0  # Default threshold
    )

    # Mock judge response with score of 7.5 (passes at 8.0 but fails at default)
    judge_response = {
        'correctness': {'score': 7.5, 'explanation': 'Decent'}
    }

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Actual response'}]}

    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = {
        'content': [{'text': json.dumps(judge_response)}]
    }

    mock_agent_class.side_effect = [mock_judge, mock_subject]

    result = await evaluator.evaluate_scenario(scenario)

    # Should fail because 7.5 < 8.0 (scenario-specific threshold)
    assert result['passed'] is False
    assert result['score'] == 7.5


@pytest.mark.asyncio
async def test_timing_metrics(mock_agent_class, mock_judge_response):
    """Test that timing metrics are captured and included in results."""
    scenario = TestScenario(
        name="Test timing",
        description="Desc",
        input_message="Input",
        expected_output="Output",
        metrics=["correctness"]
    )

    evaluator = AgentEvaluator(
        agent_class=mock_agent_class,
        project_root="/tmp"
    )

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Response'}]}

    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = {
        'content': [{'text': json.dumps(mock_judge_response)}]
    }

    mock_agent_class.side_effect = [mock_judge, mock_subject]

    result = await evaluator.evaluate_scenario(scenario)

    # Verify timing fields are present
    assert 'duration_ms' in result
    assert 'agent_invocation_ms' in result
    assert 'judge_invocation_ms' in result
    assert result['duration_ms'] > 0
    assert result['agent_invocation_ms'] > 0
    assert result['judge_invocation_ms'] > 0
    # Total should be >= sum of parts (may include initialization overhead)
    assert result['duration_ms'] >= result['agent_invocation_ms']


@pytest.mark.asyncio
async def test_model_config_override(mock_agent_class, mock_judge_response):
    """Test that agent_model_config overrides agent_config model settings."""
    scenario = TestScenario(
        name="Test model override",
        description="Desc",
        input_message="Input",
        expected_output="Output",
        metrics=["correctness"],
        agent_config={'model': {'model_id': 'gpt-3.5-turbo', 'temperature': 0.5}},
        agent_model_config={'model_id': 'gpt-4o', 'temperature': 0.7}
    )

    evaluator = AgentEvaluator(
        agent_class=mock_agent_class,
        project_root="/tmp"
    )

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Response'}]}

    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = {
        'content': [{'text': json.dumps(mock_judge_response)}]
    }

    mock_agent_class.side_effect = [mock_judge, mock_subject]

    result = await evaluator.evaluate_scenario(scenario)

    call_kwargs = mock_agent_class.call_args_list[1][1]
    assert call_kwargs['agent_config']['model']['model_id'] == 'gpt-4o'
    assert call_kwargs['agent_config']['model']['temperature'] == 0.7


@pytest.mark.asyncio
async def test_model_config_override_no_model_key(mock_agent_class, mock_judge_response):
    """Test model config override when agent_config lacks a model key, and response string fallback."""
    scenario = TestScenario(
        name="Test no model key",
        description="Desc",
        input_message="Input",
        expected_output="Output",
        metrics=["correctness"],
        agent_config={'other': 123},
        agent_model_config={'model_id': 'gpt-4o'}
    )
    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = "raw-string-response"  # triggers str(response) at line 303

    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = mock_judge_response

    mock_agent_class.side_effect = [mock_judge, mock_subject]
    result = await evaluator.evaluate_scenario(scenario)

    assert result['actual_output'] == "raw-string-response"
    call_kwargs = mock_agent_class.call_args_list[1][1]
    assert call_kwargs['agent_config']['model']['model_id'] == 'gpt-4o'


@pytest.mark.asyncio
async def test_crewai_judge(mock_agent_class):
    """Test CrewAI detection and judge configuration rendering."""
    mock_agent_class.__name__ = "CrewAIAgent"
    mock_agent_class.__module__ = "crewai"
    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")
    # Verify detecting CrewAI via class name
    judge = await evaluator._initialize_judge_agent()
    assert judge is not None
    # Class was called with CrewAI specific judge config dict
    config = mock_agent_class.call_args[1].get('agent_config')
    assert config is not None
    # It should have mapped 'type': 'crewai' or similar
    assert config.get('type') == 'crewai'


@pytest.mark.asyncio
async def test_scenario_override_judge_model_id(mock_agent_class):
    """Test scenario specific judge model ID override."""
    scenario = TestScenario(
        name="Test",
        description="Desc",
        input_message="I",
        expected_output="O",
        judge_model_id="scenario-override-judge"
    )
    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp", judge_model_id="default-judge")
    
    mock_judge = AsyncMock()
    mock_agent_class.return_value = mock_judge

    judge = await evaluator._initialize_judge_agent(scenario)
    assert mock_agent_class.call_count == 1
    call_kwargs = mock_agent_class.call_args[1]
    assert call_kwargs['agent_config']['model']['model_id'] == "scenario-override-judge"


@pytest.mark.asyncio
async def test_evaluate_scenario_general_exception(mock_agent_class):
    """Test exception handler in evaluate_scenario."""
    scenario = TestScenario(
        name="Test Exception",
        description="Desc",
        input_message="I",
        expected_output="O"
    )
    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")
    
    # Force initialize judge to fail
    mock_agent_class.side_effect = Exception("Init failure")
    
    result = await evaluator.evaluate_scenario(scenario)
    assert result['passed'] is False
    assert "Init failure" in result['error']


@pytest.mark.asyncio
async def test_judge_response_parsing_variants(mock_agent_class):
    """Test various judge response object shapes and fallback string parsing."""
    scenario = TestScenario(
        name="Test Variants",
        description="Desc",
        input_message="I",
        expected_output="O",
        evaluation_criteria="Criteria X"
    )
    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")

    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Resp'}]}

    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    
    # 1. Content is a dict instead of a list
    mock_judge.ainvoke.return_value = {
        'content': {'text': '{"correctness": {"score": 8, "explanation": "Ok"}}'}
    }
    mock_agent_class.side_effect = [mock_judge, mock_subject]
    result = await evaluator.evaluate_scenario(scenario)
    assert result['score'] == 8.0

    # 2. Content is not dict or list (fall back to str(response))
    mock_judge.ainvoke.return_value = '{"correctness": {"score": 5, "explanation": "Mid"}}'
    mock_agent_class.side_effect = [mock_judge, mock_subject]
    result = await evaluator.evaluate_scenario(scenario)
    assert result['score'] == 5.0

    # 3. Dict subject response, token_usage, and None judge fallback
    evaluator._agent_cache.clear()
    mock_subject_dict = AsyncMock()
    mock_subject_dict.initialize = AsyncMock()
    mock_subject_dict.ainvoke.return_value = {
        'content': {'text': 'Subject text dict'},
        'token_usage': {'prompt_tokens': 100}
    }
    mock_agent_class.side_effect = [mock_subject_dict]
    
    evaluator._initialize_judge_agent = AsyncMock(return_value=None)
    evaluator.judge_agent = mock_judge
    
    mock_judge.ainvoke.return_value = {
        'content': [{'text': '{"correctness": {"score": 9, "explanation": "Ok"}}'}]
    }
    
    result = await evaluator.evaluate_scenario(scenario)
    assert result['actual_output'] == 'Subject text dict'
    assert result['token_usage'] == {'prompt_tokens': 100}
    assert result['score'] == 9.0


@pytest.mark.asyncio
async def test_model_id_extraction_fallbacks(mock_agent_class, mock_judge_response):
    """Test fallback paths for model_id extraction."""
    evaluator = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")
    
    # 1. Extract from scenario.agent_config model settings
    scenario = TestScenario(
        name="Test Extract",
        description="Desc",
        input_message="I",
        expected_output="O",
        agent_config={'model': {'model_id': 'extracted-config-id'}}
    )
    mock_subject = AsyncMock()
    mock_subject.initialize = AsyncMock()
    mock_subject.ainvoke.return_value = {'content': [{'text': 'Resp'}]}
    
    mock_judge = AsyncMock()
    mock_judge.initialize = AsyncMock()
    mock_judge.ainvoke.return_value = mock_judge_response
    mock_agent_class.side_effect = [mock_judge, mock_subject]
    
    res = await evaluator.evaluate_scenario(scenario)
    assert res['model_id'] == 'extracted-config-id'

    # 2. Extract from response directly
    evaluator2 = AgentEvaluator(agent_class=mock_agent_class, project_root="/tmp")
    scenario2 = TestScenario(name="Test Extract 2", description="Desc", input_message="I", expected_output="O")
    mock_subject2 = AsyncMock()
    mock_subject2.initialize = AsyncMock()
    mock_subject2.ainvoke.return_value = {
        'content': [{'text': 'Resp'}],
        'model': {'model_id': 'response-model-id'}
    }
    mock_agent_class.side_effect = [mock_judge, mock_subject2]
    res2 = await evaluator2.evaluate_scenario(scenario2)
    assert res2['model_id'] == 'response-model-id'

    # 3. Extract from agent.llm.model
    class FakeLLM:
        model = "agent-llm-model-id"
    class FakeAgentWithLLM:
        def __init__(self, *args, **kwargs):
            self.llm = FakeLLM()
        async def initialize(self):
            pass
        async def ainvoke(self, *args, **kwargs):
            return {'content': [{'text': 'Resp'}], 'model': {'model_id': None}}
            
    evaluator_llm = AgentEvaluator(agent_class=FakeAgentWithLLM, project_root="/tmp")
    evaluator_llm.judge_agent = mock_judge
    res3 = await evaluator_llm.evaluate_scenario(scenario2)
    assert res3['model_id'] == 'agent-llm-model-id'

