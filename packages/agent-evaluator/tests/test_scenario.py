import pytest

from oai_agent_evaluator.scenario import TestScenario


def test_scenario_init_valid():
    scenario = TestScenario(
        name="Test 1",
        description="Desc",
        input_message="Input",
        expected_output="Output"
    )
    assert scenario.name == "Test 1"
    assert scenario.metrics == ["correctness"]  # Default


def test_scenario_init_invalid():
    with pytest.raises(ValueError):
        TestScenario(
            name="Test 1",
            description="Desc",
            input_message="Input"
            # Missing expected_output AND evaluation_criteria
        )


def test_scenario_custom_metrics():
    scenario = TestScenario(
        name="Test 1",
        description="Desc",
        input_message="Input",
        expected_output="Output",
        metrics=["safety", "relevance"]
    )
    assert "safety" in scenario.metrics
    assert "relevance" in scenario.metrics


def test_scenario_invalid_pass_threshold():
    with pytest.raises(ValueError, match="pass_threshold must be between 0 and 10"):
        TestScenario(
            name="Test 1",
            description="Desc",
            input_message="Input",
            expected_output="Output",
            pass_threshold=11.0
        )
    with pytest.raises(ValueError, match="pass_threshold must be between 0 and 10"):
        TestScenario(
            name="Test 1",
            description="Desc",
            input_message="Input",
            expected_output="Output",
            pass_threshold=-1.0
        )
