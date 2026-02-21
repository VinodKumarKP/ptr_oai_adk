import os

from oai_agent_evaluator.reporter import HtmlReporter


def test_generate_report(tmp_path):
    output_dir = tmp_path / "reports"
    reporter = HtmlReporter(output_dir=str(output_dir))

    results = [
        {
            'scenario': 'S1',
            'agent_name': 'AgentA',
            'passed': True,
            'score': 10,
            'input': 'Input',
            'actual_output': 'Output',
            'expected_output': 'Output',
            'explanation': 'Good',
            'metrics': {'correctness': {'score': 10}}
        },
        {
            'scenario': 'S2',
            'agent_name': 'AgentA',
            'passed': False,
            'score': 0,
            'input': 'Input',
            'actual_output': 'Bad',
            'expected_output': 'Good',
            'explanation': 'Bad',
            'error': 'Some error'
        }
    ]

    report_path = reporter.generate_report(results, "TestAgent")

    assert os.path.exists(report_path)
    assert "TestAgent" in report_path

    with open(report_path, 'r') as f:
        content = f.read()
        assert "Agent Regression Report - TestAgent" in content
        assert "S1" in content
        assert "S2" in content
        assert "PASSED" in content
        assert "FAILED" in content
        assert "Some error" in content
