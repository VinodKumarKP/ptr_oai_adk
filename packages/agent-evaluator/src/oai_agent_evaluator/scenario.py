from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

@dataclass
class TestScenario:
    """Defines a test scenario for agent regression testing."""
    name: str
    description: str
    input_message: str
    expected_output: Optional[str] = None
    evaluation_criteria: Optional[str] = None
    agent_config: Dict[str, Any] = field(default_factory=dict)
    config_overrides: Optional[Dict[str, Any]] = None
    agent_name: str = "UnknownAgent"
    metrics: List[str] = field(default_factory=lambda: ["correctness"])
    agent_class: Optional[str] = None
    agent_model_config: Optional[Dict[str, Any]] = None
    judge_model_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.expected_output and not self.evaluation_criteria:
            raise ValueError("Either expected_output or evaluation_criteria must be provided.")
