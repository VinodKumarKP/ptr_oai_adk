import yaml
import os
from typing import List, Dict, Any, Union
from pathlib import Path
from .scenario import TestScenario

class ScenarioLoader:
    """Loads test scenarios from YAML files."""
    
    @staticmethod
    def load_from_file(file_path: str) -> List[TestScenario]:
        """Loads scenarios from a single YAML file."""
        with open(file_path, 'r') as f:
            # Load all documents if multiple are present (separated by ---)
            documents = list(yaml.safe_load_all(f))
            
        scenarios = []
        base_path = os.path.dirname(os.path.abspath(file_path))
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        for doc in documents:
            if not doc:
                continue
                
            # Check for global agent_config at the document level
            global_agent_config = None
            if isinstance(doc, dict) and 'agent_config' in doc:
                global_agent_config = doc['agent_config']
            
            # Check for global agent_name
            global_agent_name = None
            if isinstance(doc, dict) and 'agent_name' in doc:
                global_agent_name = doc['agent_name']

            # Check for global metrics
            global_metrics = None
            if isinstance(doc, dict) and 'metrics' in doc:
                global_metrics = doc['metrics']
                
            # Check for global agent_class
            global_agent_class = None
            if isinstance(doc, dict) and 'agent_class' in doc:
                global_agent_class = doc['agent_class']

            # Check for global agent_model_config
            global_agent_model_config = None
            if isinstance(doc, dict) and 'agent_model_config' in doc:
                global_agent_model_config = doc['agent_model_config']
            
            # Check for global judge_model_id
            global_judge_model_id = None
            if isinstance(doc, dict) and 'judge_model_id' in doc:
                global_judge_model_id = doc['judge_model_id']
                
            items = []
            # Check if the document has a 'scenarios' key
            if isinstance(doc, dict) and 'scenarios' in doc:
                content = doc['scenarios']
                if isinstance(content, list):
                    items = content
                elif isinstance(content, dict):
                    # Handle case where 'scenarios' points to a single dict instead of a list
                    items = [content]
            elif isinstance(doc, list):
                # The document itself is a list of scenarios
                items = doc
            elif isinstance(doc, dict) and 'scenarios' not in doc:
                # Single scenario at root (if it has name/input_message)
                if 'name' in doc and 'input_message' in doc:
                    items = [doc]
                
            for item in items:
                if not isinstance(item, dict):
                    continue

                # Determine agent config: item specific > global doc level > empty
                agent_config = item.get('agent_config', global_agent_config)
                if agent_config is None:
                    agent_config = {}
                    
                # Handle external agent config file reference
                if isinstance(agent_config, str):
                    # Resolve relative path
                    config_path = os.path.join(base_path, agent_config)
                    if os.path.exists(config_path):
                        with open(config_path, 'r') as cf:
                            agent_config = yaml.safe_load(cf)
                    else:
                        raise FileNotFoundError(f"Agent config file not found: {config_path}")
                
                # Determine agent name
                agent_name = item.get('agent_name', global_agent_name)
                if not agent_name:
                    agent_name = file_name

                # Determine metrics: item specific > global doc level > default
                metrics = item.get('metrics', global_metrics)
                if metrics is None:
                    metrics = ["correctness", "relevance", "safety"]
                    
                # Determine agent class: item specific > global doc level
                agent_class = item.get('agent_class', global_agent_class)

                # Determine agent model config: item specific > global doc level
                # If item specific is None, use global. If global is None, it remains None.
                # If both are present, item specific takes precedence.
                # BUT, if global is a list and item is None, we want to use the global list for matrix testing.
                agent_model_config = item.get('agent_model_config')
                if agent_model_config is None:
                    agent_model_config = global_agent_model_config
                
                # Determine judge model id: item specific > global doc level
                judge_model_id = item.get('judge_model_id', global_judge_model_id)

                # Handle list of model configs for matrix testing
                model_configs = []
                if isinstance(agent_model_config, list):
                    model_configs = agent_model_config
                else:
                    model_configs = [agent_model_config]

                for model_cfg in model_configs:
                    # Use original scenario name without appending model ID
                    scenario_name = item['name']

                    scenarios.append(TestScenario(
                        name=scenario_name,
                        description=item.get('description', ''),
                        input_message=item['input_message'],
                        expected_output=item.get('expected_output'),
                        evaluation_criteria=item.get('evaluation_criteria'),
                        agent_config=agent_config,
                        config_overrides=item.get('config_overrides'),
                        agent_name=agent_name,
                        metrics=metrics,
                        agent_class=agent_class,
                        agent_model_config=model_cfg,
                        judge_model_id=judge_model_id
                    ))
        return scenarios

    @staticmethod
    def load_from_directory(directory_path: str) -> List[TestScenario]:
        """Loads scenarios from all YAML files in a directory."""
        scenarios = []
        path = Path(directory_path)
        for file_path in path.glob('*.yaml'):
            scenarios.extend(ScenarioLoader.load_from_file(str(file_path)))
        return scenarios
