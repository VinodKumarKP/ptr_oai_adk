import pytest
from unittest.mock import MagicMock, patch
from oai_agent_core.processing.message_formatter import MessageFormatter


class TestMessageFormatter:
    
    @pytest.fixture
    def formatter(self):
        return MessageFormatter()
    
    def test_init(self, formatter):
        assert formatter.logger is not None
    
    def test_format_message_string(self, formatter):
        result = formatter.format_message("test message", {})
        assert result == "test message"
    
    def test_format_message_dict_with_content(self, formatter):
        message = "Hello {name}!"
        inputs = {"name": "World"}
        result = formatter.format_message(message, inputs)
        assert result == "Hello World!"
    
    def test_format_message_dict_with_message(self, formatter):
        message = "Process {input} and save to {output}"
        inputs = {"input": "data.txt", "output": "result.txt"}
        result = formatter.format_message(message, inputs)
        assert result == "Process data.txt and save to result.txt"
    
    def test_format_message_dict_fallback(self, formatter):
        message = "No variables here"
        inputs = {"unused": "value"}
        result = formatter.format_message(message, inputs)
        assert result == "No variables here"
    
    def test_format_message_list(self, formatter):
        message = "Hello {name}, you have {count} items"
        inputs = {"name": "Alice", "count": 5}
        result = formatter.format_message(message, inputs)
        assert result == "Hello Alice, you have 5 items"
    
    def test_format_message_none(self, formatter):
        result = formatter.format_message("", {})
        assert result == ""
    
    def test_format_message_exception(self, formatter):
        message = "Hello {name}!"
        inputs = {"name": "World"}
        result = formatter.format_message(message, inputs)
        assert result == "Hello World!"
    
    def test_extract_variables(self, formatter):
        text = "Process {input} and save to {output}"
        variables = formatter.extract_variables(text)
        assert variables == {"input", "output"}
    
    def test_extract_variables_from_config(self, formatter):
        config = {
            "agent_list": [
                {"agent1": {"system_prompt": "Hello {name}"}}
            ],
            "task_list": [
                {"task1": {"description": "Process {data}"}}
            ]
        }
        variables = formatter.extract_variables_from_config(config)
        assert "name" in variables
        assert "data" in variables
    
    def test_get_inputs(self, formatter):
        config = {
            "agent_list": [
                {"agent1": {"system_prompt": "Hello {name}"}}
            ]
        }
        inputs = formatter.get_inputs("World", config)
        assert inputs["name"] == "World"
    
    def test_create_default_inputs(self, formatter):
        variables = {"topic", "style"}
        inputs = formatter.create_default_inputs("Write a blog post", variables)
        assert inputs["topic"] == "Write a blog post"
        assert inputs["style"] == "Write a blog post"
    
    def test_merge_inputs(self, formatter):
        default = {"name": "default", "age": 25}
        override = {"name": "Alice"}
        merged = formatter.merge_inputs(default, override)
        assert merged["name"] == "Alice"
        assert merged["age"] == 25
    
    def test_validate_inputs(self, formatter):
        inputs = {"name": "Alice", "age": 25}
        required = {"name", "age"}
        is_valid, missing = formatter.validate_inputs(inputs, required)
        assert is_valid is True
        assert missing == []
    
    def test_validate_inputs_missing(self, formatter):
        inputs = {"name": "Alice"}
        required = {"name", "age"}
        is_valid, missing = formatter.validate_inputs(inputs, required)
        assert is_valid is False
        assert "age" in missing
    
    def test_format_with_validation(self, formatter):
        message = "Hello {name}!"
        inputs = {"name": "World"}
        result = formatter.format_with_validation(message, inputs)
        assert result == "Hello World!"
    
    def test_format_with_validation_strict(self, formatter):
        message = "Hello {name}!"
        inputs = {}
        with pytest.raises(ValueError):
            formatter.format_with_validation(message, inputs, strict=True)
    
    def test_get_template_info(self, formatter):
        text = "Hello {name}, you have {count} items"
        info = formatter.get_template_info(text)
        assert info["variable_count"] == 2
        assert "name" in info["variables"]
        assert "count" in info["variables"]
        assert info["has_variables"] is True