"""Additional coverage for ConfigValidator."""
from unittest.mock import MagicMock

from oai_agent_core.services.config_validator import ConfigValidator


def make():
    return ConfigValidator(logger=MagicMock())


def test_empty_config_valid():
    assert make().validate_agent_config({}) is True


def test_missing_type_warning():
    v = make()
    assert v.validate_agent_config({"model": {"name": "x"}}) is True
    assert any("type" in w for w in v.get_warnings())


def test_model_config_errors():
    v = make()
    v._validate_model_config("notdict")
    assert v.errors
    v2 = make()
    v2._validate_model_config({})  # empty -> early return (line 97)
    assert not v2.errors
    v3 = make()
    v3._validate_model_config({"temperature": 0.5})  # real settings, missing name/provider
    assert any("name" in e for e in v3.errors)
    assert any("provider" in e for e in v3.errors)
    v4 = make()
    v4._validate_model_config({"name": 123, "provider": 456})  # wrong types -> warnings
    assert len(v4.warnings) == 2


def test_tools_config():
    v = make()
    v._validate_tools_config("notdict")
    assert v.errors
    v2 = make()
    v2._validate_tools_config({"t1": "notdict", "t2": {}})
    assert any("must be a dictionary" in e for e in v2.errors)
    assert any("missing" in w for w in v2.warnings)


def test_kb_config():
    v = make()
    v._validate_kb_config({"sources": "notlist"})
    assert any("sources" in w for w in v.warnings)
    v2 = make()
    v2._validate_kb_config([{"a": 1}, "notdict"])
    assert any("must be a dictionary" in e for e in v2.errors)
    v3 = make()
    v3._validate_kb_config("badtype")
    assert v3.errors


def test_skills_config():
    v = make()
    v._validate_skills_config("notdict")
    assert v.errors
    v2 = make()
    v2._validate_skills_config({"registry": "notdict"})
    assert any("skill_dir" in e for e in v2.errors)
    assert any("registry" in e for e in v2.errors)


def test_memory_config():
    v = make()
    v._validate_memory_config("notdict")
    assert v.errors
    v2 = make()
    v2._validate_memory_config({"type": "unknownmem"})
    assert any("Unknown memory type" in w for w in v2.warnings)
    v3 = make()
    v3._validate_memory_config({})
    assert any("type" in w for w in v3.warnings)


def test_guardrails_config():
    v = make()
    v._validate_guardrails_config("notdict")
    assert v.errors
    v2 = make()
    v2._validate_guardrails_config({})
    assert any("guard" in w for w in v2.warnings)


def test_full_config_with_all_sections():
    v = make()
    config = {
        "type": "crewai",
        "model": {"name": "m", "provider": "openai", "temperature": 0.5},
        "tools": {"t": {"module": "x"}},
        "knowledge_base": {"sources": []},
        "skills": {"skill_dir": "d"},
        "memory": {"type": "simple"},
        "guardrails": {"guard": "g"},
    }
    assert v.validate_agent_config(config) is True


def test_export_json_schema():
    schema = make().export_json_schema()
    assert schema["title"] == "Agent Configuration Schema"
    assert "model" in schema["properties"]


def test_migrate_config_list_kb():
    v = make()
    old = {
        "knowledge_base": [
            {"registry_url": "http://r", "auth_token": "tok", "name": "kb1"}
        ]
    }
    new = v.migrate_config(old)
    assert isinstance(new["knowledge_base"], dict)
    assert new["knowledge_base"]["registry"]["url"] == "http://r"
    assert new["knowledge_base"]["registry"]["token"] == "tok"
    assert new["type"] == "base"


def test_migrate_config_empty_list_kb():
    v = make()
    new = v.migrate_config({"knowledge_base": [], "type": "x"})
    assert new["knowledge_base"]["sources"] == []


def test_validate_with_details():
    v = make()
    result = v.validate_with_details({
        "model": {"temperature": 0.5},  # missing name/provider -> errors
    })
    assert result["valid"] is False
    assert result["errors"]
    assert any("required fields" in s for s in result["suggestions"])


def test_validate_with_details_type_suggestion():
    v = make()
    result = v.validate_with_details({"model": {"name": "x"}})
    assert any("type" in s.lower() for s in result["suggestions"])
