"""Additional coverage for OutputModelRegistry."""
from unittest.mock import MagicMock

from oai_agent_core.components.output_parser.output_model_registry import OutputModelRegistry


def test_discover_no_directory_provided():
    reg = OutputModelRegistry(logger=MagicMock())
    reg.discover_output_models("")
    assert reg.get_all_models() == {}


def test_discover_missing_directory(tmp_path):
    reg = OutputModelRegistry(logger=MagicMock(), project_root=str(tmp_path))
    reg.discover_output_models(str(tmp_path / "does_not_exist"))
    assert reg.get_all_models() == {}


def test_discover_finds_models_and_skips_init(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "__init__.py").write_text("")
    (model_dir / "my_models.py").write_text(
        "from pydantic import BaseModel\n"
        "class Person(BaseModel):\n"
        "    name: str\n"
    )
    reg = OutputModelRegistry(logger=MagicMock())
    reg.discover_output_models(str(model_dir))
    assert "Person" in reg.get_all_models()
    assert reg.get_model("Person") is not None
    assert reg.get_model("Missing") is None


def test_discover_handles_bad_module(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "broken.py").write_text("this is not valid python !!!\n")
    logger = MagicMock()
    reg = OutputModelRegistry(logger=logger)
    reg.discover_output_models(str(model_dir))
    assert logger.error.called


def test_get_model_as_str_and_system_prompt(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "schemas.py").write_text(
        "from pydantic import BaseModel\n"
        "class Answer(BaseModel):\n"
        "    text: str\n"
    )
    reg = OutputModelRegistry(logger=MagicMock())
    reg.discover_output_models(str(model_dir))
    as_str = reg.get_model_as_str("Answer")
    assert as_str is not None and "properties" in as_str
    assert reg.get_model_as_str("Missing") is None
    prompt = reg.get_system_prompt("Answer")
    assert "JSON Schema" in prompt
    assert reg.get_system_prompt("Missing") is None
