"""Additional coverage for skills parser, models, errors, and prompt."""
import pytest

from oai_agent_core.components.skills import parser
from oai_agent_core.components.skills.errors import ParseError, ValidationError
from oai_agent_core.components.skills.models import SkillProperties
from oai_agent_core.components.skills.prompt import (
    generate_skills_prompt,
    generate_skill_instructions_prompt,
)


VALID_SKILL = """---
name: web-research
description: Research the web
license: MIT
allowed-tools: search
metadata:
  category: research
  level: 1
---
# Body
Do the research.
"""


def _write_skill(tmp_path, content, name="SKILL.md"):
    (tmp_path / name).write_text(content)
    return tmp_path


def test_find_skill_md_uppercase_and_lowercase(tmp_path):
    assert parser.find_skill_md(tmp_path) is None
    (tmp_path / "SKILL.md").write_text("x")
    found = parser.find_skill_md(tmp_path)
    assert found is not None
    assert found.name.lower() == "skill.md"


def test_parse_invalid_frontmatter():
    with pytest.raises(ParseError):
        parser._parse_skill_md("no frontmatter here")


def test_parse_non_mapping_frontmatter():
    # A bare scalar parses but is not a mapping -> ParseError
    with pytest.raises(ParseError):
        parser._parse_skill_md("---\nhello\n---\nbody")


def test_load_metadata_success(tmp_path):
    _write_skill(tmp_path, VALID_SKILL)
    props = parser.load_metadata(tmp_path)
    assert props.name == "web-research"
    assert props.description == "Research the web"
    assert props.metadata["category"] == "research"
    assert props.metadata["level"] == "1"


def test_load_metadata_missing_file(tmp_path):
    with pytest.raises(ParseError):
        parser.load_metadata(tmp_path)


def test_load_metadata_missing_name(tmp_path):
    _write_skill(tmp_path, "---\ndescription: d\n---\nbody")
    with pytest.raises(ValidationError):
        parser.load_metadata(tmp_path)


def test_load_metadata_missing_description(tmp_path):
    _write_skill(tmp_path, "---\nname: n\n---\nbody")
    with pytest.raises(ValidationError):
        parser.load_metadata(tmp_path)


def test_load_metadata_empty_name(tmp_path):
    _write_skill(tmp_path, "---\nname: ' '\ndescription: d\n---\nbody")
    with pytest.raises(ValidationError):
        parser.load_metadata(tmp_path)


def test_load_instructions(tmp_path):
    p = _write_skill(tmp_path, VALID_SKILL)
    body = parser.load_instructions(p / "SKILL.md")
    assert "Do the research." in body


def test_load_instructions_failure(tmp_path):
    with pytest.raises(ParseError):
        parser.load_instructions(tmp_path / "nope.md")


def test_load_resource_success(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "helper.py").write_text("print('hi')")
    content = parser.load_resource(tmp_path, "scripts/helper.py")
    assert "print" in content


def test_load_resource_outside_dir(tmp_path):
    with pytest.raises(ParseError):
        parser.load_resource(tmp_path, "../escape.txt")


def test_load_resource_missing(tmp_path):
    with pytest.raises(ParseError):
        parser.load_resource(tmp_path, "missing.txt")


def test_load_resource_not_a_file(tmp_path):
    (tmp_path / "adir").mkdir()
    with pytest.raises(ParseError):
        parser.load_resource(tmp_path, "adir")


def test_load_resource_too_large(tmp_path, monkeypatch):
    big = tmp_path / "big.txt"
    big.write_text("x")

    import stat as stat_module

    class FakeStat:
        st_size = 20 * 1024 * 1024
        st_mode = stat_module.S_IFREG | 0o644

    real_stat = type(big).stat

    def fake_stat(self, *args, **kwargs):
        if self.name == "big.txt":
            return FakeStat()
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr("pathlib.Path.stat", fake_stat)
    with pytest.raises(ParseError):
        parser.load_resource(tmp_path, "big.txt")


def test_skill_properties_to_dict_minimal():
    props = SkillProperties(name="n", description="d", path="p", skill_dir="sd")
    d = props.to_dict()
    assert d == {"name": "n", "description": "d", "path": "p", "skill_dir": "sd"}


def test_skill_properties_to_dict_full():
    props = SkillProperties(
        name="n", description="d", path="p", skill_dir="sd",
        license="MIT", compatibility="x", allowed_tools="search",
        metadata={"k": "v"},
    )
    d = props.to_dict()
    assert d["license"] == "MIT"
    assert d["compatibility"] == "x"
    assert d["allowed-tools"] == "search"
    assert d["metadata"] == {"k": "v"}


def test_validation_error_carries_errors():
    err = ValidationError("oops")
    assert err.errors == ["oops"]
    err2 = ValidationError("oops", ["a", "b"])
    assert err2.errors == ["a", "b"]


def test_generate_skills_prompt_empty():
    assert generate_skills_prompt([]) == ""


def test_generate_skills_prompt_sorts():
    skills = [
        SkillProperties("zeta", "dz", "pz", "sz"),
        SkillProperties("alpha", "da", "pa", "sa"),
    ]
    out = generate_skills_prompt(skills)
    assert out.index("alpha") < out.index("zeta")


def test_generate_skill_instructions_prompt():
    out = generate_skill_instructions_prompt("research", "step 1")
    assert "research" in out
    assert "step 1" in out
