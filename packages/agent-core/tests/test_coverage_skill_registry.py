"""Additional coverage for SkillRegistry."""
import asyncio
from unittest.mock import MagicMock

import httpx
import pytest

import oai_agent_core.components.skills.skill_registry as srmod
from oai_agent_core.components.skills.skill_registry import SkillRegistry
from oai_agent_core.components.skills.models import SkillProperties


def run(coro):
    return asyncio.run(coro)


class FakeResponse:
    def __init__(self, json_data=None, raise_status=False):
        self._json = json_data
        self._raise = raise_status

    def raise_for_status(self):
        if self._raise:
            raise httpx.HTTPError("bad status")

    def json(self):
        return self._json


class FakeAsyncClient:
    get_response = None
    post_response = None
    get_exc = None

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, headers=None):
        if type(self).get_exc:
            raise type(self).get_exc
        return type(self).get_response

    async def post(self, url, json=None, headers=None):
        if type(self).get_exc:
            raise type(self).get_exc
        return type(self).post_response


def patch_httpx(monkeypatch, get_response=None, post_response=None, get_exc=None):
    FakeAsyncClient.get_response = get_response
    FakeAsyncClient.post_response = post_response
    FakeAsyncClient.get_exc = get_exc
    monkeypatch.setattr(srmod.httpx, "AsyncClient", FakeAsyncClient)


# ---- init / config ----

def test_init_local_only():
    reg = SkillRegistry(logger=MagicMock())
    assert reg.registry_url is None
    assert reg.skills_cache_dir is None


def test_init_hybrid(tmp_path):
    reg = SkillRegistry(
        logger=MagicMock(),
        project_root=str(tmp_path),
        registry_url="http://registry/",
        git_provider=MagicMock(),
        skills_cache_dir="cache",
    )
    assert reg.registry_url == "http://registry"
    assert reg.skills_cache_dir.exists()


def test_init_hybrid_initializes_git_provider(tmp_path):
    # Exercises _initialize_git_provider (GitHubProvider present or ImportError path).
    reg = SkillRegistry(
        logger=MagicMock(),
        project_root=str(tmp_path),
        registry_url="http://registry",
    )
    assert reg.registry_url == "http://registry"


def test_init_git_provider_import_error(monkeypatch, tmp_path):
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("oai_skills_registry"):
            raise ImportError("not installed")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    reg = SkillRegistry(
        logger=MagicMock(),
        project_root=str(tmp_path),
        registry_url="http://registry",
    )
    assert reg.git_provider is None


def test_get_auth_headers():
    reg = SkillRegistry(auth_token="tok")
    assert reg._get_auth_headers()["Authorization"] == "Bearer tok"
    reg2 = SkillRegistry(auth_token="Bearer xyz")
    assert reg2._get_auth_headers()["Authorization"] == "Bearer xyz"
    reg3 = SkillRegistry()
    assert reg3._get_auth_headers() == {}


def test_get_registry_status(tmp_path):
    reg = SkillRegistry(registry_url="http://r", git_provider=MagicMock(), project_root=str(tmp_path))
    status = reg.get_registry_status()
    assert status["mode"] == "hybrid"
    assert status["status"] == "healthy"
    reg2 = SkillRegistry()
    assert reg2.get_registry_status()["mode"] == "local-only"


# ---- discover_skills ----

VALID = "---\nname: my-skill\ndescription: does things\n---\nbody"


def test_discover_skills_no_dir():
    reg = SkillRegistry(logger=MagicMock())
    reg.discover_skills("")
    assert reg.skills == {}


def test_discover_skills_missing_dir(tmp_path):
    reg = SkillRegistry(logger=MagicMock(), project_root=str(tmp_path))
    reg.discover_skills(str(tmp_path / "nope"))
    assert reg.skills == {}


def test_discover_skills_valid_and_invalid(tmp_path):
    skills_dir = tmp_path / "skills"
    good = skills_dir / "good"
    good.mkdir(parents=True)
    (good / "SKILL.md").write_text(VALID)
    bad = skills_dir / "bad"
    bad.mkdir()
    (bad / "SKILL.md").write_text("no frontmatter")
    empty = skills_dir / "empty"
    empty.mkdir()
    # a file (not a dir) should be skipped
    (skills_dir / "afile.txt").write_text("x")
    reg = SkillRegistry(logger=MagicMock())
    reg.discover_skills(str(skills_dir))
    assert "my-skill" in reg.skills


def test_get_skill_and_get_skills(tmp_path):
    reg = SkillRegistry(logger=MagicMock())
    props = SkillProperties(name="s1", description="d", path="p", skill_dir="sd")
    reg.skills["s1"] = props
    assert reg.get_skill("s1") is props
    assert reg.get_skill("missing") is None
    assert reg.get_skills(["s1", "missing"]) == [props]


def test_get_all_skills_local():
    reg = SkillRegistry(logger=MagicMock())
    reg.skills["b"] = SkillProperties(name="b", description="d", path="p", skill_dir="sd")
    reg.skills["a"] = SkillProperties(name="a", description="d", path="p", skill_dir="sd")
    names = [s.name for s in reg.get_all_skills()]
    assert names == ["a", "b"]


def test_get_all_skills_remote_metadata_path():
    reg = SkillRegistry(logger=MagicMock())
    reg.skill_metadata_cache["remote"] = {"name": "remote", "description": "d"}
    # SkillProperties does not accept author/category kwargs -> TypeError path
    with pytest.raises(TypeError):
        reg.get_all_skills()


def test_generate_skills_prompt():
    reg = SkillRegistry(logger=MagicMock())
    props = SkillProperties(name="s1", description="d", path="p", skill_dir="sd")
    out = reg.generate_skills_prompt([props])
    assert "s1" in out


# ---- async API methods ----

def test_initialize_not_hybrid():
    reg = SkillRegistry(logger=MagicMock())
    run(reg.initialize())  # no-op


def test_initialize_hybrid_success(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_response=FakeResponse({"skills": [{"name": "s1"}]}))
    run(reg.initialize())
    assert "s1" in reg.skill_metadata_cache


def test_initialize_hybrid_network_error(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    monkeypatch.setattr(reg, "_load_remote_skill_metadata",
                        MagicMock(side_effect=httpx.HTTPError("net")))
    # _load_remote is async; wrap to raise inside coroutine

    async def raiser():
        raise httpx.HTTPError("net")

    monkeypatch.setattr(reg, "_load_remote_skill_metadata", raiser)
    run(reg.initialize())  # error caught


def test_load_remote_metadata_list(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_response=FakeResponse([{"name": "a"}, {"noname": 1}]))
    run(reg._load_remote_skill_metadata())
    assert "a" in reg.skill_metadata_cache


def test_load_remote_metadata_http_error(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_exc=httpx.HTTPError("boom"))
    run(reg._load_remote_skill_metadata())  # caught, logged


def test_get_skill_metadata_cache_hit():
    reg = SkillRegistry(logger=MagicMock())
    reg.skill_metadata_cache["s1"] = {"name": "s1"}
    assert run(reg.get_skill_metadata("s1")) == {"name": "s1"}


def test_get_skill_metadata_not_hybrid():
    reg = SkillRegistry(logger=MagicMock())
    assert run(reg.get_skill_metadata("s1")) is None


def test_get_skill_metadata_api(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_response=FakeResponse({"name": "s1", "git_repository_url": "u"}))
    md = run(reg.get_skill_metadata("s1"))
    assert md["name"] == "s1"


def test_get_skill_metadata_api_error(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_exc=httpx.HTTPError("x"))
    assert run(reg.get_skill_metadata("s1")) is None


def test_list_versions_not_hybrid():
    reg = SkillRegistry(logger=MagicMock())
    assert run(reg.list_available_versions("s1")) == []


def test_list_versions_success(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_response=FakeResponse({"versions": [{"version": "1.0"}]}))
    assert run(reg.list_available_versions("s1")) == [{"version": "1.0"}]


def test_list_versions_error(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_exc=httpx.HTTPError("x"))
    assert run(reg.list_available_versions("s1")) == []


def test_report_usage_not_hybrid():
    reg = SkillRegistry(logger=MagicMock())
    assert run(reg.report_skill_usage("s1", "agent", True)) is False


def test_report_usage_success(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, post_response=FakeResponse({}))
    assert run(reg.report_skill_usage("s1", "agent", True, duration_ms=10)) is True


def test_report_usage_error(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    patch_httpx(monkeypatch, get_exc=httpx.HTTPError("x"))
    assert run(reg.report_skill_usage("s1", "agent", False)) is False


# ---- pull_skill / fetch ----

def test_pull_skill_not_hybrid():
    reg = SkillRegistry(logger=MagicMock())
    assert run(reg.pull_skill("s1")) is False


def test_pull_skill_no_metadata(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))

    async def no_md(name):
        return None

    monkeypatch.setattr(reg, "get_skill_metadata", no_md)
    assert run(reg.pull_skill("s1")) is False


def test_pull_skill_no_git_url(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))

    async def md(name):
        return {"name": name}

    monkeypatch.setattr(reg, "get_skill_metadata", md)
    assert run(reg.pull_skill("s1")) is False


def test_pull_skill_success(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))

    async def md(name):
        return {"name": name, "git_repository_url": "github.com/o/r", "git_branch": "main"}

    async def fetch(**k):
        return True

    monkeypatch.setattr(reg, "get_skill_metadata", md)
    monkeypatch.setattr(reg, "_fetch_skill_code_from_github", fetch)
    assert run(reg.pull_skill("s1")) is True


def test_pull_skill_with_version(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))

    async def md(name):
        return {"name": name, "git_repository_url": "github.com/o/r"}

    async def versions(name):
        return [{"version": "2.0", "git_tag": "v2.0"}]

    async def fetch(**k):
        return k.get("branch") == "v2.0"

    monkeypatch.setattr(reg, "get_skill_metadata", md)
    monkeypatch.setattr(reg, "list_available_versions", versions)
    monkeypatch.setattr(reg, "_fetch_skill_code_from_github", fetch)
    assert run(reg.pull_skill("s1", version="2.0")) is True


def test_pull_skill_fetch_fails(monkeypatch, tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))

    async def md(name):
        return {"name": name, "git_repository_url": "github.com/o/r"}

    async def fetch(**k):
        return False

    monkeypatch.setattr(reg, "get_skill_metadata", md)
    monkeypatch.setattr(reg, "_fetch_skill_code_from_github", fetch)
    assert run(reg.pull_skill("s1")) is False


def test_fetch_code_no_provider(tmp_path):
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=MagicMock(),
                        project_root=str(tmp_path))
    reg.git_provider = None
    assert run(reg._fetch_skill_code_from_github("github.com/o/r", "s1")) is False


def test_fetch_code_fetch_skill_files(tmp_path):
    gp = MagicMock()

    async def fetch_skill_files(**k):
        return {"SKILL.md": "content", "src/x.py": "code"}

    gp.fetch_skill_files = fetch_skill_files
    # ensure clone_repository absent
    del gp.clone_repository
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=gp,
                        project_root=str(tmp_path))
    ok = run(reg._fetch_skill_code_from_github("https://github.com/o/r.git", "s1"))
    assert ok is True
    assert (reg.skills_cache_dir / "s1" / "SKILL.md").exists()


def test_fetch_code_clone_repository(tmp_path):
    gp = MagicMock(spec=["clone_repository"])

    async def clone(**k):
        return None

    gp.clone_repository = clone
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=gp,
                        project_root=str(tmp_path))
    ok = run(reg._fetch_skill_code_from_github("github.com/o/r", "s1", branch="dev"))
    assert ok is True


def test_fetch_code_unsupported_provider(tmp_path):
    gp = MagicMock(spec=[])
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=gp,
                        project_root=str(tmp_path))
    assert run(reg._fetch_skill_code_from_github("github.com/o/r", "s1")) is False


def test_fetch_code_exception(tmp_path):
    gp = MagicMock()

    async def boom(**k):
        raise RuntimeError("clone fail")

    gp.fetch_skill_files = boom
    del gp.clone_repository
    reg = SkillRegistry(logger=MagicMock(), registry_url="http://r", git_provider=gp,
                        project_root=str(tmp_path))
    assert run(reg._fetch_skill_code_from_github("github.com/o/r", "s1")) is False
