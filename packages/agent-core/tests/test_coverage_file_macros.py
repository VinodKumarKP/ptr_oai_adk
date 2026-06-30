"""Additional coverage for file_macros."""
import base64
import json
from types import SimpleNamespace

import pytest

from oai_agent_core.macros import file_macros as fm


@pytest.fixture
def proc(tmp_path):
    return SimpleNamespace(project_root=str(tmp_path))


def test_macro_open_no_args():
    assert fm.macro_open().startswith("Error: OPEN macro requires")


def test_macro_open_not_found(proc):
    assert "File not found" in fm.macro_open("missing.txt", processor=proc)


def test_macro_open_plain_text(tmp_path, proc):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    assert fm.macro_open("a.txt", processor=proc) == "hello world"


def test_macro_open_explicit_type_unknown(tmp_path, proc):
    f = tmp_path / "data.bin"
    f.write_text("raw")
    # explicit unknown type falls through to plain read
    assert fm.macro_open("data.bin", "bin", processor=proc) == "raw"


def test_macro_open_absolute_path(tmp_path):
    f = tmp_path / "abs.txt"
    f.write_text("abs")
    assert fm.macro_open(str(f)) == "abs"


def test_macro_open_pdf_dispatch(tmp_path, proc, monkeypatch):
    f = tmp_path / "x.pdf"
    f.write_text("notrealpdf")
    monkeypatch.setattr(fm, "_read_pdf", lambda p: "PDFTEXT")
    assert fm.macro_open("x.pdf", processor=proc) == "PDFTEXT"


def test_macro_open_docx_dispatch(tmp_path, proc, monkeypatch):
    f = tmp_path / "x.docx"
    f.write_text("d")
    monkeypatch.setattr(fm, "_read_docx", lambda p: "DOCX")
    assert fm.macro_open("x.docx", processor=proc) == "DOCX"


def test_macro_open_ppt_excel_image(tmp_path, proc, monkeypatch):
    for ext, target, ret in [
        ("pptx", "_read_ppt", "PPT"),
        ("xlsx", "_read_excel", "XLS"),
        ("png", "_read_image_ocr", "IMG"),
    ]:
        f = tmp_path / f"f.{ext}"
        f.write_text("x")
        monkeypatch.setattr(fm, target, lambda p, r=ret: r)
        assert fm.macro_open(f"f.{ext}", processor=proc) == ret


def test_macro_open_read_error(tmp_path, proc, monkeypatch):
    f = tmp_path / "x.txt"
    f.write_text("x")

    def boom(*a, **k):
        raise OSError("io")

    monkeypatch.setattr("builtins.open", boom)
    assert "Error reading file" in fm.macro_open("x.txt", processor=proc)


def test_read_pdf_import_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pypdf", None)
    assert "pypdf package is required" in fm._read_pdf("x.pdf")


def test_read_pdf_success(tmp_path, monkeypatch):
    fake = __import__("types").ModuleType("pypdf")

    class Page:
        def extract_text(self):
            return "page text"

    class Reader:
        def __init__(self, f):
            self.pages = [Page()]

    fake.PdfReader = Reader
    monkeypatch.setitem(__import__("sys").modules, "pypdf", fake)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"x")
    assert "page text" in fm._read_pdf(str(f))


def test_read_pdf_error(tmp_path, monkeypatch):
    fake = __import__("types").ModuleType("pypdf")

    class Reader:
        def __init__(self, f):
            raise RuntimeError("bad pdf")

    fake.PdfReader = Reader
    monkeypatch.setitem(__import__("sys").modules, "pypdf", fake)
    f = tmp_path / "x.pdf"
    f.write_bytes(b"x")
    assert "Error reading PDF" in fm._read_pdf(str(f))


def test_read_excel_import_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pandas", None)
    assert "pandas" in fm._read_excel("x.xlsx")


def test_read_image_ocr_import_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "pytesseract", None)
    assert "pytesseract" in fm._read_image_ocr("x.png")


def test_macro_image(tmp_path, proc):
    f = tmp_path / "img.bin"
    f.write_bytes(b"abc")
    encoded = fm.macro_image("img.bin", processor=proc)
    assert base64.b64decode(encoded) == b"abc"


def test_macro_image_errors(proc):
    assert fm.macro_image().startswith("Error: IMAGE macro requires")
    assert "File not found" in fm.macro_image("nope.bin", processor=proc)


def test_macro_image_read_error(tmp_path, proc, monkeypatch):
    f = tmp_path / "img.bin"
    f.write_bytes(b"abc")
    monkeypatch.setattr("builtins.open", lambda *a, **k: (_ for _ in ()).throw(OSError("x")))
    assert "Error reading image file" in fm.macro_image("img.bin", processor=proc)


def test_macro_path(proc, tmp_path):
    assert fm.macro_path().startswith("Error: PATH macro requires")
    assert fm.macro_path("/abs/path") == "/abs/path"
    assert fm.macro_path("rel", processor=proc) == f"{tmp_path}/rel"
    assert fm.macro_path("rel", "/custom") == "/custom/rel"
    assert "Project root not set" in fm.macro_path("rel")


def test_macro_list_files(tmp_path, proc):
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "b.log").write_text("x")
    assert fm.macro_list_files().startswith("Error: LIST_FILES macro requires")
    assert "Directory not found" in fm.macro_list_files("nodir", processor=proc)
    all_files = json.loads(fm.macro_list_files(str(tmp_path)))
    assert "a.txt" in all_files and "b.log" in all_files
    filtered = json.loads(fm.macro_list_files(str(tmp_path), "*.txt"))
    assert filtered == ["a.txt"]


def test_macro_list_files_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fm.os, "listdir", lambda p: (_ for _ in ()).throw(OSError("x")))
    assert "Error listing files" in fm.macro_list_files(str(tmp_path))


def test_macro_concat(tmp_path, proc):
    (tmp_path / "a.txt").write_text("AAA")
    (tmp_path / "b.txt").write_text("BBB")
    assert fm.macro_concat().startswith("Error: CONCAT macro requires")
    out = fm.macro_concat("a.txt", "b.txt", "missing.txt", processor=proc)
    assert "AAA" in out and "BBB" in out and "File not found" in out


def test_macro_sample(tmp_path, proc):
    f = tmp_path / "lines.txt"
    f.write_text("l1\nl2\nl3\nl4\n")
    assert fm.macro_sample("x").startswith("Error: SAMPLE macro requires")
    assert "must be an integer" in fm.macro_sample("x", "abc")
    assert "File not found" in fm.macro_sample("none.txt", "2", processor=proc)
    first2 = fm.macro_sample("lines.txt", "2", processor=proc)
    assert first2 == "l1\nl2\n"
    rnd = fm.macro_sample("lines.txt", "2", "random", processor=proc)
    assert len(rnd.splitlines()) == 2


def test_macro_sample_empty(tmp_path, proc):
    f = tmp_path / "empty.txt"
    f.write_text("")
    assert fm.macro_sample("empty.txt", "3", processor=proc) == ""


def test_macro_json_extract(tmp_path, proc):
    f = tmp_path / "data.json"
    f.write_text(json.dumps({"a": {"b": [10, 20]}, "list": [1, 2, 3]}))
    assert fm.macro_json_extract("x").startswith("Error: JSON_EXTRACT macro requires")
    assert "File not found" in fm.macro_json_extract("nope.json", "$.a", processor=proc)
    assert fm.macro_json_extract("data.json", "$.a.b[1]", processor=proc) == "20"
    whole_list = fm.macro_json_extract("data.json", "$.list[*]", processor=proc)
    assert json.loads(whole_list) == [1, 2, 3]


def test_macro_json_extract_invalid_json(tmp_path, proc):
    f = tmp_path / "bad.json"
    f.write_text("{not json")
    assert "Invalid JSON" in fm.macro_json_extract("bad.json", "$.a", processor=proc)


def test_jsonpath_extract_edges():
    assert fm._jsonpath_extract({"a": 1}, "$") == {"a": 1}
    with pytest.raises(ValueError):
        fm._jsonpath_extract({}, "a")
    # wildcard on dict returns values
    assert fm._jsonpath_extract({"x": 1, "y": 2}, "$[*]") == [1, 2]
    # index on non-list returns None
    assert fm._jsonpath_extract({"a": 1}, "$[0]") is None
    # key on non-dict returns None
    assert fm._jsonpath_extract([1, 2], "$.key") is None
    # out of range index
    assert fm._jsonpath_extract([1], "$[5]") is None
    # wildcard on scalar
    assert fm._jsonpath_extract(5, "$[*]") == 5


def test_macro_template(tmp_path, proc):
    f = tmp_path / "tpl.txt"
    f.write_text("Hello {name}, you are {age}")
    assert fm.macro_template("x").startswith("Error: TEMPLATE macro requires")
    assert "Template file not found" in fm.macro_template("none.txt", "{}", processor=proc)
    out = fm.macro_template("tpl.txt", json.dumps({"name": "Bob", "age": 5}), processor=proc)
    assert out == "Hello Bob, you are 5"


def test_macro_template_invalid_json(tmp_path, proc):
    f = tmp_path / "tpl.txt"
    f.write_text("x")
    assert "Invalid JSON in variables" in fm.macro_template("tpl.txt", "{bad", processor=proc)


def test_macro_template_dict_vars(tmp_path, proc):
    f = tmp_path / "tpl.txt"
    f.write_text("Hi {n}")
    out = fm.macro_template("tpl.txt", {"n": "X"}, processor=proc)
    assert out == "Hi X"
