"""Additional coverage for data_macros."""
import sys
import uuid
from unittest.mock import MagicMock

import pytest

from oai_agent_core.macros import data_macros as dm


def test_macro_uuid_is_valid():
    val = dm.macro_uuid()
    uuid.UUID(val)  # does not raise


def test_random_choice():
    assert dm.macro_random_choice("a") == "a"
    assert dm.macro_random_choice() .startswith("Error")
    assert dm.macro_random_choice("x", "y") in ("x", "y")


def test_random_int():
    assert dm.macro_random_int("1", "1") == "1"
    assert dm.macro_random_int("1").startswith("Error")
    assert dm.macro_random_int("a", "b").startswith("Error")


def test_base64():
    assert dm.macro_base64() == ""
    assert dm.macro_base64("hello") == "aGVsbG8="


def test_json_escape():
    assert dm.macro_json_escape() == ""
    assert dm.macro_json_escape('a"b') == 'a\\"b'


def _install_fake_faker(monkeypatch, fake_obj=None, raise_init=False):
    module = MagicMock()

    class FakeFaker:
        def __init__(self, locale="en_US"):
            if raise_init:
                raise RuntimeError("bad locale")
            self._obj = fake_obj

        def __getattr__(self, item):
            return getattr(self._obj, item)

    module.Faker = FakeFaker
    monkeypatch.setitem(sys.modules, "faker", module)


def test_faker_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "faker", None)
    assert dm.macro_faker("name").startswith("Error: faker package")


def test_faker_requires_method(monkeypatch):
    class Obj:
        pass

    _install_fake_faker(monkeypatch, Obj())
    assert dm.macro_faker().startswith("Error: FAKER macro requires")


def test_faker_init_failure(monkeypatch):
    _install_fake_faker(monkeypatch, raise_init=True)
    assert dm.macro_faker("name").startswith("Error initializing Faker")


def test_faker_unknown_method(monkeypatch):
    class Obj:
        pass

    _install_fake_faker(monkeypatch, Obj())
    assert "no method" in dm.macro_faker("nonexistent_method_zzz")


def test_faker_success_with_arg_conversion(monkeypatch):
    class Obj:
        def number(self, a, b):
            return a + b

    _install_fake_faker(monkeypatch, Obj())
    assert dm.macro_faker("number", "1", "2") == "3"


def test_faker_method_raises(monkeypatch):
    class Obj:
        def boom(self):
            raise ValueError("kaboom")

    _install_fake_faker(monkeypatch, Obj())
    assert dm.macro_faker("boom").startswith("Error executing Faker")


def test_faker_locale_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "faker", None)
    assert dm.macro_faker_locale("fr_FR", "name").startswith("Error: faker package")


def test_faker_locale_requires_args(monkeypatch):
    class Obj:
        pass

    _install_fake_faker(monkeypatch, Obj())
    assert dm.macro_faker_locale("fr_FR").startswith("Error: FAKER_LOCALE macro requires")


def test_faker_locale_init_failure(monkeypatch):
    _install_fake_faker(monkeypatch, raise_init=True)
    assert dm.macro_faker_locale("xx", "name").startswith("Error initializing Faker")


def test_faker_locale_unknown_method(monkeypatch):
    class Obj:
        pass

    _install_fake_faker(monkeypatch, Obj())
    assert "no method" in dm.macro_faker_locale("fr_FR", "nope_zzz")


def test_faker_locale_success_with_float(monkeypatch):
    class Obj:
        def echo(self, v):
            return v

    _install_fake_faker(monkeypatch, Obj())
    assert dm.macro_faker_locale("fr_FR", "echo", "1.5") == "1.5"


def test_faker_locale_method_raises(monkeypatch):
    class Obj:
        def boom(self):
            raise ValueError("x")

    _install_fake_faker(monkeypatch, Obj())
    assert dm.macro_faker_locale("fr_FR", "boom").startswith("Error executing Faker")
