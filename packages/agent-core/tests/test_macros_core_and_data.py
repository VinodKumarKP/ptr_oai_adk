import pytest
from unittest.mock import MagicMock

# Import from the package so the monkey-patched MacroProcessor is used
from oai_agent_core.macros import MacroProcessor, DEFAULT_MACROS
from oai_agent_core.macros.data_macros import (
    macro_uuid,
    macro_random_choice,
    macro_random_int,
    macro_base64,
    macro_json_escape,
)


# ============================================================================
# MacroProcessor Tests
# ============================================================================

class TestMacroProcessor:
    def test_init_with_no_macros(self):
        proc = MacroProcessor(project_root="/root")
        assert proc.project_root == "/root"
        # Default macros are injected by __init__.py monkey-patch
        assert len(proc.macro_functions) >= len(DEFAULT_MACROS)

    def test_init_with_extra_macros(self):
        fn = lambda *a, **kw: "ok"
        proc = MacroProcessor(project_root="/root", macro_functions={"MY": fn})
        assert "MY" in proc.macro_functions
        # Default macros are also present
        assert "DATE" in proc.macro_functions

    def test_register_macro(self):
        proc = MacroProcessor(project_root="/root")
        fn = lambda *a, **kw: "registered"
        proc.register_macro("REG", fn)
        assert "REG" in proc.macro_functions

    def test_process_empty_string(self):
        proc = MacroProcessor(project_root="/root")
        assert proc.process("") == ""
        assert proc.process(None) is None

    def test_process_no_macros(self):
        proc = MacroProcessor(project_root="/root")
        text = "Hello world! No macros here."
        assert proc.process(text) == text

    def test_process_simple_macro(self):
        proc = MacroProcessor(project_root="/root")
        proc.register_macro("GREET", lambda *a, **kw: "Hello!")
        result = proc.process("{{ GREET }}")
        assert result == "Hello!"

    def test_process_macro_with_args(self):
        proc = MacroProcessor(project_root="/root")
        proc.register_macro("ADD", lambda a, b, **kw: str(int(a) + int(b)))
        result = proc.process("{{ ADD 3 5 }}")
        assert result == "8"

    def test_process_macro_with_context(self):
        proc = MacroProcessor(project_root="/root")

        def set_var(*args, context=None, **kw):
            if context is not None and len(args) >= 2:
                context[args[0]] = args[1]
            return ""

        proc.register_macro("SET", set_var)
        ctx = {}
        proc.process("{{ SET MY_KEY my_value }}", context=ctx)
        assert ctx.get("MY_KEY") == "my_value"

    def test_process_unknown_macro_unchanged(self):
        proc = MacroProcessor(project_root="/root")
        text = "{{ UNKNOWN_MACRO_XYZ }}"
        result = proc.process(text)
        # Unknown macro is kept as-is
        assert "UNKNOWN_MACRO_XYZ" in result

    def test_process_multiple_macros(self):
        proc = MacroProcessor(project_root="/root")
        proc.register_macro("A", lambda *a, **kw: "alpha")
        proc.register_macro("B", lambda *a, **kw: "beta")
        result = proc.process("{{ A }} and {{ B }}")
        assert result == "alpha and beta"

    def test_process_macro_exception_returns_error(self):
        proc = MacroProcessor(project_root="/root")

        def bad_macro(*a, **kw):
            raise RuntimeError("macro crash")

        proc.register_macro("BAD", bad_macro)
        result = proc.process("{{ BAD }}")
        assert "Error" in result

    def test_process_uses_instance_variables_by_default(self):
        proc = MacroProcessor(project_root="/root")
        proc._variables["MY_KEY"] = "stored"

        def get_var(*args, context=None, **kw):
            return context.get(args[0], "") if context else ""

        proc.register_macro("GET", get_var)
        result = proc.process("{{ GET MY_KEY }}")
        assert result == "stored"

    def test_process_shlex_parse_fallback(self):
        proc = MacroProcessor(project_root="/root")
        proc.register_macro("ECHO", lambda *a, **kw: " ".join(a))
        # Unclosed quotes trigger shlex ValueError → fall back to split
        result = proc.process("{{ ECHO hello world }}")
        assert "hello" in result


# ============================================================================
# Data Macros Tests
# ============================================================================

class TestDataMacros:
    def test_uuid(self):
        result = macro_uuid()
        # Should be valid UUID format
        assert len(result) == 36
        assert result.count("-") == 4

    def test_random_choice(self):
        result = macro_random_choice("a", "b", "c")
        assert result in ("a", "b", "c")

    def test_random_choice_no_args(self):
        result = macro_random_choice()
        assert "Error" in result

    def test_random_int(self):
        result = int(macro_random_int("1", "10"))
        assert 1 <= result <= 10

    def test_random_int_wrong_arg_count(self):
        result = macro_random_int("1")
        assert "Error" in result

    def test_random_int_invalid_args(self):
        result = macro_random_int("a", "b")
        assert "Error" in result

    def test_base64_encode(self):
        import base64
        result = macro_base64("hello world")
        decoded = base64.b64decode(result).decode("utf-8")
        assert decoded == "hello world"

    def test_base64_no_args(self):
        assert macro_base64() == ""

    def test_json_escape(self):
        result = macro_json_escape('He said "hello"')
        # Quotes should be escaped
        assert '\\"' in result

    def test_json_escape_no_args(self):
        assert macro_json_escape() == ""
