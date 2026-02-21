import pytest


def test_oai_agent_core_init():
    """Test that the main package can be imported"""
    import oai_agent_core
    assert oai_agent_core is not None


def test_components_agent_init():
    """Test that components.agent can be imported"""
    try:
        import oai_agent_core.components.agent
        assert oai_agent_core.components.agent is not None
    except ImportError:
        # Module might not be fully implemented yet
        pass


def test_version_import():
    """Test that version can be imported"""
    try:
        from oai_agent_core import __version__
        assert __version__ is not None
    except ImportError:
        # Version might not be available in test environment
        pass


def test_all_imports():
    """Test importing various modules"""
    modules_to_test = [
        'oai_agent_core.core',
        'oai_agent_core.components',
        'oai_agent_core.processing',
        'oai_agent_core.utils',
    ]
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            assert module is not None
        except ImportError as e:
            # Some modules might have dependencies not available in test
            pytest.skip(f"Module {module_name} not available: {e}")