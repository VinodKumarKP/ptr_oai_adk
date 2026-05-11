from oai_agent_core.macros.core import MacroProcessor as CoreMacroProcessor
from oai_agent_core.macros import DEFAULT_MACROS

class MacroProcessor(CoreMacroProcessor):
    """
    Extends the core MacroProcessor to support dynamic value injection in test scenarios.

    Provides access to a comprehensive set of built-in macros for test scenario input/output:
    - File operations: OPEN, CONCAT, SAMPLE, LIST_FILES, IMAGE
    - Text manipulation: TRUNCATE, REPEAT, BASE64, JSON_ESCAPE, URL_ENCODE, HASH
    - Data extraction: JSON_EXTRACT, TEMPLATE
    - Date & time: DATE, NOW, TIMESTAMP, BUSINESS_DAYS_FROM
    - Random data: UUID, RANDOM_INT, RANDOM_CHOICE, FAKER, FAKER_LOCALE
    - Variables: SET, GET, IF, LOOP
    - External data: HTTP_GET, SQL_QUERY
    - Utilities: PATH, ENV, CALC

    Macro syntax: {{ MACRO_NAME args }}
    Supports nested macros and variable context sharing across input/output processing.

    Args:
        project_root: Root directory for relative path resolution.
        macro_functions: Optional dict of custom macro functions to register.
        logger: Optional logger instance for debugging macro processing.
    """

    def __init__(self, project_root: str, macro_functions=None, logger=None):
        macros = DEFAULT_MACROS.copy()
        if macro_functions:
            macros.update(macro_functions)
        super().__init__(project_root, macros, logger)
