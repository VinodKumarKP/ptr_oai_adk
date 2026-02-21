from oai_agent_evaluator.macros.core import MacroProcessor as CoreMacroProcessor
from oai_agent_evaluator.macros import DEFAULT_MACROS

class MacroProcessor(CoreMacroProcessor):
    def __init__(self, project_root: str, macro_functions=None, logger=None):
        macros = DEFAULT_MACROS.copy()
        if macro_functions:
            macros.update(macro_functions)
        super().__init__(project_root, macros, logger)
