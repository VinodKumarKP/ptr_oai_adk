import logging
import re
import shlex
from typing import Dict, Callable, Optional, Any

class MacroProcessor:
    """Handles processing of macros in text strings."""

    def __init__(self, project_root: str, macro_functions: Optional[Dict[str, Callable]] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the MacroProcessor.

        Args:
            project_root: The root directory of the project, used for resolving paths.
            macro_functions: Optional dictionary of custom macro functions.
            logger: Optional logger instance.
        """
        self.project_root = project_root
        self.logger = logger or logging.getLogger(__name__)
        self.macro_functions = {}
        
        if macro_functions:
            self.macro_functions.update(macro_functions)
        
        # Context for storing variables (scoped per process call ideally, but for now instance level)
        self._variables: Dict[str, Any] = {}

    def register_macro(self, name: str, func: Callable):
        """Register a new macro function."""
        self.macro_functions[name] = func

    def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process macros in the text.
        
        Args:
            text: The text containing macros.
            context: Optional dictionary to store variables for SET/GET macros. 
                     If None, uses the instance's global variable store.
        """
        if not text:
            return text

        current_context = context if context is not None else self._variables

        def replace_match(match):
            content = match.group(1).strip()
            if not content:
                return match.group(0)
            
            try:
                parts = shlex.split(content)
            except ValueError:
                parts = content.split()
                
            if not parts:
                return match.group(0)
                
            func_name = parts[0]
            args = parts[1:]
            
            if func_name in self.macro_functions:
                try:
                    # Bind 'self' if the function expects it (method of a class)
                    # But here functions are standalone or bound methods.
                    # We pass 'context' and 'processor' (self) as kwargs if accepted
                    return str(self.macro_functions[func_name](*args, context=current_context, processor=self))
                except TypeError:
                    # Fallback for macros that don't accept context/processor
                    try:
                        return str(self.macro_functions[func_name](*args, context=current_context))
                    except TypeError:
                         try:
                            return str(self.macro_functions[func_name](*args))
                         except Exception as e:
                            self.logger.error(f"Error executing macro {func_name}: {e}")
                            return f"Error executing macro {func_name}: {e}"
                except Exception as e:
                    self.logger.error(f"Error executing macro {func_name}: {e}")
                    return f"Error executing macro {func_name}: {e}"
            return match.group(0)

        # Loop to handle nested macros
        max_iterations = 10
        for _ in range(max_iterations):
            # 1. Try to match innermost macros first (no nested braces)
            new_text, n = re.subn(r'\{\{([^{}]+?)\}\}', replace_match, text)
            
            if n > 0:
                text = new_text
                continue
            
            # 2. If no innermost macros found, try to match generic macros
            new_text, n = re.subn(r'\{\{(.*?)\}\}', replace_match, text)
            
            if n > 0:
                if new_text == text:
                     break
                text = new_text
                continue
                
            # No macros found
            break
            
        return text
