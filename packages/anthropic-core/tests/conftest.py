import os
import sys

# Make the shared ``oai_agent_core`` namespace resolvable for tests: add both this
# package's src and agent-core's src so base classes and the anthropic_core modules
# import against this monorepo.
_here = os.path.dirname(__file__)
_anthropic_src = os.path.abspath(os.path.join(_here, "..", "src"))
_agent_core_src = os.path.abspath(os.path.join(_here, "..", "..", "agent-core", "src"))

for _p in (_anthropic_src, _agent_core_src):
    if _p not in sys.path:
        sys.path.insert(0, _p)
