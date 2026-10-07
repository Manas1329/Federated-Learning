"""
dacu.py
=======
Backward-compatibility wrapper for dacm.py.
Redirects all imports and executions to the canonical DACM engine.
"""

from dacm import *  # noqa: F401, F403

if __name__ == "__main__":
    import dacm
    import sys

    # Forward CLI execution to dacm
    from dacm import __file__ as _dacm_file
    import runpy
    runpy.run_path(_dacm_file, run_name="__main__")