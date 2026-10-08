"""
run_dacu.py
===========
Backward-compatibility wrapper for run_dacm.py.
Redirects execution to run_dacm.
"""

from run_dacm import *  # noqa: F401, F403

if __name__ == "__main__":
    from run_dacm import __file__ as _runner_file
    import runpy
    runpy.run_path(_runner_file, run_name="__main__")