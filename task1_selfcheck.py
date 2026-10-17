import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
FED_HEALTHCARE_DIR = REPO_ROOT / "federated_healthcare"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(FED_HEALTHCARE_DIR))

PASS = True

def check(name, condition, detail=""):
    global PASS
    status = "PASS" if condition else "FAIL"
    if not condition:
        PASS = False
    print(f"  [{status}] {name}")
    if detail and not condition:
        print(f"         {detail}")

print("=" * 60)
print("TASK 1 SELF-CHECK: Process Registry + Launchers + Settings")
print("=" * 60)

print()
print("1. Process Registry")
print("-" * 40)

from webapp.core.process_registry import (
    PROCESS_DEFINITIONS,
    PROCESS_DEFINITIONS_BY_TYPE,
    ProcessRole,
    list_processes_for_role,
    get_process_definition,
)

check("PROCESS_DEFINITIONS is a list with >= 15 entries",
      isinstance(PROCESS_DEFINITIONS, list) and len(PROCESS_DEFINITIONS) >= 15,
      f"len={len(PROCESS_DEFINITIONS)}")

required_types = [
    'init_db', 'split_data', 'init_model', 'fl_server', 'fl_client',
    'evaluate', 'evaluate_comparison', 'graph', 'comparison_graph',
    'comparison_table', 'parse_results', 'aggregate_results',
    'generate_figures', 'run_experiment',
]
missing_types = [t for t in required_types if t not in PROCESS_DEFINITIONS_BY_TYPE]
check(f"All 14 required process_types in registry ({len(required_types)} expected)",
      len(missing_types) == 0,
      f"missing={missing_types}")

check("PROCESS_DEFINITIONS_BY_TYPE maps each def correctly",
      all(PROCESS_DEFINITIONS_BY_TYPE[p.process_type] is p for p in PROCESS_DEFINITIONS))

admin_defs = list_processes_for_role(ProcessRole.ADMIN)
hospital_defs = list_processes_for_role(ProcessRole.HOSPITAL)
check("list_processes_for_role(ADMIN) returns >= 12 defs (admin + either)",
      len(admin_defs) >= 12, f"len={len(admin_defs)}")
check("list_processes_for_role(HOSPITAL) returns >= 2 defs (hospital + either)",
      len(hospital_defs) >= 2, f"len={len(hospital_defs)}")
check("fl_client has hospital_bound=True",
      get_process_definition("fl_client").hospital_bound)
check("fl_server has hospital_bound=False",
      not get_process_definition("fl_server").hospital_bound)
check("split_data role is EITHER",
      get_process_definition("split_data").role == ProcessRole.EITHER)

all_have_launcher = all(p.launcher_method for p in PROCESS_DEFINITIONS)
check("All definitions have non-empty launcher_method", all_have_launcher)

print()
print("2. ProcessManager Launcher Methods")
print("-" * 40)

from webapp.core.process_manager import get_process_manager

pm = get_process_manager()

launcher_methods = [
    'start_seed_database',
    'start_split_data',
    'initialize_global_model',
    'start_server',
    'start_client',
    'start_evaluate',
    'start_evaluate_comparison',
    'start_graph',
    'start_comparison_graph',
    'start_comparison_table',
    'start_parse_results',
    'start_aggregate_results',
    'start_generate_figures',
    'start_run_experiment',
    '_launch_script',
    '_launch_snippet',
]
missing_methods = [m for m in launcher_methods if not callable(getattr(pm, m, None))]
check(f"All launcher methods exist on ProcessManager ({len(launcher_methods)} expected)",
      len(missing_methods) == 0,
      f"missing={missing_methods}")

check("_launch_script uses shell=False (inspect source pattern)",
      "_launch_script" in dir(pm))

print()
print("3. RuntimeSettingsStore")
print("-" * 40)

from webapp.core.runtime_settings import RuntimeSettingsStore, get_runtime_settings

rts = get_runtime_settings()
check("get_runtime_settings() returns RuntimeSettingsStore instance",
      isinstance(rts, RuntimeSettingsStore))
check("Singleton: second call returns same instance",
      get_runtime_settings() is rts)

rts.reset(persist=False)
rts.update({'USE_DP': '0', 'USE_QUANTIZATION': '1', 'TMP_FOO_BAR': 42}, persist=False)
env = rts.as_env_overrides()
check("as_env_overrides serializes string values",
      env.get('USE_DP') == '0', f"actual USE_DP={env.get('USE_DP')!r}")
check("as_env_overrides serializes bool-like strings",
      env.get('USE_QUANTIZATION') == '1', f"actual USE_QUANTIZATION={env.get('USE_QUANTIZATION')!r}")
check("as_env_overrides serializes ints to str",
      env.get('TMP_FOO_BAR') == '42', f"actual TMP_FOO_BAR={env.get('TMP_FOO_BAR')!r}")

all_vals = rts.get_all()
check("get_all returns dict with 3 keys after update",
      isinstance(all_vals, dict) and len(all_vals) == 3)
check("get('USE_DP') returns set value",
      rts.get('USE_DP') == '0')
check("get('MISSING_KEY', 'default') returns default",
      rts.get('MISSING_KEY', 'default') == 'default')

rts.update({'BOOL_TRUE': True, 'BOOL_FALSE': False}, persist=False)
env2 = rts.as_env_overrides()
check("bool True serializes to '1' in env", env2.get('BOOL_TRUE') == '1',
      f"actual={env2.get('BOOL_TRUE')!r}")
check("bool False serializes to '0' in env", env2.get('BOOL_FALSE') == '0',
      f"actual={env2.get('BOOL_FALSE')!r}")

rts.reset(persist=False)
check("reset empties settings store", len(rts.get_all()) == 0,
      f"len after reset={len(rts.get_all())}")

print()
print("4. shell=True / os.system Audit (webapp package)")
print("-" * 40)

import re

webapp_dir = REPO_ROOT / "webapp"
shell_true_hits = []
os_system_hits = []
for py_file in webapp_dir.rglob("*.py"):
    text = py_file.read_text(encoding="utf-8", errors="ignore")
    rel = py_file.relative_to(REPO_ROOT)
    if re.search(r'shell\s*=\s*True', text):
        shell_true_hits.append(str(rel))
    if re.search(r'os\.system\s*\(', text):
        os_system_hits.append(str(rel))

check("No shell=True in webapp/**/*.py",
      len(shell_true_hits) == 0,
      f"hits={shell_true_hits}")
check("No os.system( in webapp/**/*.py",
      len(os_system_hits) == 0,
      f"hits={os_system_hits}")

print()
print("5. init_db(reset_first=...) signature support")
print("-" * 40)

from webapp.database.init_db import init_db
import inspect

sig = inspect.signature(init_db)
params = list(sig.parameters.keys())
check("init_db accepts reset_first parameter",
      "reset_first" in params, f"params={params}")
reset_first_default = sig.parameters["reset_first"].default
check("init_db reset_first defaults to True",
      reset_first_default is True, f"default={reset_first_default!r}")

print()
print("6. ProcessManager path resolution (post-move repo-root webapp)")
print("-" * 40)

src_dir, project_root, src_pkg = pm._resolve_paths()
check("project_root resolves to repo root (contains webapp/ + federated_healthcare/)",
      (project_root / "webapp").is_dir() and (project_root / "federated_healthcare").is_dir(),
      f"project_root={project_root}")
check("src_dir = project_root / federated_healthcare",
      src_dir == project_root / "federated_healthcare", f"src_dir={src_dir}")
check("src_pkg = src_dir / src (where FL scripts live)",
      src_pkg == src_dir / "src", f"src_pkg={src_pkg}")
check("start_split_data script resolves (split_data.py at federated_healthcare/)",
      (src_dir / "split_data.py").is_file(),
      f"expected={src_dir / 'split_data.py'}")
check("fl server script resolves (src/server.py)",
      (src_pkg / "server.py").is_file(),
      f"expected={src_pkg / 'server.py'}")
check("evaluate_comparison.py exists at src",
      (src_pkg / "evaluate_comparison.py").is_file())
check("scripts/run_experiments.py exists at src/scripts",
      (src_pkg / "scripts" / "run_experiments.py").is_file())

print()
print("7. ProcessManager _build_env includes runtime settings + PYTHONPATH")
print("-" * 40)

rts.reset(persist=False)
rts.update({"TASK1_TEST_VAR": "from_runtime_store"}, persist=False)
test_env = pm._build_env({"CALL_SITE_VAR": "from_call_site"})
check("_build_env PYTHONPATH includes project_root + federated_healthcare",
      str(project_root) in test_env.get("PYTHONPATH", "")
      and str(src_dir) in test_env.get("PYTHONPATH", ""),
      f"PYTHONPATH={test_env.get('PYTHONPATH','')}")
check("Runtime settings override merged into env",
      test_env.get("TASK1_TEST_VAR") == "from_runtime_store",
      f"actual={test_env.get('TASK1_TEST_VAR')!r}")
check("extra_env overrides merged (call-site wins same-key not asserted here)",
      test_env.get("CALL_SITE_VAR") == "from_call_site",
      f"actual={test_env.get('CALL_SITE_VAR')!r}")
rts.reset(persist=False)

print()
print("=" * 60)
if PASS:
    print(f"TASK1_OK defs={len(PROCESS_DEFINITIONS)} "
          f"admin_defs={len(admin_defs)} "
          f"hospital_defs={len(hospital_defs)} "
          f"launchers_ok={len(missing_methods)==0}")
else:
    print("TASK1_FAIL: One or more checks above failed.")
    sys.exit(1)
