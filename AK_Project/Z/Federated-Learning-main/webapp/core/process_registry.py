from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class ProcessRole(str, Enum):
    ADMIN = "admin"
    HOSPITAL = "hospital"
    EITHER = "either"


@dataclass(frozen=True)
class InputParameter:
    name: str
    label: str
    kind: str = "str"
    required: bool = False
    default: Any = None
    options: Optional[List[Any]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    step: Optional[float] = None
    help: Optional[str] = None


@dataclass(frozen=True)
class ProcessDefinition:
    process_type: str
    title: str
    description: str
    launcher_method: str
    role: ProcessRole = ProcessRole.ADMIN
    hospital_bound: bool = False
    input_parameters: List[InputParameter] = field(default_factory=list)
    output_roots: List[str] = field(default_factory=list)
    audit_event: Optional[str] = None


PROCESS_DEFINITIONS: List[ProcessDefinition] = [
    ProcessDefinition(
        process_type="init_db",
        title="Initialize / Seed Database",
        description="Run the database seed script to create tables and populate default admin, hospitals, and demo data.",
        launcher_method="start_seed_database",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[
            InputParameter(
                name="reset_first",
                label="Reset existing data first",
                kind="bool",
                required=False,
                default=False,
                help="If true, tables are dropped and recreated before seeding. Leave false for idempotent refresh (recommended)."
            )
        ],
        output_roots=["webapp/database"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="split_data",
        title="Prepare & Split Dataset",
        description="Split the raw chest X-ray dataset non-IID across hospital nodes using the configured hospital specs.",
        launcher_method="start_split_data",
        role=ProcessRole.EITHER,
        hospital_bound=False,
        input_parameters=[
            InputParameter(
                name="RAW_DATA_DIR",
                label="Raw dataset directory (NORMAL/PNEUMONIA)",
                kind="path",
                required=False,
                help="Override the default raw chest X-ray source directory."
            ),
            InputParameter(
                name="HOSPITAL_SPEC_OVERRIDE",
                label="Hospital split spec (JSON, optional)",
                kind="json",
                required=False,
                help='Optional JSON map like {"hospital_A":{"NORMAL":1000,"PNEUMONIA":250}}.'
            )
        ],
        output_roots=["data"],
        audit_event="DATA_ACTION"
    ),
    ProcessDefinition(
        process_type="init_model",
        title="Initialize Global Model",
        description="Create the initial ChestCNN global model checkpoint files (pure, quantized, DP variants).",
        launcher_method="initialize_global_model",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[],
        output_roots=["federated_healthcare/models"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="fl_server",
        title="Start FL Server",
        description="Start the centralized federated aggregation server for the configured number of rounds.",
        launcher_method="start_server",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[
            InputParameter(name="target_clients", label="Target clients per round", kind="int", required=True, default=3, min=1, max=50, step=1),
            InputParameter(name="min_clients", label="Minimum clients for aggregation", kind="int", required=True, default=2, min=1, max=50, step=1),
            InputParameter(name="num_rounds", label="Number of rounds", kind="int", required=True, default=20, min=1, max=500, step=1),
            InputParameter(name="USE_QUANTIZATION", label="INT8 quantization", kind="bool", required=False, default=True),
            InputParameter(name="USE_DP", label="Differential Privacy", kind="bool", required=False, default=False),
            InputParameter(name="FORCE_CPU", label="Force CPU on server side", kind="bool", required=False, default=False),
            InputParameter(name="ADAPTIVE_DROPOUT_ENABLED", label="Adaptive dropout engine", kind="bool", required=False, default=True),
            InputParameter(name="FIXED_DEADLINE_CONTROL", label="Fixed deadline control", kind="bool", required=False, default=False),
            InputParameter(name="DROPOUT_HARD_DEADLINE", label="Dropout hard deadline (seconds)", kind="float", required=False, default=60.0, min=1.0, step=1.0),
            InputParameter(name="ROUND_TIMEOUT", label="Round timeout (seconds)", kind="float", required=False, default=300.0, min=1.0, step=1.0)
        ],
        output_roots=["federated_healthcare/dashboard/results", "federated_healthcare/models"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="fl_client",
        title="Start FL Client Node",
        description="Start a hospital's local FL client that trains on its dataset and sends updates to the server.",
        launcher_method="start_client",
        role=ProcessRole.HOSPITAL,
        hospital_bound=True,
        input_parameters=[
            InputParameter(name="server_address", label="Server address (host:port)", kind="str", required=True, default="localhost:8080"),
            InputParameter(name="data_path", label="Local data directory", kind="path", required=False),
            InputParameter(name="use_quantization", label="INT8 quantization", kind="bool", required=False, default=True),
            InputParameter(name="use_dp", label="Differential Privacy", kind="bool", required=False, default=False),
            InputParameter(name="force_cpu", label="Force CPU", kind="bool", required=False, default=False)
        ],
        output_roots=["federated_healthcare/dashboard/results", "data"],
        audit_event="TRAINING_START"
    ),
    ProcessDefinition(
        process_type="evaluate",
        title="Run Global Model Evaluation",
        description="Evaluate the latest global model checkpoint on the test set and save plots + classification report.",
        launcher_method="start_evaluate",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[
            InputParameter(name="USE_QUANTIZATION", label="Evaluate quantized variant", kind="bool", required=False, default=True),
            InputParameter(name="USE_DP", label="Evaluate DP variant", kind="bool", required=False, default=False),
            InputParameter(name="FORCE_CPU", label="Force CPU inference", kind="bool", required=False, default=False),
            InputParameter(name="DATA_PATH", label="Test data path (optional)", kind="path", required=False)
        ],
        output_roots=["federated_healthcare/dashboard/plots", "federated_healthcare/dashboard/classification_reports"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="evaluate_comparison",
        title="Run Local vs Global Model Comparison",
        description="Compare a local model checkpoint against the global model across all hospital test sets.",
        launcher_method="start_evaluate_comparison",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[
            InputParameter(name="USE_QUANTIZATION", label="Use quantized global model suffix", kind="bool", required=False, default=True),
            InputParameter(name="USE_DP", label="Use DP global model suffix", kind="bool", required=False, default=False),
            InputParameter(name="FORCE_CPU", label="Force CPU inference", kind="bool", required=False, default=False)
        ],
        output_roots=["federated_healthcare/dashboard/plots/comparisons"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="graph",
        title="Generate Single-Run Graphs",
        description="Read the latest metrics CSVs and generate accuracy/loss curves plus cross-hospital comparison plots.",
        launcher_method="start_graph",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[
            InputParameter(name="USE_QUANTIZATION", label="Quantized suffix", kind="bool", required=False, default=True),
            InputParameter(name="USE_DP", label="DP suffix", kind="bool", required=False, default=False)
        ],
        output_roots=["federated_healthcare/dashboard/plots"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="comparison_graph",
        title="Generate Cross-Config Comparison Graphs",
        description="Produce accuracy/loss/payload comparison figures across the three global model configs.",
        launcher_method="start_comparison_graph",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[],
        output_roots=["federated_healthcare/dashboard/comparisons"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="comparison_table",
        title="Compute Local vs Federated Comparison Table",
        description="Generate comparison_results.csv comparing standalone local training vs federated global model performance.",
        launcher_method="start_comparison_table",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[],
        output_roots=["federated_healthcare/dashboard/results"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="parse_results",
        title="Parse Experiment Logs",
        description="Parse all experiment execution logs into the final_experiment_results.csv summary.",
        launcher_method="start_parse_results",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[],
        output_roots=["federated_healthcare/dashboard/results/ADSM_results/experiments"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="aggregate_results",
        title="Aggregate Experiment Summaries",
        description="Walk all experiment run directories and create experiment_summary.csv + experiment_summary.md.",
        launcher_method="start_aggregate_results",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[],
        output_roots=["federated_healthcare/dashboard/results/ADSM_results/experiments"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="generate_figures",
        title="Generate Publication Figures",
        description="Render the four publication figures (PNG + PDF) from final_experiment_results.csv.",
        launcher_method="start_generate_figures",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[],
        output_roots=["federated_healthcare/dashboard/plots/figures"],
        audit_event="ADMIN_ACTION"
    ),
    ProcessDefinition(
        process_type="run_experiment",
        title="Run Experiment",
        description="Run a single experiment preset (e.g. exp1_baseline) for a number of repeated runs.",
        launcher_method="start_run_experiment",
        role=ProcessRole.ADMIN,
        hospital_bound=False,
        input_parameters=[
            InputParameter(
                name="exp",
                label="Experiment preset",
                kind="str",
                required=True,
                default="exp1_baseline",
                options=[
                    "exp1_baseline",
                    "exp2_one_straggler",
                    "exp3_two_stragglers",
                    "exp4_network_dropout",
                    "exp5_recovery",
                    "exp6_adaptive_off",
                    "exp7_nonstationary",
                    "exp8_fixed_deadline",
                    "all"
                ]
            ),
            InputParameter(name="runs", label="Repeat runs", kind="int", required=True, default=1, min=1, max=50, step=1),
            InputParameter(name="num_rounds", label="Override FL rounds (optional)", kind="int", required=False, min=1, max=500, step=1),
            InputParameter(name="EXPERIMENT_TIMEOUT_SEC", label="Experiment timeout (seconds)", kind="int", required=False, default=600, min=60, step=10)
        ],
        output_roots=["federated_healthcare/dashboard/results/ADSM_results/experiments"],
        audit_event="ADMIN_ACTION"
    )
]

PROCESS_DEFINITIONS_BY_TYPE: Dict[str, ProcessDefinition] = {
    p.process_type: p for p in PROCESS_DEFINITIONS
}


def list_processes_for_role(role: ProcessRole) -> List[ProcessDefinition]:
    return [
        p for p in PROCESS_DEFINITIONS
        if p.role == ProcessRole.EITHER or p.role == role
    ]


def get_process_definition(process_type: str) -> Optional[ProcessDefinition]:
    return PROCESS_DEFINITIONS_BY_TYPE.get(process_type)
