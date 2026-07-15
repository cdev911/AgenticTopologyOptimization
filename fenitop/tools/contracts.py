"""Typed public contracts and application-owned policies for the three tools."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fenitop.tools.config_models import AgentSafeConfig, CONFIG_SCHEMA_VERSION
from fenitop.tools.schema import TOOL_CONTRACT_VERSION


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class IssueRecord(ContractModel):
    code: str
    path: str
    message: str
    severity: Literal["warning", "error"]
    retryable: bool


class CostEstimate(ContractModel):
    cell_type: Literal["quadrilateral", "triangle"]
    solver_profile: Literal["iterative", "direct"]
    num_elements: int
    num_nodes: int
    displacement_dofs: int
    num_design_variables: int
    max_iter: int
    evaluated_states: int
    linear_solves_per_iteration: int
    complexity_score: float
    work_units: float
    estimated_peak_memory_mb: float
    estimated_output_mb: float
    estimated_wall_time_seconds: float
    risk_level: Literal["low", "medium", "high"]
    exceeds_default_safety_ceiling: bool


class CheckRecord(ContractModel):
    structural: bool
    resource: bool
    geometry: bool


class EntityMatchRecord(ContractModel):
    path: str
    entity_kind: Literal["facet", "node", "cell"]
    count: int
    bounds: tuple[tuple[float, float], tuple[float, float]]


class GeometryReport(ContractModel):
    total_boundary_facets: int
    total_nodes: int
    total_cells: int
    rigid_body_rank: int
    entities: list[EntityMatchRecord]


class ArtifactRecord(ContractModel):
    role: str
    format: str
    path: str
    complete: bool = True


class ManifestArtifactRecord(ContractModel):
    role: str
    format: str
    path: str
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exists: Literal[True] = True
    complete: Literal[True] = True

    @field_validator("path")
    @classmethod
    def require_contained_relative_path(cls, value: str) -> str:
        path = Path(value)
        if (
            not value
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
        ):
            raise ValueError("Manifest artifact paths must be normalized and relative.")
        return path.as_posix()


class ValidateConfigRequest(ContractModel):
    config: AgentSafeConfig


class RunTopoptRequest(ContractModel):
    config: AgentSafeConfig


class ResourceLimits(ContractModel):
    """Trusted serial-demo admission limits; never exported as an LLM schema."""

    max_elements: int = Field(default=250_000, gt=0)
    max_nodes: int = Field(default=300_000, gt=0)
    max_displacement_dofs: int = Field(default=600_000, gt=0)
    max_iterations: int = Field(default=2_000, gt=0)
    max_work_units: float = Field(default=500_000_000.0, gt=0)
    max_peak_memory_mb: float = Field(default=2_048.0, gt=0)
    max_output_mb: float = Field(default=1_024.0, gt=0)
    max_estimated_wall_time_seconds: float = Field(default=900.0, gt=0)


class TrustedValidationPolicy(ContractModel):
    """Application-owned validation controls; never exported as an LLM schema."""

    check_geometry: bool = True
    enforce_resource_limits: bool = True
    solver_profile: Literal["auto", "iterative", "direct"] = "auto"
    output_interval: int = Field(default=20, ge=1, le=100)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)


class TrustedRunPolicy(ContractModel):
    """Application-owned execution authority; never exported as an LLM schema."""

    output_root: Path = Path("results")
    run_id: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$"
    )
    output_prefix: str | None = Field(
        default=None, pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$"
    )
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)
    render_snapshot: bool = True
    solver_profile: Literal["auto", "iterative", "direct"] = "auto"
    output_interval: int = Field(default=20, ge=1, le=100)
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    timeout_seconds: float = Field(default=900.0, gt=0)
    termination_grace_seconds: float = Field(default=2.0, gt=0, le=30)
    poll_interval_seconds: float = Field(default=0.05, gt=0, le=1)
    min_free_disk_mb: float = Field(default=128.0, ge=0)

    @field_validator("run_id", "output_prefix")
    @classmethod
    def reject_reserved_identifiers(cls, value: str | None) -> str | None:
        if value is None:
            return value
        reserved = {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)),
            *(f"LPT{i}" for i in range(1, 10)),
        }
        if value.upper() in reserved:
            raise ValueError("Identifier is a reserved filesystem name.")
        return value

    @model_validator(mode="after")
    def explicit_run_id_requires_idempotency(self):
        if self.run_id is not None and self.idempotency_key is None:
            raise ValueError(
                "An explicit run_id requires an idempotency_key so retries "
                "cannot accidentally alias unrelated work."
            )
        return self


class TrustedAnalysisPolicy(ContractModel):
    """Application-owned deterministic analysis controls."""

    allowed_roots: tuple[Path, ...] = (Path("results"),)
    grayness_threshold: float = Field(default=0.4, ge=0, le=1)
    checkerboard_threshold: float = Field(default=0.15, ge=0)
    density_threshold: float = Field(default=0.5, ge=0, le=1)
    volume_tolerance: float = Field(default=0.01, ge=0, le=0.1)
    max_grid_mb: float = Field(default=64.0, gt=0)
    max_total_artifact_mb: float = Field(default=2_048.0, gt=0)
    make_plots: bool = True


class ValidateConfigResponse(ContractModel):
    contract_version: Literal[TOOL_CONTRACT_VERSION]
    tool: Literal["validate_config"]
    status: Literal["ok", "error"]
    warnings: list[IssueRecord]
    errors: list[IssueRecord]
    stage: str | None = None
    checked: CheckRecord
    problem_type: Literal["minimize_compliance", "compliant_mechanism"] | None = None
    normalized_config: AgentSafeConfig | None = None
    estimated_cost: CostEstimate | None = None
    geometry_report: GeometryReport | None = None


class RunMetrics(ContractModel):
    final_compliance: float | None
    final_volume: float | None
    final_objective: float | None
    grayness: float | None
    binarization_score: float | None
    final_change: float | None
    opt_tol: float | None
    final_beta: float | None
    continuation_completed: bool | None


class OptimizerStatusRecord(ContractModel):
    method: Literal["oc", "mma"]
    converged: bool
    outer_iterations: int
    newton_iterations: int = 0
    line_search_iterations: int = 0
    residual_norm: float | None = None
    residual_max: float | None = None


class SolverErrorRecord(ContractModel):
    exception_type: str
    message: str
    code: str | None = None
    component: str | None = None
    iteration: int | None = None
    reason: int | None = None
    residual_norm: float | None = None
    debug_artifact_role: str | None = None


class JobLifecycleRecord(ContractModel):
    state: Literal[
        "queued", "running", "succeeded", "failed",
        "timed_out", "cancelled", "orphaned",
    ]
    run_id: str
    request_hash: str
    idempotency_key_hash: str | None = None
    created_at: str
    updated_at: str
    parent_pid: int | None = None
    worker_pid: int | None = None
    exit_code: int | None = None
    terminating_signal: int | None = None
    timed_out: bool = False
    cancelled: bool = False
    last_iteration: int | None = None
    worker_api_key_present: bool | None = None
    message: str | None = None


class IterationMetrics(ContractModel):
    state: Literal["initial", "iterate"]
    iteration: int
    beta: float
    change: float | None = None
    compliance: float | None = None
    volume: float | None = None
    objective: float | None = None
    grayness: float | None = None
    binarization_score: float | None = None
    initial_density: float | None = None
    optimizer_status: OptimizerStatusRecord | None = None
    time_seconds: float | None = None


class RuntimeVersionRecord(ContractModel):
    name: str
    version: str


class RunManifest(ContractModel):
    manifest_version: Literal["1.0"] = "1.0"
    manifest_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    contract_version: Literal[TOOL_CONTRACT_VERSION]
    config_schema_version: Literal[CONFIG_SCHEMA_VERSION]
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    run_directory: str
    output_prefix: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    problem_type: Literal["minimize_compliance", "compliant_mechanism"]
    normalized_config: AgentSafeConfig
    lifecycle: JobLifecycleRecord
    numerical_status: Literal["succeeded"]
    converged: bool
    stop_reason: Literal[
        "tolerance_met", "max_iterations_reached", "continuation_incomplete"
    ]
    iterations: int = Field(ge=0)
    metrics: RunMetrics
    optimizer_status: OptimizerStatusRecord
    mma_inner_iteration_warnings: int = Field(ge=0)
    estimated_cost: CostEstimate
    geometry_report: GeometryReport
    warnings: list[IssueRecord]
    runtime_versions: list[RuntimeVersionRecord]
    artifacts: list[ManifestArtifactRecord]

    @field_validator("run_directory")
    @classmethod
    def require_absolute_run_directory(cls, value: str) -> str:
        path = Path(value)
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("RunManifest run_directory must be an absolute path.")
        return str(path)

    @model_validator(mode="after")
    def require_successful_lifecycle(self):
        if self.lifecycle.state != "succeeded":
            raise ValueError("A successful RunManifest requires lifecycle=succeeded.")
        if self.lifecycle.run_id != self.run_id:
            raise ValueError("RunManifest and lifecycle run IDs must match.")
        if self.lifecycle.request_hash != self.request_hash:
            raise ValueError("RunManifest and lifecycle request hashes must match.")
        if self.normalized_config.opt.problem_type != self.problem_type:
            raise ValueError("RunManifest problem type must match its normalized config.")
        required_metrics = (
            "final_compliance",
            "final_volume",
            "final_objective",
            "grayness",
            "binarization_score",
            "final_change",
            "opt_tol",
            "final_beta",
            "continuation_completed",
        )
        if any(getattr(self.metrics, name) is None for name in required_metrics):
            raise ValueError("A successful RunManifest requires complete final metrics.")
        if not self.optimizer_status.converged:
            raise ValueError("A successful RunManifest requires optimizer success.")
        return self


class RunTopoptResponse(ContractModel):
    contract_version: Literal[TOOL_CONTRACT_VERSION]
    tool: Literal["run_topopt"]
    status: Literal["ok", "error"]
    warnings: list[IssueRecord]
    errors: list[IssueRecord]
    stage: str | None = None
    message: str | None = None
    run_id: str | None = None
    problem_type: Literal["minimize_compliance", "compliant_mechanism"] | None = None
    converged: bool | None = None
    stop_reason: str | None = None
    iterations: int | None = None
    metrics: RunMetrics | None = None
    optimizer_status: OptimizerStatusRecord | None = None
    mma_inner_iteration_warnings: int | None = None
    wall_time_seconds: float | None = None
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    lifecycle: JobLifecycleRecord | None = None
    idempotent_replay: bool = False
    validation: ValidateConfigResponse | None = None
    estimated_cost: CostEstimate | None = None
    error: SolverErrorRecord | None = None
    last_known_good_metrics: IterationMetrics | None = None
    rank: int | None = None
    note: str | None = None
    run_manifest: RunManifest | None = None


class AnalyzeResultsRequest(ContractModel):
    run_manifest: RunManifest = Field(
        description="The exact successful RunManifest returned by run_topopt."
    )


class AnalysisSource(ContractModel):
    run_directory: str
    output_prefix: str
    run_id: str
    manifest_hash: str


class ConvergenceAnalysis(ContractModel):
    converged: bool | None
    stop_reason: Literal[
        "tolerance_met", "max_iterations_reached", "continuation_incomplete", "unknown"
    ]
    iterations: int | None
    final_change: float | None
    opt_tol: float | None
    fraction_iterations_at_move_limit: float | None
    move_limit: float | None
    final_beta: float | None = None
    continuation_completed: bool | None = None
    iteration_cap_reached: bool
    move_limit_pinned: bool | None
    oscillation_detected: bool | None
    plateau_detected: bool | None
    optimizer_warning_count: int = Field(ge=0)


class ConnectivityRecord(ContractModel):
    region_kind: Literal["traction", "spring"]
    region_index: int = Field(ge=0)
    matched_grid_points: int = Field(ge=0)
    connected_to_support: bool | None
    nearby_component_labels: list[int]


class QualityFlags(ContractModel):
    grayness: float | None
    binarization_score: float | None
    grayness_threshold: float
    high_grayness_warning: bool
    checkerboard_detected: bool | None
    checkerboard_score: float | None
    num_components: int | None
    largest_component_fraction: float | None
    has_disconnected_material: bool | None
    load_path_connected: bool | None
    checkerboard_method: str | None
    connectivity_method: str | None
    connectivity: list[ConnectivityRecord]


class ConstraintAnalysis(ContractModel):
    volume_target: float
    volume_error: float
    volume_tolerance: float
    volume_satisfied: bool
    compliance_bound: float | None
    compliance_bound_satisfied: bool | None
    density_bounds_satisfied: bool


class AnalysisMetrics(ContractModel):
    final_compliance: float | None
    final_volume: float | None
    final_objective: float | None
    constraints: ConstraintAnalysis


class PlotRecord(ContractModel):
    role: str
    path: str
    source: str | None = None


class AnalyzeResultsResponse(ContractModel):
    contract_version: Literal[TOOL_CONTRACT_VERSION]
    tool: Literal["analyze_results"]
    status: Literal["ok", "error"]
    warnings: list[IssueRecord]
    errors: list[IssueRecord]
    stage: str | None = None
    message: str | None = None
    source: AnalysisSource | None = None
    convergence: ConvergenceAnalysis | None = None
    quality_flags: QualityFlags | None = None
    metrics: AnalysisMetrics | None = None
    plots: list[PlotRecord] = Field(default_factory=list)
    narrative: str | None = None


def agent_tool_schemas() -> dict[str, dict]:
    """The only schemas intended for an LLM/CrewAI/MCP adapter."""
    return {
        "validate_config": ValidateConfigRequest.model_json_schema(),
        "run_topopt": RunTopoptRequest.model_json_schema(),
        "analyze_results": AnalyzeResultsRequest.model_json_schema(),
    }
