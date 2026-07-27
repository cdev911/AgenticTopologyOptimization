"""Typed public contracts and application-owned policies for the three tools."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fenitop.tools.config_models import AgentSafeConfig
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
    scoped_output: bool = True
    render_snapshot: bool = True
    solver_profile: Literal["auto", "iterative", "direct"] = "auto"
    output_interval: int = Field(default=20, ge=1, le=100)
    allow_large_run: bool = False
    resource_limits: ResourceLimits = Field(default_factory=ResourceLimits)
    timeout_seconds: float = Field(default=900.0, gt=0)


class TrustedAnalysisPolicy(ContractModel):
    """Application-owned deterministic analysis controls."""

    allowed_roots: tuple[Path, ...] = (Path("results"),)
    grayness_threshold: float = Field(default=0.4, ge=0, le=1)
    checkerboard_threshold: float = Field(default=0.15, ge=0)
    density_threshold: float = Field(default=0.5, ge=0, le=1)
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
    final_change: float | None
    opt_tol: float | None


class SolverErrorRecord(ContractModel):
    exception_type: str
    message: str
    traceback: str


class IterationMetrics(ContractModel):
    state: Literal["initial", "iterate"]
    iteration: int
    beta: float
    change: float | None = None
    compliance: float | None = None
    volume: float | None = None
    objective: float | None = None
    grayness: float | None = None
    initial_density: float | None = None
    time_seconds: float | None = None


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
    mma_inner_iteration_warnings: int | None = None
    wall_time_seconds: float | None = None
    artifacts: list[ArtifactRecord] = Field(default_factory=list)
    validation: ValidateConfigResponse | None = None
    estimated_cost: CostEstimate | None = None
    error: SolverErrorRecord | None = None
    last_known_good_metrics: IterationMetrics | None = None
    rank: int | None = None
    note: str | None = None


class AnalyzeResultsRequest(ContractModel):
    run_topopt_envelope: RunTopoptResponse = Field(
        description="The exact successful envelope returned by run_topopt."
    )


class AnalysisSource(ContractModel):
    output_folder: str
    output_prefix: str
    run_id: str | None


class ConvergenceAnalysis(ContractModel):
    converged: bool | None
    stop_reason: Literal[
        "tolerance_met", "max_iterations_reached", "unknown"
    ]
    iterations: int | None
    final_change: float | None
    opt_tol: float | None
    fraction_iterations_at_move_limit: float | None
    move_limit: float | None


class QualityFlags(ContractModel):
    grayness: float | None
    binarization_score: float | None
    grayness_threshold: float
    low_grayness_warning: bool
    checkerboard_detected: bool | None
    checkerboard_score: float | None
    num_components: int | None
    largest_component_fraction: float | None
    has_disconnected_material: bool | None
    load_path_connected: bool | None


class AnalysisMetrics(ContractModel):
    final_compliance: float | None
    final_volume: float | None
    final_objective: float | None
    vol_frac_target: float | None


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
