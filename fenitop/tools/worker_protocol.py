"""Typed, application-internal protocol between Tool 2 and its solver worker."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fenitop.tools.config_models import AgentSafeConfig
from fenitop.tools.contracts import RunTopoptResponse, ValidateConfigResponse
from fenitop.tools.schema import TOOL_CONTRACT_VERSION


class WorkerModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class SolverWorkerRequest(WorkerModel):
    contract_version: Literal[TOOL_CONTRACT_VERSION] = TOOL_CONTRACT_VERSION
    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    output_dir: Path
    output_prefix: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    config: AgentSafeConfig
    validation: ValidateConfigResponse
    render_snapshot: bool
    solver_profile: Literal["auto", "iterative", "direct"]
    output_interval: int = Field(ge=1, le=100)


class SolverWorkerResult(WorkerModel):
    contract_version: Literal[TOOL_CONTRACT_VERSION] = TOOL_CONTRACT_VERSION
    worker_api_key_present: bool
    response: RunTopoptResponse
