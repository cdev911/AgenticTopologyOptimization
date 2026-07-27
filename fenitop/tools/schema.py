"""Shared JSON envelope helpers for the agent-facing tools.

Every tool (validate_config, run_topopt, analyze_results) returns the same
envelope shape: {"tool": ..., "status": "ok"|"error", "warnings": [...], ...}.
Finer distinctions (validation failure vs. safety-ceiling rejection vs.
runtime crash) go in a "stage" field, never a third status value, so a caller
has one robust branch to check regardless of which tool it called.
"""
import dataclasses
from pathlib import Path
from typing import Any, Iterable, Optional

TOOL_CONTRACT_VERSION = "4.0.0"


@dataclasses.dataclass
class FieldError:
    """A single validation/runtime error, pointed at a dotted config path."""
    path: str
    code: str
    message: str
    severity: str = "error"
    retryable: bool = False


@dataclasses.dataclass
class WarningRecord:
    code: str
    path: str
    message: str
    severity: str = "warning"
    retryable: bool = False


def _warning_records(warnings: Optional[Iterable[Any]]) -> list:
    records = []
    for warning in warnings or []:
        if isinstance(warning, str):
            records.append(
                WarningRecord(
                    code="general_warning",
                    path="<root>",
                    message=warning,
                )
            )
        else:
            records.append(warning)
    return records


def jsonify(value: Any) -> Any:
    """Recursively convert dataclasses/Path/numpy values into plain JSON-safe types."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: jsonify(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonify(v) for v in value]
    try:
        import numpy as np
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return jsonify(value.tolist())
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass
    return value


def ok_envelope(tool: str, *, warnings: Optional[Iterable[str]] = None, **payload) -> dict:
    """Build the success envelope shared by all three tools."""
    envelope = {
        "contract_version": TOOL_CONTRACT_VERSION,
        "tool": tool,
        "status": "ok",
        "warnings": _warning_records(warnings),
        "errors": [],
    }
    envelope.update(payload)
    return jsonify(envelope)


def error_envelope(tool: str, errors: Optional[Iterable[FieldError]] = None, *,
                    stage: Optional[str] = None,
                    warnings: Optional[Iterable[str]] = None, **payload) -> dict:
    """Build the failure envelope shared by all three tools.

    `stage` carries the finer distinction between failure kinds (e.g.
    "structural_validation", "geometry_validation", "safety_check", "solve").
    `status` itself is always exactly "ok" or "error", never a third value.
    """
    envelope = {
        "contract_version": TOOL_CONTRACT_VERSION,
        "tool": tool,
        "status": "error",
        "stage": stage,
        "errors": list(errors or []),
        "warnings": _warning_records(warnings),
    }
    envelope.update(payload)
    return jsonify(envelope)
