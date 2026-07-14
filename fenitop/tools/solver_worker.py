"""Dedicated serial solver-worker entry point used by Tool 2's parent."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from fenitop.tools.contracts import RunTopoptResponse, TrustedRunPolicy
from fenitop.tools.lifecycle import (
    WORKER_REQUEST_NAME,
    WORKER_RESULT_NAME,
    atomic_write_json,
    canonical_json_hash,
    read_json,
)
from fenitop.tools.schema import FieldError, error_envelope
from fenitop.tools.worker_process import has_sensitive_credentials
from fenitop.tools.worker_protocol import SolverWorkerRequest, SolverWorkerResult


def _credential_present() -> bool:
    return has_sensitive_credentials(os.environ)


def _error_response(code: str, message: str, run_id: str | None = None) -> dict:
    response = error_envelope(
        "run_topopt",
        [FieldError("worker", code, message)],
        stage="worker",
        run_id=run_id,
    )
    RunTopoptResponse.model_validate(response)
    return response


def execute_request(request_path: Path) -> int:
    if request_path.name != WORKER_REQUEST_NAME or request_path.is_symlink():
        return 2
    request_path = request_path.resolve(strict=True)
    run_dir = request_path.parent
    if run_dir.is_symlink():
        return 2
    result_path = run_dir / WORKER_RESULT_NAME

    parsed = SolverWorkerRequest.model_validate(read_json(request_path))
    if parsed.output_dir.resolve(strict=True) != run_dir:
        return 2
    expected_hash = canonical_json_hash({
        "contract_version": parsed.validation.contract_version,
        "config": parsed.validation.normalized_config.model_dump(mode="json"),
        "solver_profile": parsed.solver_profile,
        "output_interval": parsed.output_interval,
        "render_snapshot": parsed.render_snapshot,
    })
    if expected_hash != parsed.request_hash:
        response = _error_response(
            "worker_request_hash_mismatch",
            "Worker request does not match its parent-computed hash.",
            parsed.run_id,
        )
        atomic_write_json(
            result_path,
            SolverWorkerResult(
                worker_api_key_present=_credential_present(),
                response=response,
            ).model_dump(mode="json"),
        )
        return 0

    credential_present = _credential_present()
    if credential_present:
        response = _error_response(
            "worker_secret_present",
            "Solver worker refused to run because an API credential was inherited.",
            parsed.run_id,
        )
    else:
        from fenitop.tools.run_topopt import _run_topopt_in_process

        response = _run_topopt_in_process(
            {"config": parsed.config},
            policy=TrustedRunPolicy(
                output_root=run_dir.parent,
                run_id=parsed.run_id,
                idempotency_key=parsed.request_hash,
                output_prefix=parsed.output_prefix,
                render_snapshot=parsed.render_snapshot,
                solver_profile=parsed.solver_profile,
                output_interval=parsed.output_interval,
            ),
        )

    result = SolverWorkerResult(
        worker_api_key_present=credential_present,
        response=response,
    )
    atomic_write_json(result_path, result.model_dump(mode="json"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Internal fenitop solver worker.")
    parser.add_argument("--request", required=True)
    args = parser.parse_args()
    try:
        return execute_request(Path(args.request))
    except Exception:
        # The parent owns crash translation and retains stderr for local debugging.
        raise


if __name__ == "__main__":
    raise SystemExit(main())
