"""Build and verify self-contained successful-run manifests."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any, Iterable

from fenitop.tools.config_models import CONFIG_SCHEMA_VERSION
from fenitop.tools.contracts import (
    ManifestArtifactRecord,
    RunManifest,
    RuntimeVersionRecord,
)
from fenitop.tools.lifecycle import atomic_write_json, canonical_json_hash
from fenitop.tools.schema import TOOL_CONTRACT_VERSION

RUN_MANIFEST_NAME = "run_manifest.json"
_HASH_CHUNK_BYTES = 1024 * 1024
_RUNTIME_PACKAGES = (
    "fenitop",
    "numpy",
    "scipy",
    "pydantic",
    "mcp",
    "mpi4py",
    "petsc4py",
    "fenics-dolfinx",
)


class ManifestError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_versions() -> list[RuntimeVersionRecord]:
    records = [
        RuntimeVersionRecord(name="python", version=platform.python_version()),
        RuntimeVersionRecord(name="platform", version=platform.platform()),
    ]
    for package in _RUNTIME_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
        records.append(RuntimeVersionRecord(name=package, version=version))
    return records


def inventory_artifacts(
    run_dir: Path,
    artifacts: Iterable[dict[str, Any]],
) -> list[ManifestArtifactRecord]:
    """Verify complete files and convert their paths to checksummed relative records."""
    run_dir = run_dir.resolve(strict=True)
    records: list[ManifestArtifactRecord] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    for artifact in artifacts:
        role = artifact["role"]
        if role in seen_roles:
            raise ManifestError("duplicate_artifact_role", f"Duplicate artifact role: {role}")
        source = Path(artifact["path"])
        if source.is_symlink():
            raise ManifestError("symlinked_artifact", f"Artifact {role} is a symlink.")
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ManifestError(
                "artifact_missing", f"Artifact {role} does not exist: {source}"
            ) from exc
        if not resolved.is_relative_to(run_dir) or not resolved.is_file():
            raise ManifestError(
                "artifact_outside_run", f"Artifact {role} is outside the run directory."
            )
        relative = resolved.relative_to(run_dir).as_posix()
        if relative in seen_paths:
            raise ManifestError(
                "duplicate_artifact_path", f"Duplicate artifact path: {relative}"
            )
        if artifact.get("complete") is not True:
            raise ManifestError("artifact_incomplete", f"Artifact {role} is incomplete.")
        records.append(
            ManifestArtifactRecord(
                role=role,
                format=artifact["format"],
                path=relative,
                size_bytes=resolved.stat().st_size,
                sha256=sha256_file(resolved),
            )
        )
        seen_roles.add(role)
        seen_paths.add(relative)
    return records


def _manifest_hash(payload: dict[str, Any]) -> str:
    material = {key: value for key, value in payload.items() if key != "manifest_hash"}
    return canonical_json_hash(material)


def build_run_manifest(
    *,
    run_dir: Path,
    output_prefix: str,
    request_hash: str,
    response: dict[str, Any],
    lifecycle: dict[str, Any],
    artifacts: Iterable[dict[str, Any]],
) -> RunManifest:
    validation = response["validation"]
    normalized_config = validation["normalized_config"]
    payload = {
        "manifest_version": "1.0",
        "manifest_hash": "0" * 64,
        "contract_version": TOOL_CONTRACT_VERSION,
        "config_schema_version": CONFIG_SCHEMA_VERSION,
        "run_id": response["run_id"],
        "run_directory": str(run_dir.resolve(strict=True)),
        "output_prefix": output_prefix,
        "request_hash": request_hash,
        "config_hash": canonical_json_hash(normalized_config),
        "problem_type": response["problem_type"],
        "normalized_config": normalized_config,
        "lifecycle": lifecycle,
        "numerical_status": "succeeded",
        "converged": response["converged"],
        "stop_reason": response["stop_reason"],
        "iterations": response["iterations"],
        "metrics": response["metrics"],
        "optimizer_status": response["optimizer_status"],
        "mma_inner_iteration_warnings": response["mma_inner_iteration_warnings"],
        "estimated_cost": validation["estimated_cost"],
        "geometry_report": validation["geometry_report"],
        "warnings": response["warnings"],
        "runtime_versions": [
            record.model_dump(mode="json") for record in runtime_versions()
        ],
        "artifacts": [
            record.model_dump(mode="json")
            for record in inventory_artifacts(run_dir, artifacts)
        ],
    }
    payload["manifest_hash"] = _manifest_hash(payload)
    return RunManifest.model_validate(payload)


def write_run_manifest(run_dir: Path, manifest: RunManifest) -> Path:
    path = run_dir / RUN_MANIFEST_NAME
    atomic_write_json(path, manifest.model_dump(mode="json"))
    return path


def verify_manifest_hash(manifest: RunManifest) -> None:
    payload = manifest.model_dump(mode="json")
    observed = _manifest_hash(payload)
    if observed != manifest.manifest_hash:
        raise ManifestError(
            "manifest_hash_mismatch",
            "RunManifest content does not match its canonical hash.",
        )


def verify_durable_manifest(
    manifest: RunManifest,
    *,
    expected_run_dir: Path | None = None,
) -> Path:
    """Verify the canonical hash and exact durable run_manifest.json copy."""
    verify_manifest_hash(manifest)
    run_dir = Path(manifest.run_directory)
    if run_dir.is_symlink():
        raise ManifestError("symlinked_run_directory", "Run directory is a symlink.")
    run_dir = run_dir.resolve(strict=True)
    if (
        expected_run_dir is not None
        and run_dir != expected_run_dir.resolve(strict=True)
    ):
        raise ManifestError(
            "run_manifest_directory_mismatch",
            "RunManifest directory does not match its owning job directory.",
        )
    manifest_path = run_dir / RUN_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ManifestError(
            "run_manifest_missing",
            "The durable run_manifest.json file is missing or symlinked.",
        )
    if manifest_path.stat().st_size > 4 * 1024 * 1024:
        raise ManifestError("run_manifest_too_large", "run_manifest.json is too large.")
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            durable_manifest = RunManifest.model_validate(json.load(handle))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ManifestError(
            "run_manifest_invalid", f"Cannot validate durable run_manifest.json: {exc}"
        ) from exc
    if durable_manifest.model_dump(mode="json") != manifest.model_dump(mode="json"):
        raise ManifestError(
            "run_manifest_mismatch",
            "Provided RunManifest differs from durable run_manifest.json.",
        )
    return run_dir


def verify_manifest_artifacts(
    manifest: RunManifest,
    allowed_roots: tuple[Path, ...],
    *,
    max_total_bytes: int = 2 * 1024**3,
) -> dict[str, Path]:
    """Resolve and checksum every listed artifact beneath a trusted run directory."""
    roots = [root.resolve(strict=True) for root in allowed_roots]
    if not roots:
        raise ManifestError(
            "no_trusted_artifact_root",
            "Application policy did not configure an allowed artifact root.",
        )
    run_dir = verify_durable_manifest(manifest)
    if not any(run_dir.is_relative_to(root) for root in roots):
        raise ManifestError(
            "run_outside_trusted_root",
            "Run directory is outside the application-owned analysis roots.",
        )
    total_bytes = 0
    paths: dict[str, Path] = {}
    for record in manifest.artifacts:
        candidate = run_dir / record.path
        if candidate.is_symlink():
            raise ManifestError(
                "symlinked_artifact", f"Artifact {record.role} is a symlink."
            )
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise ManifestError(
                "artifact_missing", f"Artifact {record.role} is missing."
            ) from exc
        if not resolved.is_relative_to(run_dir) or not resolved.is_file():
            raise ManifestError(
                "artifact_outside_run",
                f"Artifact {record.role} escaped the run directory.",
            )
        size = resolved.stat().st_size
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise ManifestError(
                "artifact_inventory_too_large",
                "Manifest artifact inventory exceeds the analysis byte limit.",
            )
        if size != record.size_bytes:
            raise ManifestError(
                "artifact_size_mismatch",
                f"Artifact {record.role} size does not match the manifest.",
            )
        if sha256_file(resolved) != record.sha256:
            raise ManifestError(
                "artifact_checksum_mismatch",
                f"Artifact {record.role} checksum does not match the manifest.",
            )
        if record.role in paths:
            raise ManifestError(
                "duplicate_artifact_role",
                f"Duplicate artifact role: {record.role}",
            )
        paths[record.role] = resolved
    return paths
