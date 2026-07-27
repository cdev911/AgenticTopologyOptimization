"""Trusted filesystem and lifecycle primitives for isolated solver jobs."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fenitop.tools.contracts import JobLifecycleRecord

JOB_MANIFEST_NAME = "job_manifest.json"
RESPONSE_NAME = "response.json"
WORKER_REQUEST_NAME = "worker_request.json"
WORKER_RESULT_NAME = "worker_result.json"
CANCEL_NAME = "cancel.request"
ACTIVE_LOCK_NAME = ".fenitop-active.lock"
_INCOMPLETE_LOCK_STALE_SECONDS = 30.0

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class LifecycleError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_identifier(value: str, *, field: str) -> str:
    if not _IDENTIFIER_RE.fullmatch(value):
        raise LifecycleError(
            "invalid_identifier",
            f"{field} must contain only 1-80 ASCII letters, digits, '_' or '-'.",
        )
    if value.upper() in _RESERVED:
        raise LifecycleError(
            "reserved_identifier", f"{field} is a reserved filesystem name."
        )
    return value


def canonical_json_hash(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def idempotency_hash(value: str | None) -> str | None:
    return hashlib.sha256(value.encode("utf-8")).hexdigest() if value else None


def resolve_output_root(root: Path) -> Path:
    """Create the trusted root and resolve it once before deriving child paths."""
    root = root.expanduser()
    root.mkdir(parents=True, exist_ok=True)
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise LifecycleError("invalid_output_root", "Output root is not a directory.")
    return resolved


def allocate_run_directory(root: Path, run_id: str) -> tuple[Path, bool]:
    """Atomically allocate a fresh run directory.

    Returns ``(path, created)``. Existing real directories are idempotency
    candidates; symlinks and non-directories are always rejected.
    """
    validate_identifier(run_id, field="run_id")
    candidate = root / run_id
    if candidate.parent != root:
        raise LifecycleError("run_path_escape", "Run path escaped the output root.")
    try:
        candidate.mkdir(mode=0o700)
        created = True
    except FileExistsError:
        created = False
    if candidate.is_symlink():
        raise LifecycleError(
            "symlinked_run_directory", "A run directory may not be a symlink."
        )
    if not candidate.is_dir():
        raise LifecycleError(
            "invalid_run_directory", "Existing run path is not a directory."
        )
    if candidate.resolve(strict=True).parent != root:
        raise LifecycleError("run_path_escape", "Run directory escaped the output root.")
    return candidate, created


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably replace a JSON file without exposing a partial document."""
    if path.parent.is_symlink():
        raise LifecycleError(
            "symlinked_run_directory", "Refusing to write through a symlinked directory."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path, *, max_bytes: int = 4 * 1024 * 1024) -> dict[str, Any]:
    if path.is_symlink():
        raise LifecycleError("symlinked_job_file", f"{path.name} may not be a symlink.")
    if not path.is_file():
        raise LifecycleError("job_file_missing", f"{path.name} does not exist.")
    if path.stat().st_size > max_bytes:
        raise LifecycleError("job_file_too_large", f"{path.name} is unexpectedly large.")
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise LifecycleError("invalid_job_file", f"{path.name} must contain an object.")
    return value


def new_lifecycle(
    *,
    run_id: str,
    request_hash: str,
    idempotency_key_hash: str | None,
) -> dict[str, Any]:
    now = utc_now()
    return JobLifecycleRecord(
        state="queued",
        run_id=run_id,
        request_hash=request_hash,
        idempotency_key_hash=idempotency_key_hash,
        created_at=now,
        updated_at=now,
        parent_pid=os.getpid(),
    ).model_dump(mode="json")


def write_lifecycle(run_dir: Path, lifecycle: dict[str, Any]) -> dict[str, Any]:
    validated = JobLifecycleRecord.model_validate(lifecycle).model_dump(mode="json")
    atomic_write_json(run_dir / JOB_MANIFEST_NAME, validated)
    return validated


def update_lifecycle(
    run_dir: Path, lifecycle: dict[str, Any], **updates: Any
) -> dict[str, Any]:
    candidate = {**lifecycle, **updates, "updated_at": utc_now()}
    return write_lifecycle(run_dir, candidate)


def read_lifecycle(run_dir: Path) -> dict[str, Any]:
    value = read_json(run_dir / JOB_MANIFEST_NAME)
    return JobLifecycleRecord.model_validate(value).model_dump(mode="json")


def check_disk_capacity(
    root: Path, *, estimated_output_mb: float, minimum_free_mb: float
) -> None:
    free_mb = shutil.disk_usage(root).free / (1024**2)
    required_mb = minimum_free_mb + 1.25 * estimated_output_mb
    if free_mb < required_mb:
        raise LifecycleError(
            "insufficient_disk_space",
            f"Free disk {free_mb:.1f} MiB is below required {required_mb:.1f} MiB.",
        )


def _pid_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_active_lock(root: Path, run_id: str) -> Path:
    lock_path = root / ACTIVE_LOCK_NAME
    payload = {
        "run_id": run_id,
        "parent_pid": os.getpid(),
        "created_at": utc_now(),
    }
    try:
        descriptor = os.open(
            lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError as exc:
        raise LifecycleError(
            "solve_capacity_reached",
            "The serial demo already has an active solver job.",
        ) from exc
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    return lock_path


def release_active_lock(lock_path: Path, run_id: str) -> None:
    try:
        payload = read_json(lock_path)
    except (LifecycleError, json.JSONDecodeError, OSError):
        return
    if payload.get("run_id") == run_id:
        lock_path.unlink(missing_ok=True)


def recover_orphaned_jobs(root: Path) -> list[str]:
    """Mark active manifests whose parent no longer exists as orphaned."""
    recovered: list[str] = []
    for child in root.iterdir():
        if child.name.startswith(".") or child.is_symlink() or not child.is_dir():
            continue
        manifest_path = child / JOB_MANIFEST_NAME
        if not manifest_path.is_file() or manifest_path.is_symlink():
            continue
        try:
            lifecycle = read_lifecycle(child)
        except (LifecycleError, json.JSONDecodeError, ValueError, OSError):
            continue
        if lifecycle["state"] not in {"queued", "running"}:
            continue
        if _pid_alive(lifecycle.get("parent_pid")):
            continue
        lifecycle = update_lifecycle(
            child,
            lifecycle,
            state="orphaned",
            message="Recovered stale active job after its parent process exited.",
        )
        recovered.append(lifecycle["run_id"])

    lock_path = root / ACTIVE_LOCK_NAME
    if lock_path.exists() and not lock_path.is_symlink():
        try:
            lock = read_json(lock_path)
        except (LifecycleError, json.JSONDecodeError, OSError):
            # An exclusive lock is visible just before its JSON payload is fully
            # written. Never delete a fresh malformed lock and open a race for a
            # second solve; only reclaim one that has remained incomplete.
            try:
                age_seconds = time.time() - lock_path.stat().st_mtime
            except OSError:
                pass
            else:
                if age_seconds >= _INCOMPLETE_LOCK_STALE_SECONDS:
                    lock_path.unlink(missing_ok=True)
        else:
            if not _pid_alive(lock.get("parent_pid")):
                lock_path.unlink(missing_ok=True)
    return recovered


def request_cancellation(output_root: Path, run_id: str) -> bool:
    """Request cancellation without accepting an arbitrary filesystem path."""
    root = resolve_output_root(output_root)
    validate_identifier(run_id, field="run_id")
    run_dir = root / run_id
    if run_dir.is_symlink() or not run_dir.is_dir():
        return False
    lifecycle = read_lifecycle(run_dir)
    if lifecycle["state"] not in {"queued", "running"}:
        return False
    cancel_path = run_dir / CANCEL_NAME
    try:
        descriptor = os.open(
            cancel_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        return True
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(utc_now())
        handle.flush()
        os.fsync(handle.fileno())
    return True
