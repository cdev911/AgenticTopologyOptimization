"""Subprocess launch, cancellation, and termination primitives for TH-4."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

_SENSITIVE_ENV_SUFFIXES = (
    "_API_KEY",
    "_ACCESS_KEY",
    "_PRIVATE_KEY",
    "_AUTH_TOKEN",
    "_ACCESS_TOKEN",
    "_SESSION_TOKEN",
    "_PASSWORD",
    "_SECRET",
    "_CREDENTIALS",
)
_SENSITIVE_ENV_NAMES = {
    "API_KEY",
    "ACCESS_KEY",
    "PRIVATE_KEY",
    "AUTH_TOKEN",
    "ACCESS_TOKEN",
    "SESSION_TOKEN",
    "PASSWORD",
    "SECRET",
    "CREDENTIALS",
}


@dataclass(frozen=True)
class ProcessOutcome:
    worker_pid: int
    exit_code: int
    terminating_signal: int | None
    timed_out: bool
    cancelled: bool
    wall_time_seconds: float


def has_sensitive_credentials(environment: Mapping[str, str]) -> bool:
    """Return whether an environment contains a recognized credential name."""
    return any(
        name.upper().startswith("OPENAI_")
        or name.upper() in _SENSITIVE_ENV_NAMES
        or name.upper().endswith(_SENSITIVE_ENV_SUFFIXES)
        for name in environment
    )


def sanitized_worker_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Retain native-runtime settings while removing application credentials."""
    environment = dict(source if source is not None else os.environ)
    for name in list(environment):
        upper = name.upper()
        if (
            upper.startswith("OPENAI_")
            or upper in _SENSITIVE_ENV_NAMES
            or upper.endswith(_SENSITIVE_ENV_SUFFIXES)
        ):
            environment.pop(name, None)
    environment.pop("PYTHONINSPECT", None)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["FENITOP_SOLVER_WORKER"] = "1"
    return environment


def _terminate_process_group(process: subprocess.Popen, grace_seconds: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def launch_worker_process(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    cancel_path: Path,
    timeout_seconds: float,
    termination_grace_seconds: float,
    poll_interval_seconds: float,
    on_started: Callable[[int], None] | None = None,
) -> ProcessOutcome:
    """Launch one process group and translate timeout/cancellation to state."""
    started = time.monotonic()
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            close_fds=True,
        )
        if on_started is not None:
            on_started(process.pid)
        timed_out = False
        cancelled = False
        while process.poll() is None:
            if cancel_path.exists():
                cancelled = True
                _terminate_process_group(process, termination_grace_seconds)
                break
            if time.monotonic() - started >= timeout_seconds:
                timed_out = True
                _terminate_process_group(process, termination_grace_seconds)
                break
            time.sleep(poll_interval_seconds)
        exit_code = process.wait()
    return ProcessOutcome(
        worker_pid=process.pid,
        exit_code=exit_code,
        terminating_signal=(-exit_code if exit_code < 0 else None),
        timed_out=timed_out,
        cancelled=cancelled,
        wall_time_seconds=time.monotonic() - started,
    )
