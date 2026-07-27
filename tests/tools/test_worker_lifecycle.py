"""TH-4 filesystem, idempotency, and solver-process containment tests."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from pydantic import ValidationError

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _config():
    with (FIXTURES / "smoke_beam_2d.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


class WorkerLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="fenitop_lifecycle_"))
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def _policy(self, **updates):
        from fenitop.tools.contracts import TrustedRunPolicy

        return TrustedRunPolicy(
            output_root=self.tmp_dir,
            output_prefix="lifecycle",
            render_snapshot=False,
            **updates,
        )

    def test_success_is_isolated_secret_free_and_idempotently_replayed(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        policy = self._policy(idempotency_key="stable-request")
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "must-not-leak"}):
            first = run_topopt_tool({"config": _config()}, policy=policy)
        self.assertEqual(first["status"], "ok", first.get("error"))
        self.assertEqual(first["lifecycle"]["state"], "succeeded")
        self.assertFalse(first["lifecycle"]["worker_api_key_present"])
        self.assertFalse(first["idempotent_replay"])
        run_dir = Path(first["artifacts"][0]["path"]).parent
        worker_result = run_dir / "worker_result.json"
        first_result_mtime = worker_result.stat().st_mtime_ns

        second = run_topopt_tool({"config": _config()}, policy=policy)
        self.assertEqual(second["status"], "ok", second.get("error"))
        self.assertTrue(second["idempotent_replay"])
        self.assertEqual(second["run_id"], first["run_id"])
        self.assertEqual(worker_result.stat().st_mtime_ns, first_result_mtime)

    def test_same_idempotency_key_rejects_changed_request(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        policy = self._policy(idempotency_key="conflict")
        first = run_topopt_tool({"config": _config()}, policy=policy)
        self.assertEqual(first["status"], "ok", first.get("error"))
        changed = _config()
        changed["opt"]["initial_density"] = 0.35
        second = run_topopt_tool({"config": changed}, policy=policy)
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["stage"], "idempotency")
        self.assertEqual(second["errors"][0]["code"], "idempotency_conflict")

    def test_idempotent_replay_revalidates_artifact_containment(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        policy = self._policy(idempotency_key="replay-artifact-check")
        first = run_topopt_tool({"config": _config()}, policy=policy)
        self.assertEqual(first["status"], "ok", first.get("error"))
        artifact_path = Path(first["artifacts"][0]["path"])
        artifact_path.unlink()
        artifact_path.symlink_to(self.tmp_dir / "outside")

        replay = run_topopt_tool({"config": _config()}, policy=policy)
        self.assertEqual(replay["status"], "error")
        self.assertEqual(replay["stage"], "idempotency")
        self.assertEqual(replay["errors"][0]["code"], "stored_response_invalid")

    def test_traversal_reserved_and_flat_run_capabilities_are_unrepresentable(self):
        from fenitop.tools.contracts import TrustedRunPolicy

        for run_id in ("../escape", "a/b", "NUL", "bad*glob"):
            with self.subTest(run_id=run_id), self.assertRaises(ValidationError):
                TrustedRunPolicy(
                    output_root=self.tmp_dir,
                    run_id=run_id,
                    idempotency_key="key",
                )
        with self.assertRaises(ValidationError):
            TrustedRunPolicy(output_root=self.tmp_dir, scoped_output=False)

    def test_preexisting_symlinked_run_directory_is_rejected(self):
        from fenitop.tools.lifecycle import idempotency_hash
        from fenitop.tools.run_topopt import run_topopt_tool

        key = "symlink"
        run_id = f"lifecycle_{idempotency_hash(key)[:16]}"
        outside = self.tmp_dir / "outside"
        outside.mkdir()
        (self.tmp_dir / run_id).symlink_to(outside, target_is_directory=True)
        result = run_topopt_tool(
            {"config": _config()}, policy=self._policy(idempotency_key=key)
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "filesystem")
        self.assertEqual(result["errors"][0]["code"], "symlinked_run_directory")
        self.assertEqual(list(outside.iterdir()), [])

    def test_disk_capacity_is_checked_before_run_allocation(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        result = run_topopt_tool(
            {"config": _config()},
            policy=self._policy(min_free_disk_mb=10**12),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["errors"][0]["code"], "insufficient_disk_space")

    def test_agent_facing_execution_rejects_mpi_worlds(self):
        from fenitop.tools.run_topopt import run_topopt_tool

        with mock.patch(
            "fenitop.tools.run_topopt._agent_world_size", return_value=2
        ):
            result = run_topopt_tool(
                {"config": _config()},
                policy=self._policy(idempotency_key="mpi-rejected"),
            )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "execution_policy")
        self.assertEqual(result["errors"][0]["code"], "mpi_unsupported")
        self.assertEqual(list(self.tmp_dir.iterdir()), [])

    def test_global_serial_capacity_lock_rejects_a_second_job(self):
        from fenitop.tools.lifecycle import acquire_active_lock, release_active_lock
        from fenitop.tools.run_topopt import run_topopt_tool

        lock = acquire_active_lock(self.tmp_dir.resolve(), "already_running")
        self.addCleanup(release_active_lock, lock, "already_running")
        result = run_topopt_tool(
            {"config": _config()},
            policy=self._policy(idempotency_key="capacity"),
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "capacity")
        self.assertEqual(result["lifecycle"]["state"], "failed")
        self.assertTrue(result["errors"][0]["retryable"])

    def test_worker_request_write_failure_releases_capacity_lock(self):
        from fenitop.tools import lifecycle as lifecycle_module
        from fenitop.tools.lifecycle import ACTIVE_LOCK_NAME, WORKER_REQUEST_NAME
        from fenitop.tools.run_topopt import run_topopt_tool

        real_write = lifecycle_module.atomic_write_json

        def fail_worker_request(path, value):
            if path.name == WORKER_REQUEST_NAME:
                raise OSError("injected request write failure")
            return real_write(path, value)

        with mock.patch(
            "fenitop.tools.lifecycle.atomic_write_json",
            side_effect=fail_worker_request,
        ):
            result = run_topopt_tool(
                {"config": _config()},
                policy=self._policy(idempotency_key="request-write-fault"),
            )
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "worker_launch")
        self.assertEqual(result["lifecycle"]["state"], "failed")
        self.assertFalse((self.tmp_dir / ACTIVE_LOCK_NAME).exists())

    def test_active_duplicate_and_real_cancellation_do_not_start_twice(self):
        from fenitop.tools.lifecycle import (
            JOB_MANIFEST_NAME,
            idempotency_hash,
            read_json,
            request_cancellation,
        )
        from fenitop.tools.run_topopt import run_topopt_tool

        config = _config()
        config["mesh"]["divisions"] = [40, 20]
        config["opt"]["max_iter"] = 50
        key = "cancel-me"
        policy = self._policy(
            idempotency_key=key,
            timeout_seconds=30,
            poll_interval_seconds=0.02,
        )
        run_id = f"lifecycle_{idempotency_hash(key)[:16]}"
        result_holder = {}

        thread = threading.Thread(
            target=lambda: result_holder.setdefault(
                "result", run_topopt_tool({"config": config}, policy=policy)
            )
        )
        thread.start()
        manifest_path = self.tmp_dir / run_id / JOB_MANIFEST_NAME
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if manifest_path.is_file():
                manifest = read_json(manifest_path)
                if manifest["state"] == "running":
                    break
            time.sleep(0.02)
        else:
            self.fail("worker did not reach running state")

        duplicate = run_topopt_tool({"config": config}, policy=policy)
        self.assertEqual(duplicate["status"], "error")
        self.assertEqual(duplicate["errors"][0]["code"], "job_already_active")
        self.assertTrue(duplicate["idempotent_replay"])
        self.assertTrue(request_cancellation(self.tmp_dir, run_id))
        thread.join(timeout=15)
        self.assertFalse(thread.is_alive())
        cancelled = result_holder["result"]
        self.assertEqual(cancelled["status"], "error")
        self.assertEqual(cancelled["stage"], "cancelled")
        self.assertEqual(cancelled["lifecycle"]["state"], "cancelled")

    def test_parent_translates_timeout_and_native_crash(self):
        from fenitop.tools.worker_process import ProcessOutcome
        from fenitop.tools.run_topopt import run_topopt_tool

        outcomes = [
            (
                ProcessOutcome(123, -15, 15, True, False, 0.1),
                "timeout",
                "timed_out",
            ),
            (
                ProcessOutcome(124, -6, 6, False, False, 0.1),
                "worker_crash",
                "failed",
            ),
        ]
        for index, (outcome, stage, state) in enumerate(outcomes):
            with self.subTest(stage=stage), mock.patch(
                "fenitop.tools.worker_process.launch_worker_process",
                return_value=outcome,
            ):
                result = run_topopt_tool(
                    {"config": _config()},
                    policy=self._policy(idempotency_key=f"fault-{index}"),
                )
            self.assertEqual(result["status"], "error")
            self.assertEqual(result["stage"], stage)
            self.assertEqual(result["lifecycle"]["state"], state)
            self.assertEqual(
                result["lifecycle"]["terminating_signal"],
                outcome.terminating_signal,
            )

    def test_parent_preserves_a_typed_worker_failure(self):
        from fenitop.tools.contracts import RunTopoptResponse
        from fenitop.tools.lifecycle import atomic_write_json
        from fenitop.tools.run_topopt import run_topopt_tool
        from fenitop.tools.schema import FieldError, error_envelope
        from fenitop.tools.worker_process import ProcessOutcome
        from fenitop.tools.worker_protocol import SolverWorkerResult

        def return_typed_failure(command, **kwargs):
            request_path = Path(command[-1])
            response = error_envelope(
                "run_topopt",
                [
                    FieldError(
                        "solver.elasticity",
                        "linear_solve_diverged",
                        "Injected typed worker failure.",
                    )
                ],
                stage="numerical",
                run_id=request_path.parent.name,
            )
            atomic_write_json(
                request_path.parent / "worker_result.json",
                SolverWorkerResult(
                    worker_api_key_present=False,
                    response=RunTopoptResponse.model_validate(response),
                ).model_dump(mode="json"),
            )
            kwargs["on_started"](4321)
            return ProcessOutcome(4321, 0, None, False, False, 0.1)

        with mock.patch(
            "fenitop.tools.worker_process.launch_worker_process",
            side_effect=return_typed_failure,
        ):
            result = run_topopt_tool(
                {"config": _config()},
                policy=self._policy(idempotency_key="typed-worker-failure"),
            )

        self.assertEqual(result["status"], "error")
        self.assertEqual(result["stage"], "numerical")
        self.assertEqual(result["errors"][0]["code"], "linear_solve_diverged")
        self.assertEqual(result["lifecycle"]["state"], "failed")
        self.assertIsNone(result["run_manifest"])

    def test_parent_rejects_symlinked_and_escaped_worker_artifacts(self):
        from fenitop.tools.run_topopt import _validate_worker_artifacts

        run_dir = self.tmp_dir / "run"
        run_dir.mkdir()
        outside = self.tmp_dir / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        symlink = run_dir / "artifact.json"
        symlink.symlink_to(outside)
        for path in (symlink, outside):
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                _validate_worker_artifacts(
                    {
                        "status": "ok",
                        "artifacts": [{"path": str(path)}],
                    },
                    run_dir.resolve(),
                )

    def test_process_group_launcher_handles_timeout_cancel_and_signal(self):
        from fenitop.tools.worker_process import (
            has_sensitive_credentials,
            launch_worker_process,
            sanitized_worker_environment,
        )

        environment = sanitized_worker_environment(
            {
                "PATH": os.environ["PATH"],
                "API_KEY": "one",
                "AWS_SECRET_ACCESS_KEY": "two",
                "SERVICE_SESSION_TOKEN": "three",
            }
        )
        self.assertFalse(has_sensitive_credentials(environment))
        self.assertEqual(environment["PATH"], os.environ["PATH"])

        def launch(code, name, *, timeout=2, cancelled=False):
            cancel_path = self.tmp_dir / f"{name}.cancel"
            if cancelled:
                cancel_path.write_text("cancel", encoding="utf-8")
            return launch_worker_process(
                [sys.executable, "-c", code],
                cwd=REPO_ROOT,
                environment=sanitized_worker_environment(),
                stdout_path=self.tmp_dir / f"{name}.stdout",
                stderr_path=self.tmp_dir / f"{name}.stderr",
                cancel_path=cancel_path,
                timeout_seconds=timeout,
                termination_grace_seconds=0.1,
                poll_interval_seconds=0.01,
            )

        timed_out = launch("import time; time.sleep(10)", "timeout", timeout=0.05)
        self.assertTrue(timed_out.timed_out)
        cancelled = launch(
            "import time; time.sleep(10)", "cancel", cancelled=True
        )
        self.assertTrue(cancelled.cancelled)
        crashed = launch(
            "import os, signal; os.kill(os.getpid(), signal.SIGABRT)", "crash"
        )
        self.assertEqual(crashed.terminating_signal, 6)

    def test_restart_recovery_marks_stale_running_job_orphaned(self):
        from fenitop.tools.lifecycle import (
            ACTIVE_LOCK_NAME,
            allocate_run_directory,
            new_lifecycle,
            read_lifecycle,
            recover_orphaned_jobs,
            update_lifecycle,
            write_lifecycle,
        )

        root = self.tmp_dir.resolve()
        run_dir, _ = allocate_run_directory(root, "stale")
        lifecycle = write_lifecycle(
            run_dir,
            new_lifecycle(
                run_id="stale",
                request_hash="a" * 64,
                idempotency_key_hash=None,
            ),
        )
        update_lifecycle(
            run_dir,
            lifecycle,
            state="running",
            parent_pid=999_999_999,
            worker_pid=999_999_998,
        )
        (root / ACTIVE_LOCK_NAME).write_text(
            json.dumps({"run_id": "stale", "parent_pid": 999_999_999}),
            encoding="utf-8",
        )
        self.assertEqual(recover_orphaned_jobs(root), ["stale"])
        self.assertEqual(read_lifecycle(run_dir)["state"], "orphaned")
        self.assertFalse((root / ACTIVE_LOCK_NAME).exists())

    def test_recovery_never_reclaims_a_fresh_partially_written_lock(self):
        from fenitop.tools import lifecycle as lifecycle_module
        from fenitop.tools.lifecycle import ACTIVE_LOCK_NAME, recover_orphaned_jobs

        lock_path = self.tmp_dir / ACTIVE_LOCK_NAME
        lock_path.write_text("", encoding="utf-8")
        recover_orphaned_jobs(self.tmp_dir.resolve())
        self.assertTrue(lock_path.exists())

        stale_time = time.time() - (
            lifecycle_module._INCOMPLETE_LOCK_STALE_SECONDS + 1
        )
        os.utime(lock_path, (stale_time, stale_time))
        recover_orphaned_jobs(self.tmp_dir.resolve())
        self.assertFalse(lock_path.exists())


if __name__ == "__main__":
    unittest.main()
