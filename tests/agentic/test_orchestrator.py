from __future__ import annotations

import unittest

from agentic.compiler import compile_intent
from agentic.intent import InterpretationEnvelope
from agentic.orchestrator import (
    AnalysisFailedWorkflow,
    AwaitingClarification,
    CompletedWorkflow,
    DeterministicOrchestrator,
)
from fenitop.tools.contracts import (
    AnalyzeResultsRequest,
    AnalyzeResultsResponse,
    JobLifecycleRecord,
    OptimizerStatusRecord,
    RunManifest,
    RunMetrics,
    RunTopoptRequest,
    RunTopoptResponse,
    ValidateConfigRequest,
)
from fenitop.tools.schema import TOOL_CONTRACT_VERSION


def ready_result():
    return InterpretationEnvelope.model_validate(
        {
            "result": {
                "status": "ready",
                "intent": {
                    "problem_type": "minimize_compliance",
                    "domain": {"bounds": [[0, 0], [10, 10]]},
                    "material": {
                        "young_modulus": 100,
                        "poisson_ratio": 0.3,
                    },
                    "supports": [
                        {
                            "region": {
                                "op": "plane",
                                "axis": "x",
                                "value": 0,
                            }
                        }
                    ],
                    "tractions": [
                        {
                            "region": {
                                "op": "plane",
                                "axis": "x",
                                "value": 10,
                            },
                            "vector": [0, -1],
                        }
                    ],
                    "volume_fraction": 0.4,
                },
            }
        }
    ).result


def clarification_result():
    return InterpretationEnvelope.model_validate(
        {
            "result": {
                "status": "needs_clarification",
                "missing_fields": ["material.young_modulus"],
                "questions": ["What Young's modulus should be used?"],
            }
        }
    ).result


def unsupported_result():
    return InterpretationEnvelope.model_validate(
        {
            "result": {
                "status": "unsupported",
                "unsupported_features": ["3D domain"],
                "explanation": "v1 supports rectangular 2D domains.",
            }
        }
    ).result


def validation_response(status="ok", *, config=None):
    error = {
        "code": "invalid_geometry",
        "path": "config.mesh",
        "message": "Geometry is invalid.",
        "severity": "error",
        "retryable": False,
    }
    return {
        "contract_version": TOOL_CONTRACT_VERSION,
        "tool": "validate_config",
        "status": status,
        "warnings": [],
        "errors": [error] if status == "error" else [],
        "stage": "geometry" if status == "error" else None,
        "checked": {
            "structural": True,
            "resource": True,
            "geometry": status == "ok",
        },
        "problem_type": "minimize_compliance",
        "normalized_config": config if status == "ok" else None,
        "estimated_cost": None,
        "geometry_report": None,
    }


class FakeInterpreter:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    def interpret(self, request):
        self.requests.append(request)
        return self.results.pop(0)


class RecordingValidator:
    def __init__(self, status="ok", timeline=None):
        self.status = status
        self.requests = []
        self.timeline = timeline

    def __call__(self, request):
        self.requests.append(request)
        if self.timeline is not None:
            self.timeline.append("validator")
        return validation_response(self.status, config=request.config)


class RecordingRunner:
    def __init__(self, *, status="ok"):
        self.status = status
        self.calls = []
        self.response = None

    def __call__(self, request, *, policy):
        self.calls.append((request, policy))
        request_hash = "1" * 64
        run_id = "agentic_test_run"
        lifecycle = JobLifecycleRecord.model_construct(
            state="succeeded",
            run_id=run_id,
            request_hash=request_hash,
        )
        metrics = RunMetrics.model_construct(
            final_compliance=1.0,
            final_volume=0.4,
            final_objective=0.0,
            grayness=0.1,
            binarization_score=0.9,
            final_change=0.001,
            opt_tol=1e-5,
            final_beta=128.0,
            continuation_completed=True,
        )
        optimizer = OptimizerStatusRecord.model_construct(
            method="oc",
            converged=True,
            outer_iterations=1,
        )
        manifest = RunManifest.model_construct(
            run_id=run_id,
            request_hash=request_hash,
            lifecycle=lifecycle,
            normalized_config=request.config,
            problem_type=request.config.opt.problem_type,
            metrics=metrics,
            optimizer_status=optimizer,
        )
        self.response = RunTopoptResponse.model_construct(
            status=self.status,
            message="run failed" if self.status == "error" else None,
            run_manifest=manifest if self.status == "ok" else None,
        )
        return self.response


class RecordingAnalyzer:
    def __init__(self, statuses=("ok",)):
        self.statuses = list(statuses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        status = self.statuses.pop(0)
        return AnalyzeResultsResponse.model_construct(
            status=status,
            message="analysis failed" if status == "error" else None,
        )


class OrchestratorTests(unittest.TestCase):
    def test_ready_compiles_emits_notice_then_validates_exact_typed_request(self):
        timeline = []
        validator = RecordingValidator(timeline=timeline)
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=validator,
            event_callback=lambda event: timeline.append(event.stage),
        )

        outcome = orchestrator.start("Optimize my square beam")

        self.assertEqual(outcome.status, "validated")
        self.assertEqual(
            timeline,
            ["interpreted", "defaults_applied", "validator", "validated"],
        )
        self.assertEqual(len(validator.requests), 1)
        self.assertIsInstance(validator.requests[0], ValidateConfigRequest)
        self.assertIs(
            validator.requests[0].config,
            outcome.compilation.config,
        )
        self.assertIn(
            "values were not provided",
            outcome.events[1].message,
        )

    def test_clarification_does_not_compile_or_validate_and_can_resume(self):
        interpreter = FakeInterpreter([clarification_result(), ready_result()])
        validator = RecordingValidator()
        compile_calls = []

        def compiler(intent):
            compile_calls.append(intent)
            return compile_intent(intent)

        orchestrator = DeterministicOrchestrator(
            interpreter,
            compiler=compiler,
            validator=validator,
        )
        waiting = orchestrator.start("Optimize a beam")

        self.assertIsInstance(waiting, AwaitingClarification)
        self.assertEqual(compile_calls, [])
        self.assertEqual(validator.requests, [])

        outcome = orchestrator.resume(waiting, "Use E=100 and nu=0.3")

        self.assertEqual(outcome.status, "validated")
        self.assertEqual(len(compile_calls), 1)
        self.assertEqual(len(validator.requests), 1)
        self.assertIn("Original request:\nOptimize a beam", interpreter.requests[1])
        self.assertIn(
            "Questions:\n- What Young's modulus should be used?",
            interpreter.requests[1],
        )
        self.assertIn(
            "User answer: Use E=100 and nu=0.3",
            interpreter.requests[1],
        )
        self.assertEqual(
            [event.sequence for event in outcome.events],
            list(range(1, len(outcome.events) + 1)),
        )

    def test_unsupported_is_terminal_without_compilation_or_validation(self):
        validator = RecordingValidator()
        compile_calls = []
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([unsupported_result()]),
            compiler=lambda intent: compile_calls.append(intent),
            validator=validator,
        )

        outcome = orchestrator.start("Optimize a 3D bracket")

        self.assertEqual(outcome.status, "unsupported")
        self.assertEqual(outcome.unsupported_features, ("3D domain",))
        self.assertEqual(compile_calls, [])
        self.assertEqual(validator.requests, [])

    def test_validation_error_is_a_typed_terminal_state(self):
        validator = RecordingValidator(status="error")
        outcome = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=validator,
        ).start("Optimize a beam")

        self.assertEqual(outcome.status, "validation_failed")
        self.assertEqual(outcome.validation.status, "error")
        self.assertEqual(outcome.validation.errors[0].code, "invalid_geometry")
        self.assertEqual(outcome.events[-1].stage, "validation_failed")

    def test_blank_start_or_resume_fails_before_interpretation(self):
        interpreter = FakeInterpreter([clarification_result()])
        orchestrator = DeterministicOrchestrator(interpreter)

        with self.assertRaises(ValueError):
            orchestrator.start(" ")
        waiting = orchestrator.start("request")
        with self.assertRaises(ValueError):
            orchestrator.resume(waiting, "\n")

        self.assertEqual(len(interpreter.requests), 1)

    def test_execute_passes_exact_typed_config_and_manifest_then_completes(self):
        runner = RecordingRunner()
        analyzer = RecordingAnalyzer()
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
            runner=runner,
            analyzer=analyzer,
        )
        validated = orchestrator.start("Optimize a beam")

        outcome = orchestrator.execute(validated)

        self.assertIsInstance(outcome, CompletedWorkflow)
        self.assertEqual(len(runner.calls), 1)
        run_request, policy = runner.calls[0]
        self.assertIsInstance(run_request, RunTopoptRequest)
        self.assertIs(run_request.config, validated.compilation.config)
        self.assertEqual(policy.idempotency_key, outcome.idempotency_key)
        self.assertTrue(
            outcome.idempotency_key.startswith("agentic-workflow-v1:")
        )
        self.assertEqual(len(analyzer.requests), 1)
        self.assertIsInstance(analyzer.requests[0], AnalyzeResultsRequest)
        self.assertIs(
            analyzer.requests[0].run_manifest,
            runner.response.run_manifest,
        )
        self.assertEqual(outcome.events[-2].stage, "run_succeeded")
        self.assertEqual(outcome.events[-1].stage, "completed")

    def test_completed_refresh_and_reexecuted_validated_state_do_not_rerun(self):
        runner = RecordingRunner()
        analyzer = RecordingAnalyzer(statuses=("ok", "ok"))
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
            runner=runner,
            analyzer=analyzer,
        )
        validated = orchestrator.start("same request")
        completed = orchestrator.execute(validated)

        refreshed = orchestrator.execute(completed)
        replayed_from_validated = orchestrator.execute(validated)

        self.assertIs(refreshed, completed)
        self.assertEqual(replayed_from_validated.status, "completed")
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(analyzer.requests), 2)

    def test_analysis_retry_uses_stored_manifest_without_repeating_run(self):
        runner = RecordingRunner()
        analyzer = RecordingAnalyzer(statuses=("error", "ok"))
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
            runner=runner,
            analyzer=analyzer,
        )
        validated = orchestrator.start("request")

        failed = orchestrator.execute(validated)
        completed = orchestrator.execute(failed)

        self.assertIsInstance(failed, AnalysisFailedWorkflow)
        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(len(analyzer.requests), 2)
        self.assertIs(
            analyzer.requests[0].run_manifest,
            analyzer.requests[1].run_manifest,
        )

    def test_run_error_stops_before_analysis(self):
        runner = RecordingRunner(status="error")
        analyzer = RecordingAnalyzer()
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
            runner=runner,
            analyzer=analyzer,
        )

        outcome = orchestrator.execute(orchestrator.start("request"))

        self.assertEqual(outcome.status, "run_failed")
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(analyzer.requests, [])
        self.assertEqual(outcome.events[-1].stage, "run_failed")

    def test_idempotency_key_is_stable_across_orchestrator_instances(self):
        keys = []
        for _ in range(2):
            runner = RecordingRunner()
            orchestrator = DeterministicOrchestrator(
                FakeInterpreter([ready_result()]),
                validator=RecordingValidator(),
                runner=runner,
                analyzer=RecordingAnalyzer(),
            )
            orchestrator.execute(orchestrator.start("stable request"))
            keys.append(runner.calls[0][1].idempotency_key)

        self.assertEqual(keys[0], keys[1])


if __name__ == "__main__":
    unittest.main()
