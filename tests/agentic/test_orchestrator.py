from __future__ import annotations

import unittest
from unittest import mock

from agentic.compiler import compile_intent
from agentic.explainer import (
    EvidenceLedger,
    ExplanationPlan,
    ExplanationResult,
)
from agentic.formulation import (
    ConversationFormulator,
    ConversationMessage,
    DraftUpdate,
    FormulationSession,
    FormulationTurn,
)
from agentic.intent import InterpretationEnvelope
from agentic.orchestrator import (
    AnalysisFailedWorkflow,
    AwaitingClarification,
    AwaitingRunApproval,
    CompletedWorkflow,
    ConversationContext,
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


class CannedFormulationAgent:
    def __init__(self, turns):
        self.turns = list(turns)

    def formulate(self, _request):
        return self.turns.pop(0)


def ready_formulation_step():
    user = (
        "Minimize compliance on [[0,0],[10,10]] with E 100, nu 0.3, "
        "the left edge fixed, traction [0,-1] on the right, and 40% material."
    )

    def update(path, value):
        return DraftUpdate(
            path=path,
            value=value,
            basis="explicit",
            source_quote=user,
            rationale="Directly stated problem fact.",
        )

    turn = FormulationTurn(
        assistant_message="The problem is complete and ready for review.",
        updates=(
            update("problem_type", "minimize_compliance"),
            update("domain.bounds", [[0, 0], [10, 10]]),
            update("material.young_modulus", 100),
            update("material.poisson_ratio", 0.3),
            update("support_edges", ["left"]),
            update(
                "tractions",
                [
                    {
                        "edge_segment": {
                            "edge": "right",
                            "center_fraction": 0.5,
                            "span_fraction": 1.0,
                        },
                        "vector": [0, -1],
                    }
                ],
            ),
            update("volume_fraction", 0.4),
        ),
        declared_state="ready",
    )
    return ConversationFormulator(
        CannedFormulationAgent([turn])
    ).start(user)


def approved(orchestrator, request="request"):
    awaiting = orchestrator.start(request)
    if not isinstance(awaiting, AwaitingRunApproval):
        raise AssertionError(f"expected approval state, got {awaiting.status}")
    return orchestrator.approve(awaiting)


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


class RecordingExplainer:
    def __init__(self):
        self.analyses = []

    def explain(self, analysis):
        self.analyses.append(analysis)
        return ExplanationResult(
            evidence=EvidenceLedger(facts=()),
            plan=ExplanationPlan(
                sections=(
                    {"heading": "Outcome", "fact_ids": ("F001",)},
                )
            ),
            markdown="# Result explanation",
        )


class OrchestratorTests(unittest.TestCase):
    def test_ready_formulation_is_the_typed_compile_validate_entry(self):
        timeline = []
        validator = RecordingValidator(timeline=timeline)
        orchestrator = DeterministicOrchestrator(
            validator=validator,
            event_callback=lambda event: timeline.append(event.stage),
        )
        step = ready_formulation_step()

        outcome = orchestrator.prepare_formulation(step)

        self.assertIsInstance(outcome, AwaitingRunApproval)
        self.assertEqual(
            timeline,
            [
                "formulated",
                "defaults_applied",
                "validator",
                "validated",
                "approval_requested",
            ],
        )
        self.assertEqual(
            outcome.conversation.original_request,
            step.session.messages[0].content,
        )
        self.assertEqual(len(validator.requests), 1)

    def test_nonready_formulation_cannot_compile_or_validate(self):
        validator = RecordingValidator()
        gathering = ConversationFormulator(
            CannedFormulationAgent(
                [
                    FormulationTurn(
                        assistant_message="I still need the material and load.",
                        questions=("What material and load should be used?",),
                    )
                ]
            )
        ).start("Make a rectangular cantilever.")
        orchestrator = DeterministicOrchestrator(validator=validator)

        with self.assertRaisesRegex(ValueError, "deterministically ready"):
            orchestrator.prepare_formulation(gathering)

        self.assertEqual(validator.requests, [])

    def test_formulation_conversation_identity_uses_ordered_user_turns(self):
        session = FormulationSession(
            messages=(
                ConversationMessage(
                    turn=1,
                    role="user",
                    content="Make a ten by five rectangle.",
                ),
                ConversationMessage(
                    turn=1,
                    role="assistant",
                    content="What material and load should be used?",
                ),
                ConversationMessage(
                    turn=2,
                    role="user",
                    content="Use E 100 and a downward traction.",
                ),
                ConversationMessage(
                    turn=2,
                    role="assistant",
                    content="I retained those facts.",
                ),
            )
        )

        context = ConversationContext.from_formulation_session(session)

        self.assertEqual(
            context.original_request,
            "Make a ten by five rectangle.",
        )
        self.assertEqual(len(context.clarifications), 1)
        self.assertEqual(
            context.clarifications[0].answer,
            "Use E 100 and a downward traction.",
        )
        self.assertNotIn(
            "I retained those facts.",
            context.model_dump_json(),
        )

    def test_legacy_start_requires_an_interpreter(self):
        with self.assertRaisesRegex(RuntimeError, "No legacy intent interpreter"):
            DeterministicOrchestrator().start("Optimize a beam.")

    def test_ready_compiles_emits_notice_then_validates_exact_typed_request(self):
        timeline = []
        validator = RecordingValidator(timeline=timeline)
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=validator,
            event_callback=lambda event: timeline.append(event.stage),
        )

        outcome = orchestrator.start("Optimize my square beam")

        self.assertEqual(outcome.status, "awaiting_run_approval")
        self.assertEqual(
            timeline,
            [
                "interpreted",
                "defaults_applied",
                "validator",
                "validated",
                "approval_requested",
            ],
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

        self.assertEqual(outcome.status, "awaiting_run_approval")
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

    def test_solver_is_blocked_until_explicit_approval(self):
        runner = RecordingRunner()
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
            runner=runner,
            analyzer=RecordingAnalyzer(),
        )

        awaiting = orchestrator.start("Optimize a beam")

        self.assertIsInstance(awaiting, AwaitingRunApproval)
        self.assertEqual(runner.calls, [])
        with self.assertRaises(TypeError):
            orchestrator.execute(awaiting)
        self.assertEqual(runner.calls, [])

        validated = orchestrator.approve(awaiting)
        completed = orchestrator.execute(validated)

        self.assertEqual(completed.status, "completed")
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(validated.events[-1].stage, "run_approved")

    def test_parameter_change_reinterprets_and_requires_fresh_approval(self):
        interpreter = FakeInterpreter([ready_result(), ready_result()])
        runner = RecordingRunner()
        orchestrator = DeterministicOrchestrator(
            interpreter,
            validator=RecordingValidator(),
            runner=runner,
        )
        first = orchestrator.start("Optimize a beam")

        revised = orchestrator.revise(first, "Use a 40 x 20 element mesh")

        self.assertIsInstance(revised, AwaitingRunApproval)
        self.assertEqual(runner.calls, [])
        self.assertIn(
            "User answer: Use a 40 x 20 element mesh",
            interpreter.requests[-1],
        )
        self.assertEqual(revised.events[-1].stage, "approval_requested")

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
        validated = approved(orchestrator, "Optimize a beam")

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
        validated = approved(orchestrator, "same request")
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
        validated = approved(orchestrator)

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

        outcome = orchestrator.execute(approved(orchestrator))

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
            orchestrator.execute(approved(orchestrator, "stable request"))
            keys.append(runner.calls[0][1].idempotency_key)

        self.assertEqual(keys[0], keys[1])

    def test_optional_explainer_preserves_analysis_and_is_idempotent(self):
        explainer = RecordingExplainer()
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
            runner=RecordingRunner(),
            analyzer=RecordingAnalyzer(),
            explainer=explainer,
        )
        completed = orchestrator.execute(approved(orchestrator))

        explained = orchestrator.explain(completed)
        refreshed = orchestrator.explain(explained)

        self.assertEqual(explained.status, "explained")
        self.assertIs(explained.analysis, completed.analysis)
        self.assertIs(explainer.analyses[0], completed.analysis)
        self.assertEqual(len(explainer.analyses), 1)
        self.assertIs(refreshed, explained)
        self.assertEqual(explained.events[-1].stage, "explained")

    def test_execution_identity_and_cancellation_use_trusted_derived_run_id(self):
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
        )
        validated = approved(orchestrator, "stable request")
        identity = orchestrator.execution_identity(validated)

        self.assertTrue(identity.idempotency_key.startswith("agentic-workflow-v1:"))
        self.assertRegex(
            identity.run_id,
            r"^compliance_2d_[0-9a-f]{16}$",
        )
        with mock.patch(
            "agentic.orchestrator.request_cancellation",
            return_value=True,
        ) as cancel:
            accepted = orchestrator.request_cancel(validated)

        self.assertTrue(accepted)
        cancel.assert_called_once_with(
            mock.ANY,
            identity.run_id,
        )
        self.assertEqual(str(cancel.call_args.args[0]), "results")

    def test_explain_requires_configured_explainer(self):
        orchestrator = DeterministicOrchestrator(
            FakeInterpreter([ready_result()]),
            validator=RecordingValidator(),
            runner=RecordingRunner(),
            analyzer=RecordingAnalyzer(),
        )
        completed = orchestrator.execute(approved(orchestrator))

        with self.assertRaises(RuntimeError):
            orchestrator.explain(completed)


if __name__ == "__main__":
    unittest.main()
