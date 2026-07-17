from __future__ import annotations

import unittest

from agentic.compiler import compile_intent
from agentic.intent import InterpretationEnvelope
from agentic.orchestrator import (
    AwaitingClarification,
    DeterministicOrchestrator,
)
from fenitop.tools.contracts import ValidateConfigRequest
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


if __name__ == "__main__":
    unittest.main()
