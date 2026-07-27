"""Small no-API harness for the deterministic Stage 1 workflow.

This uses a canned, schema-validated interpretation so it tests compilation,
validation, contained execution, manifest handoff, and analysis without spending
model credit. Repeating the command exercises the run tool's idempotent replay.
"""

from agentic.intent import InterpretationEnvelope
from agentic.orchestrator import DeterministicOrchestrator


SMALL_READY_RESULT = InterpretationEnvelope.model_validate(
    {
        "result": {
            "status": "ready",
            "intent": {
                "problem_type": "minimize_compliance",
                "domain": {"bounds": [[0, 0], [4, 2]]},
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
                            "value": 4,
                        },
                        "vector": [0, -1],
                    }
                ],
                "volume_fraction": 0.4,
                "mesh": {
                    "divisions": [8, 4],
                    "cell_type": "quadrilateral",
                },
                "optimization": {
                    "filter_radius": 0.6,
                    "max_iter": 5,
                },
            },
        }
    }
).result


class CannedInterpreter:
    def interpret(self, user_request):
        return SMALL_READY_RESULT


def main() -> int:
    orchestrator = DeterministicOrchestrator(CannedInterpreter())
    awaiting_approval = orchestrator.start(
        "stage1 deterministic workflow harness with an explicit 8 x 4 element "
        "mesh, quadrilateral cells, filter radius 0.6, and maximum iterations 5"
    )
    validated = orchestrator.approve(awaiting_approval)
    outcome = orchestrator.execute(validated)

    print(f"workflow_status={outcome.status}")
    if outcome.status != "completed":
        return 1
    print(f"run_status={outcome.run.status}")
    print(f"analysis_status={outcome.analysis.status}")
    print(f"run_id={outcome.run.run_id}")
    print(f"idempotent_replay={str(outcome.run.idempotent_replay).lower()}")
    print("events=" + ",".join(event.stage for event in outcome.events))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
