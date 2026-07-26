from __future__ import annotations

import base64
import json
import unittest
from types import SimpleNamespace
from tempfile import TemporaryDirectory
from pathlib import Path

from streamlit.testing.v1 import AppTest

from agentic.boundary_draft import (
    BoundaryCreate,
    BoundaryFieldInput,
    BoundaryPatch,
)
from agentic.compiler import compile_intent
from agentic.formulation import (
    ConversationFormulator,
    DraftUpdate,
    FormulationTurn,
)
from agentic.formulation_openai import FormulationAPIError
from agentic.intent import ComplianceProblemIntent
from agentic.orchestrator import (
    AwaitingRunApproval,
    DeterministicOrchestrator,
)
from fenitop.tools.contracts import AnalyzeResultsResponse
from fenitop.tools.contracts import ValidateConfigResponse
from fenitop.tools.validate_config import validate_config_tool


REPO_ROOT = Path(__file__).resolve().parents[2]


def _awaiting_approval():
    compilation = compile_intent(
        ComplianceProblemIntent.model_validate(
            {
                "problem_type": "minimize_compliance",
                "domain": {"bounds": [[0, 0], [4, 2]]},
                "material": {"young_modulus": 100, "poisson_ratio": 0.3},
                "supports": [
                    {"region": {"op": "plane", "axis": "x", "value": 0}}
                ],
                "tractions": [
                    {
                        "region": {"op": "plane", "axis": "x", "value": 4},
                        "vector": [0, -1],
                    }
                ],
                "volume_fraction": 0.4,
            }
        )
    )
    validation = ValidateConfigResponse.model_validate(
        validate_config_tool({"config": compilation.config})
    )
    return AwaitingRunApproval(
        conversation={"original_request": "test request"},
        compilation=compilation,
        validation=validation,
        events=(),
    )


class ApprovalOnlyOrchestrator:
    def __init__(self, outcome):
        self.outcome = outcome
        self.prepare_calls = []

    def prepare_formulation(self, step):
        self.prepare_calls.append(step)
        return self.outcome


class CannedFormulationAgent:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def formulate(self, request):
        self.requests.append(request)
        return self.turns.pop(0)


class RaisingFormulator:
    def advance(self, _session, _message):
        raise FormulationAPIError("provider", "APITimeoutError")


class ApprovalExecutionOrchestrator:
    def __init__(self):
        self.approve_calls = []
        self.execute_calls = []

    def approve(self, prior):
        self.approve_calls.append(prior)
        return DeterministicOrchestrator().approve(prior)

    def execute(self, validated):
        self.execute_calls.append(validated)
        return SimpleNamespace(status="test_terminal", events=())

    def execution_identity(self, _validated):
        return SimpleNamespace(run_id="ui_approval_test")

    def request_cancel(self, _validated):
        return False


def _update(path, value, user, *, basis="explicit"):
    return DraftUpdate(
        path=path,
        value=value,
        basis=basis,
        source_quote=None if basis == "assumption" else user,
        rationale="Canned public modeling rationale.",
    )


def _ready_turn(user):
    return FormulationTurn(
        assistant_message=(
            "I have a complete problem definition and it is ready for review."
        ),
        updates=(
            _update("problem_type", "minimize_compliance", user),
            _update("domain.bounds", [[0, 0], [4, 2]], user),
            _update("material.young_modulus", 100, user),
            _update("material.poisson_ratio", 0.3, user),
            _update("support_edges", ["left"], user),
            _update(
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
                user,
            ),
            _update("volume_fraction", 0.4, user),
        ),
        declared_state="ready",
    )


class StreamlitAppTests(unittest.TestCase):
    def test_initial_ui_is_chat_only_and_exposes_no_config_form(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.title[0].value, "Agentic Topology Optimization")
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(len(app.text_input), 0)
        self.assertEqual(len(app.text_area), 0)
        self.assertEqual(len(app.json), 0)
        self.assertEqual(len(app.chat_message), 1)

    def test_reset_keeps_a_clean_initial_conversation(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()

        app.button[0].click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.chat_input), 1)
        self.assertEqual(len(app.chat_message), 1)
        self.assertEqual(
            app.session_state["formulation_session"].draft.turn_count,
            0,
        )

    def test_valid_request_waits_in_chat_without_submitting_a_job(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        orchestrator = ApprovalOnlyOrchestrator(_awaiting_approval())
        app.session_state["orchestrator"] = orchestrator
        user = (
            "Minimize compliance on a 4 by 2 rectangle at the origin with E "
            "100, nu 0.3, left edge fixed, downward right-edge traction, and "
            "40 percent material."
        )
        app.session_state["formulator"] = ConversationFormulator(
            CannedFormulationAgent([_ready_turn(user)])
        )

        app.chat_input[0].set_value(user).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(orchestrator.prepare_calls), 1)
        self.assertEqual(
            app.session_state["outcome"].status,
            "awaiting_run_approval",
        )
        self.assertIsNone(app.session_state["job_future"])
        self.assertIn(
            "Do you approve these parameters",
            app.chat_message[-1].markdown[0].value,
        )
        self.assertEqual(len(app.image), 1)
        self.assertTrue(
            any("S1 · Support" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("L1 · Load" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Change L1" in item.value for item in app.caption)
        )

    def test_partial_first_class_bc_has_human_card_and_provenance(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        user = "Put a 10 N resultant somewhere on the right edge."

        def bc_field(name, value):
            return BoundaryFieldInput(
                field=name,
                value=value,
                basis="explicit",
                source_quote=user,
                rationale="Canned BC presentation rationale.",
            )

        turn = FormulationTurn(
            assistant_message=(
                "I retained the partial load as L1 and need its direction and "
                "extent."
            ),
            boundary_patch=BoundaryPatch(
                creates=(
                    BoundaryCreate(
                        local_ref="new_load",
                        kind="load",
                        fields=(
                            bc_field("load.kind", "resultant_magnitude"),
                            bc_field("load.magnitude", 10),
                            bc_field("load.unit", "N"),
                            bc_field("selector.kind", "unspecified_extent"),
                            bc_field("selector.edge", "right"),
                        ),
                    ),
                )
            ),
            questions=("What direction and right-edge extent should L1 use?",),
        )
        app.session_state["formulator"] = ConversationFormulator(
            CannedFormulationAgent([turn])
        )

        app.chat_input[0].set_value(user).run()

        self.assertEqual(len(app.exception), 0)
        self.assertTrue(
            any("L1 · Load" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("extent not yet specified" in item.value for item in app.markdown)
        )
        self.assertTrue(
            any("Change L1" in item.value for item in app.caption)
        )
        provenance = json.loads(app.json[-1].value)
        self.assertEqual(provenance["boundary_conditions"][0]["bc_id"], "L1")
        self.assertEqual(provenance["boundary_revisions"][0]["action"], "create")

    def test_multi_turn_formulation_retains_draft_then_prepares_when_ready(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        orchestrator = ApprovalOnlyOrchestrator(_awaiting_approval())
        app.session_state["orchestrator"] = orchestrator
        first_user = (
            "Make a 4 by 2 rectangle at the origin and clamp the left edge."
        )
        second_user = (
            "Minimize compliance with E 100, nu 0.3, traction [0,-1] on "
            "the right edge, and 40 percent material."
        )
        first_turn = FormulationTurn(
            assistant_message=(
                "I retained the geometry and support. Material and load details "
                "are still needed."
            ),
            updates=(
                _update("domain.bounds", [[0, 0], [4, 2]], first_user),
                _update("support_edges", ["left"], first_user),
            ),
            questions=(
                "What objective, material, load, and material fraction should "
                "be used?",
            ),
        )
        second_turn = FormulationTurn(
            assistant_message=(
                "The retained geometry and new physics make the draft complete."
            ),
            updates=(
                _update(
                    "problem_type",
                    "minimize_compliance",
                    second_user,
                ),
                _update(
                    "material.young_modulus",
                    100,
                    second_user,
                ),
                _update(
                    "material.poisson_ratio",
                    0.3,
                    second_user,
                ),
                _update(
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
                    second_user,
                ),
                _update("volume_fraction", 0.4, second_user),
            ),
            declared_state="ready",
        )
        agent = CannedFormulationAgent([first_turn, second_turn])
        app.session_state["formulator"] = ConversationFormulator(agent)

        app.chat_input[0].set_value(first_user).run()

        self.assertIsNone(app.session_state["outcome"])
        self.assertEqual(len(orchestrator.prepare_calls), 0)
        self.assertEqual(
            app.session_state["formulation_session"].draft.fact(
                "domain.bounds"
            ).value,
            [[0, 0], [4, 2]],
        )
        self.assertIn(
            "Questions to continue",
            app.chat_message[-1].markdown[0].value,
        )

        app.chat_input[0].set_value(second_user).run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(agent.requests), 2)
        self.assertEqual(len(orchestrator.prepare_calls), 1)
        self.assertEqual(
            app.session_state["formulation_session"].status,
            "ready_for_review",
        )
        self.assertEqual(
            app.session_state["outcome"].status,
            "awaiting_run_approval",
        )
        self.assertIsNone(app.session_state["job_future"])

    def test_assumptions_are_visible_and_cannot_prepare_a_run(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        orchestrator = ApprovalOnlyOrchestrator(_awaiting_approval())
        app.session_state["orchestrator"] = orchestrator
        user = "Use a typical material if needed."
        turn = FormulationTurn(
            assistant_message=(
                "I proposed E=100 as an assumption and need your confirmation."
            ),
            updates=(
                _update(
                    "material.young_modulus",
                    100,
                    user,
                    basis="assumption",
                ),
            ),
            questions=("Should I use Young's modulus 100?",),
        )
        app.session_state["formulator"] = ConversationFormulator(
            CannedFormulationAgent([turn])
        )

        app.chat_input[0].set_value(user).run()

        self.assertIsNone(app.session_state["outcome"])
        self.assertEqual(orchestrator.prepare_calls, [])
        self.assertTrue(
            any(
                "require your confirmation" in warning.value
                for warning in app.warning
            )
        )
        self.assertIn(
            "Confirm or correct",
            app.chat_input[0].placeholder,
        )

    def test_capability_limits_remain_visible_for_reformulation(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        orchestrator = ApprovalOnlyOrchestrator(_awaiting_approval())
        app.session_state["orchestrator"] = orchestrator
        user = "Optimize a three-dimensional mounting bracket."
        turn = FormulationTurn(
            assistant_message=(
                "Three-dimensional geometry is unsupported, but we can "
                "formulate a rectangular 2D plane-strain demonstration."
            ),
            questions=(
                "Would you like to reformulate this as a rectangular 2D "
                "plane-strain problem?",
            ),
            declared_state="gathering",
            unsupported_features=("3D geometry",),
        )
        app.session_state["formulator"] = ConversationFormulator(
            CannedFormulationAgent([turn])
        )

        app.chat_input[0].set_value(user).run()

        self.assertIsNone(app.session_state["outcome"])
        self.assertEqual(orchestrator.prepare_calls, [])
        self.assertTrue(
            any(
                "3D geometry" in warning.value
                for warning in app.warning
            )
        )

    def test_rejected_patch_is_field_specific_and_never_prepares(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        orchestrator = ApprovalOnlyOrchestrator(_awaiting_approval())
        app.session_state["orchestrator"] = orchestrator
        user = "Use Poisson ratio 0.8."
        turn = FormulationTurn(
            assistant_message="I attempted to record the Poisson ratio.",
            updates=(
                _update(
                    "material.poisson_ratio",
                    0.8,
                    user,
                ),
            ),
        )
        app.session_state["formulator"] = ConversationFormulator(
            CannedFormulationAgent([turn]),
            max_repair_attempts=0,
        )

        app.chat_input[0].set_value(user).run()

        self.assertIsNone(app.session_state["outcome"])
        self.assertEqual(orchestrator.prepare_calls, [])
        self.assertEqual(
            app.session_state["formulation_session"].status,
            "repair_needed",
        )
        self.assertTrue(
            any(
                "material.poisson_ratio" in error.value
                for error in app.error
            )
        )

    def test_failed_requested_change_invalidates_the_old_approval(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        old_proposal = _awaiting_approval()
        app.session_state["outcome"] = old_proposal
        app.session_state["formulator"] = RaisingFormulator()
        app.run()

        app.chat_input[0].set_value(
            "Change Young's modulus to 200."
        ).run()

        self.assertEqual(len(app.exception), 0)
        self.assertIsNone(app.session_state["outcome"])
        self.assertIsNone(app.session_state["job_future"])
        self.assertIn(
            "No proposal was prepared",
            app.chat_message[-1].markdown[0].value,
        )

    def test_only_explicit_green_light_submits_the_approved_state(self):
        app = AppTest.from_file(
            REPO_ROOT / "streamlit_app.py",
            default_timeout=10,
        ).run()
        orchestrator = ApprovalExecutionOrchestrator()
        app.session_state["outcome"] = _awaiting_approval()
        app.session_state["orchestrator"] = orchestrator
        app.session_state["formulator"] = RaisingFormulator()
        app.run()

        app.chat_input[0].set_value("yes").run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(orchestrator.approve_calls), 1)
        self.assertEqual(len(orchestrator.execute_calls), 1)
        self.assertEqual(
            orchestrator.execute_calls[0].status,
            "validated",
        )

    def test_completed_analysis_renders_verified_plot_gallery(self):
        with TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            density = run_dir / "density.png"
            compliance = run_dir / "compliance.png"
            png = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lE"
                "QVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
            )
            density.write_bytes(png)
            compliance.write_bytes(png)
            analysis = AnalyzeResultsResponse.model_validate(
                {
                    "contract_version": "5.1.0",
                    "tool": "analyze_results",
                    "status": "ok",
                    "warnings": [],
                    "errors": [],
                    "source": {
                        "run_directory": str(run_dir),
                        "output_prefix": "compliance_2d",
                        "run_id": "compliance_2d_test",
                        "manifest_hash": "a" * 64,
                    },
                    "plots": [
                        {"role": "density_field", "path": str(density)},
                        {
                            "role": "compliance_vs_iteration",
                            "path": str(compliance),
                        },
                    ],
                }
            )
            app = AppTest.from_file(
                REPO_ROOT / "streamlit_app.py",
                default_timeout=10,
            ).run()
            app.session_state["outcome"] = SimpleNamespace(
                events=(),
                analysis=analysis,
            )

            app.run()

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.image), 2)
            self.assertEqual(len(app.download_button), 2)
            self.assertEqual(
                [button.label for button in app.download_button],
                [
                    "Download Final optimized design",
                    "Download Compliance objective vs. iteration",
                ],
            )


if __name__ == "__main__":
    unittest.main()
