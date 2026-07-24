from __future__ import annotations

import json
import unittest

from agentic.boundary_draft import (
    BoundaryCreate,
    BoundaryFieldInput,
    BoundaryPatch,
)
from agentic.compiler import compile_formulation_draft, compile_intent
from agentic.formulation import (
    ConversationFormulator,
    DraftNotReadyError,
    DraftUpdate,
    FormulationAgentResponse,
    FormulationModelState,
    FormulationSession,
    FormulationTurn,
    ProblemDraft,
    assess_draft,
    finalize_draft,
    merge_formulation_turn,
)
from fenitop.tools.validate_config import validate_config_tool


LEFT_SUPPORT = [
    {"region": {"op": "plane", "axis": "x", "value": 0}}
]
RIGHT_TRACTION = [
    {
        "edge_segment": {
            "edge": "right",
            "center_fraction": 0.5,
            "span_fraction": 0.1,
        },
        "vector": [0, -1],
    }
]


def update(path, value, quote, *, basis="explicit", rationale="Captured fact."):
    return DraftUpdate(
        path=path,
        value=value,
        basis=basis,
        source_quote=quote,
        rationale=rationale,
    )


def complete_turn(user_message, *, bounds=None):
    bounds = bounds or [[0, 0], [10, 5]]
    return FormulationTurn(
        assistant_message=(
            "I have enough information to show you the proposed problem."
        ),
        updates=(
            update(
                "problem_type",
                "minimize_compliance",
                user_message,
            ),
            update("domain.bounds", bounds, user_message, basis="derived"),
            update("material.young_modulus", 10, user_message),
            update("material.poisson_ratio", 0.3, user_message),
            update("supports", LEFT_SUPPORT, user_message, basis="derived"),
            update("tractions", RIGHT_TRACTION, user_message, basis="derived"),
            update("volume_fraction", 0.33, user_message),
        ),
        declared_state="ready",
    )


class CannedFormulationAgent:
    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def formulate(self, request):
        self.requests.append(request)
        return self.turns.pop(0)


class FirstClassCannedFormulationAgent(CannedFormulationAgent):
    first_class_boundary_patches = True


class FormulationTests(unittest.TestCase):
    def test_first_class_adapter_migrates_a_legacy_session_before_model_turn(self):
        original = "Use the complete legacy cantilever."
        legacy = merge_formulation_turn(
            ProblemDraft(),
            complete_turn(original),
            user_message=original,
            turn_number=1,
        )
        agent = FirstClassCannedFormulationAgent([
            FormulationTurn(
                assistant_message="I retained the migrated BCs.",
                questions=("Which mechanical units should be used?",),
            )
        ])

        step = ConversationFormulator(agent).advance(
            FormulationSession(draft=legacy.draft),
            "Keep the boundary conditions as they are.",
        )

        self.assertEqual(
            [
                condition.bc_id
                for condition in agent.requests[0].draft.boundary_state.conditions
            ],
            ["S1", "L1"],
        )
        self.assertEqual(
            [
                condition.bc_id
                for condition in step.session.draft.boundary_state.conditions
            ],
            ["S1", "L1"],
        )
        self.assertIn("units.length", step.readiness.missing_fields)

    def test_ready_first_class_bc_step_uses_finalized_draft_not_legacy_intent(self):
        user = (
            "Use a 10 by 4 compliance domain, E 100 Pa, nu 0.3, units m N Pa, "
            "clamp the left edge, apply total force [0,-20] N uniformly on the "
            "right edge, and use 40 percent material."
        )

        def boundary_field(name, value):
            return BoundaryFieldInput(
                field=name,
                value=value,
                basis="explicit",
                source_quote=user,
                rationale="Captured from the complete request.",
            )

        turn = FormulationTurn(
            assistant_message="The first-class boundary problem is ready.",
            updates=(
                update("problem_type", "minimize_compliance", user),
                update("domain.bounds", [[0, 0], [10, 4]], user),
                update("material.young_modulus", 100, user),
                update("material.poisson_ratio", 0.3, user),
                update("units.length", "m", user),
                update("units.force", "N", user),
                update("units.stress", "Pa", user),
                update("volume_fraction", 0.4, user),
            ),
            boundary_patch=BoundaryPatch(
                creates=(
                    BoundaryCreate(
                        local_ref="new_support",
                        kind="support",
                        fields=(
                            boundary_field("support.kind", "fixed_all"),
                            boundary_field("selector.kind", "whole_edge"),
                            boundary_field("selector.edge", "left"),
                        ),
                    ),
                    BoundaryCreate(
                        local_ref="new_load",
                        kind="load",
                        fields=(
                            boundary_field("load.kind", "resultant_vector"),
                            boundary_field("load.vector", [0, -20]),
                            boundary_field("load.distribution", "uniform"),
                            boundary_field("selector.kind", "whole_edge"),
                            boundary_field("selector.edge", "right"),
                        ),
                    ),
                )
            ),
            declared_state="ready",
        )

        step = ConversationFormulator(
            CannedFormulationAgent([turn])
        ).start(user)

        self.assertEqual(step.session.status, "ready_for_review")
        self.assertIsNone(step.intent)
        self.assertIs(step.finalized_draft, step.session.draft)
        compilation = compile_formulation_draft(step.finalized_draft)
        self.assertEqual(
            compilation.config.fem.boundary_conditions[1].kind,
            "uniform_resultant",
        )

    def test_turn_schema_is_small_and_supports_conversation_plus_patch(self):
        schema = json.dumps(FormulationTurn.model_json_schema())

        self.assertLess(len(schema), 30_000)
        self.assertIn("assistant_message", schema)
        self.assertIn("updates", schema)
        self.assertIn("questions", schema)
        self.assertNotIn("minimize_compliance.tractions", schema)

    def test_partial_draft_reports_missing_fields_without_forcing_full_intent(self):
        user = "Make a 10 by 5 domain starting at [0, 0]."
        turn = FormulationTurn(
            assistant_message="I understand the rectangular domain.",
            updates=(
                update(
                    "domain.bounds",
                    [[0, 0], [10, 5]],
                    "10 by 5 domain starting at [0, 0]",
                    basis="derived",
                ),
            ),
            questions=("What should be optimized?",),
        )

        result = merge_formulation_turn(
            ProblemDraft(),
            turn,
            user_message=user,
            turn_number=1,
        )
        readiness = assess_draft(result.draft)

        self.assertEqual(result.issues, ())
        self.assertEqual(
            result.draft.fact("domain.bounds").value,
            [[0, 0], [10, 5]],
        )
        self.assertFalse(readiness.ready)
        self.assertIn("problem_type", readiness.missing_fields)
        self.assertIn("external_load", readiness.missing_fields)
        with self.assertRaises(DraftNotReadyError):
            finalize_draft(result.draft)

    def test_formulation_only_geometry_and_support_finalize_deterministically(self):
        first_user = (
            "Make it ten long and half as tall. Fix its left side and keep "
            "roughly one third of the material."
        )
        first = merge_formulation_turn(
            ProblemDraft(),
            FormulationTurn(
                assistant_message=(
                    "I retained the dimensions and relative support without "
                    "inventing an origin."
                ),
                updates=(
                    update(
                        "domain.width",
                        10,
                        "ten long and half as tall",
                        basis="derived",
                    ),
                    update(
                        "domain.height",
                        5,
                        "ten long and half as tall",
                        basis="derived",
                    ),
                    update(
                        "support_edges",
                        ["left"],
                        "Fix its left side",
                        basis="derived",
                    ),
                    update(
                        "volume_fraction",
                        1 / 3,
                        "roughly one third of the material",
                        basis="derived",
                    ),
                ),
            ),
            user_message=first_user,
            turn_number=1,
        )

        self.assertNotIn("domain.bounds", first.draft.values())
        self.assertEqual(first.draft.fact("domain.width").value, 10)
        self.assertEqual(first.draft.fact("domain.height").value, 5)
        self.assertEqual(first.draft.fact("support_edges").value, ["left"])

        second_user = (
            "Put the lower-left corner at the origin. Minimize compliance "
            "with E 10 and nu 0.3. Push downward with traction [0,-1] over "
            "the middle ten percent of the right edge."
        )
        second = merge_formulation_turn(
            first.draft,
            FormulationTurn(
                assistant_message="The previously partial geometry is complete.",
                updates=(
                    update(
                        "problem_type",
                        "minimize_compliance",
                        "Minimize compliance",
                    ),
                    update(
                        "domain.origin",
                        [0, 0],
                        "lower-left corner at the origin",
                        basis="derived",
                    ),
                    update(
                        "material.young_modulus",
                        10,
                        "E 10",
                    ),
                    update(
                        "material.poisson_ratio",
                        0.3,
                        "nu 0.3",
                    ),
                    update(
                        "tractions",
                        RIGHT_TRACTION,
                        (
                            "traction [0,-1] over the middle ten percent "
                            "of the right edge"
                        ),
                        basis="derived",
                    ),
                ),
                declared_state="ready",
            ),
            user_message=second_user,
            turn_number=2,
        )

        intent = finalize_draft(second.draft)

        self.assertEqual(intent.domain.bounds, ((0, 0), (10, 5)))
        self.assertEqual(
            intent.supports[0].region.model_dump(mode="json"),
            {
                "op": "plane",
                "axis": "x",
                "value": 0,
                "tol": 1e-8,
            },
        )
        self.assertAlmostEqual(intent.volume_fraction, 1 / 3)

    def test_long_short_mesh_preference_waits_for_geometry_then_maps_to_xy(self):
        first_user = (
            "Use 60 cells along the long side and 30 along the short side."
        )
        first = merge_formulation_turn(
            ProblemDraft(),
            FormulationTurn(
                assistant_message=(
                    "I retained the relative mesh preference until geometry "
                    "is available."
                ),
                updates=(
                    update(
                        "mesh.long_short_divisions",
                        [60, 30],
                        (
                            "60 cells along the long side and 30 along the "
                            "short side"
                        ),
                    ),
                ),
            ),
            user_message=first_user,
            turn_number=1,
        )
        self.assertFalse(assess_draft(first.draft).ready)

        complete_user = "Use the complete 10 by 5 compliance example."
        complete = merge_formulation_turn(
            first.draft,
            complete_turn(complete_user),
            user_message=complete_user,
            turn_number=2,
        )
        intent = finalize_draft(complete.draft)

        self.assertEqual(intent.mesh.divisions, (60, 30))
        self.assertEqual(
            complete.draft.fact("mesh.long_short_divisions").source_turn,
            1,
        )

    def test_conflicting_complete_and_component_geometry_blocks_finalization(self):
        user = "Use bounds 0,0 to 10,5 but make the width 12."
        result = merge_formulation_turn(
            ProblemDraft(),
            FormulationTurn(
                assistant_message="These geometry statements conflict.",
                updates=(
                    update(
                        "domain.bounds",
                        [[0, 0], [10, 5]],
                        "bounds 0,0 to 10,5",
                        basis="derived",
                    ),
                    update(
                        "domain.origin",
                        [0, 0],
                        "bounds 0,0 to 10,5",
                        basis="derived",
                    ),
                    update(
                        "domain.width",
                        12,
                        "width 12",
                    ),
                    update(
                        "domain.height",
                        5,
                        "bounds 0,0 to 10,5",
                        basis="derived",
                    ),
                ),
            ),
            user_message=user,
            turn_number=1,
        )

        readiness = assess_draft(result.draft)

        self.assertIn(
            "domain.bounds conflicts with domain.origin/width/height.",
            readiness.semantic_errors,
        )

    def test_model_assumption_is_visible_and_cannot_finalize_until_confirmed(self):
        first_user = "It is a cantilever."
        first = merge_formulation_turn(
            ProblemDraft(),
            FormulationTurn(
                assistant_message=(
                    "I suspect compliance minimization, but need confirmation."
                ),
                updates=(
                    DraftUpdate(
                        path="problem_type",
                        value="minimize_compliance",
                        basis="assumption",
                        rationale=(
                            "Cantilevers are commonly used as compliance examples."
                        ),
                    ),
                ),
                questions=("Should I minimize compliance?",),
            ),
            user_message=first_user,
            turn_number=1,
        )

        self.assertEqual(
            assess_draft(first.draft).unconfirmed_fields,
            ("problem_type",),
        )

        confirmation = merge_formulation_turn(
            first.draft,
            FormulationTurn(
                assistant_message="Compliance minimization is now confirmed.",
                updates=(
                    update(
                        "problem_type",
                        "minimize_compliance",
                        "Yes, minimize compliance",
                        basis="confirmed",
                    ),
                ),
            ),
            user_message="Yes, minimize compliance.",
            turn_number=2,
        )

        self.assertEqual(confirmation.issues, ())
        self.assertEqual(
            confirmation.draft.fact("problem_type").basis,
            "confirmed",
        )
        self.assertNotIn(
            "problem_type",
            assess_draft(confirmation.draft).unconfirmed_fields,
        )

    def test_provenance_and_field_validation_reject_bad_patch_without_data_loss(self):
        user = "Use Poisson ratio 0.3."
        turn = FormulationTurn(
            assistant_message="I tried to record the material.",
            updates=(
                update(
                    "material.poisson_ratio",
                    0.7,
                    "Poisson ratio 0.3",
                ),
                update(
                    "material.young_modulus",
                    100,
                    "Young's modulus 100",
                ),
            ),
        )

        result = merge_formulation_turn(
            ProblemDraft(),
            turn,
            user_message=user,
            turn_number=1,
        )

        self.assertEqual(result.accepted_paths, ())
        self.assertEqual(
            [issue.code for issue in result.issues],
            ["invalid_value", "unsupported_provenance"],
        )
        self.assertEqual(result.draft.facts, ())
        self.assertEqual(result.draft.turn_count, 1)

    def test_multi_turn_engine_preserves_real_history_and_finalizes(self):
        first_user = (
            "I need a 10 by 5 domain starting at [0, 0], clamp the entire "
            "left edge, and use 33 percent material."
        )
        first_turn = FormulationTurn(
            assistant_message=(
                "I understand the domain, clamp, and material fraction. "
                "I suspect compliance minimization, but still need material "
                "properties and the load."
            ),
            updates=(
                DraftUpdate(
                    path="problem_type",
                    value="minimize_compliance",
                    basis="assumption",
                    rationale="A cantilever-style request often minimizes compliance.",
                ),
                update(
                    "domain.bounds",
                    [[0, 0], [10, 5]],
                    "10 by 5 domain starting at [0, 0]",
                    basis="derived",
                ),
                update(
                    "supports",
                    LEFT_SUPPORT,
                    "clamp the entire left edge",
                    basis="derived",
                ),
                update(
                    "volume_fraction",
                    0.33,
                    "33 percent material",
                    basis="derived",
                ),
            ),
            questions=(
                "Should I minimize compliance?",
                "What material properties should I use?",
                "How should the right-side load be distributed?",
            ),
        )
        second_user = (
            "Yes, minimize compliance. Use Young's modulus 10 and Poisson "
            "ratio 0.3. Apply a distributed traction [0, -1] over the "
            "centered 10 percent of the right edge."
        )
        second_turn = FormulationTurn(
            assistant_message=(
                "I now have a complete problem draft and will show it for review."
            ),
            updates=(
                update(
                    "problem_type",
                    "minimize_compliance",
                    "Yes, minimize compliance",
                    basis="confirmed",
                ),
                update(
                    "material.young_modulus",
                    10,
                    "Young's modulus 10",
                ),
                update(
                    "material.poisson_ratio",
                    0.3,
                    "Poisson ratio 0.3",
                ),
                update(
                    "tractions",
                    RIGHT_TRACTION,
                    (
                        "distributed traction [0, -1] over the centered "
                        "10 percent of the right edge"
                    ),
                    basis="derived",
                ),
            ),
            declared_state="ready",
        )
        agent = CannedFormulationAgent([first_turn, second_turn])
        formulator = ConversationFormulator(agent)

        first = formulator.start(first_user)
        second = formulator.advance(first.session, second_user)

        self.assertEqual(first.session.status, "gathering")
        self.assertIsNone(first.intent)
        self.assertEqual(second.session.status, "ready_for_review")
        self.assertIsNotNone(second.intent)
        self.assertEqual(second.intent.domain.bounds, ((0, 0), (10, 5)))
        self.assertEqual(second.intent.volume_fraction, 0.33)
        self.assertEqual(
            second.intent.tractions[0].edge_segment.span_fraction,
            0.1,
        )
        self.assertEqual(len(second.session.messages), 4)
        self.assertEqual(
            agent.requests[1].history,
            first.session.messages,
        )
        self.assertEqual(
            agent.requests[1].draft,
            first.session.draft,
        )
        compilation = compile_intent(second.intent)
        validation = validate_config_tool({"config": compilation.config})
        self.assertEqual(validation["status"], "ok", validation["errors"])

    def test_user_correction_replaces_fact_and_records_revision(self):
        original_user = "Use the complete 10 by 5 compliance example."
        original = merge_formulation_turn(
            ProblemDraft(),
            complete_turn(original_user),
            user_message=original_user,
            turn_number=1,
        )
        correction_user = "Change the domain to 12 by 5."
        corrected = merge_formulation_turn(
            original.draft,
            FormulationTurn(
                assistant_message="I changed only the domain dimensions.",
                updates=(
                    update(
                        "domain.bounds",
                        [[0, 0], [12, 5]],
                        "Change the domain to 12 by 5",
                        basis="derived",
                    ),
                ),
                declared_state="ready",
            ),
            user_message=correction_user,
            turn_number=2,
        )

        intent = finalize_draft(corrected.draft)

        self.assertEqual(intent.domain.bounds, ((0, 0), (12, 5)))
        revision = corrected.draft.revisions[-1]
        self.assertEqual(revision.previous_value, [[0, 0], [10, 5]])
        self.assertEqual(revision.new_value, [[0, 0], [12, 5]])

    def test_engine_distinguishes_patch_repair_from_unsupported_problem(self):
        invalid_turn = FormulationTurn(
            assistant_message="I understood a Poisson ratio, but encoding failed.",
            updates=(
                update(
                    "material.poisson_ratio",
                    0.8,
                    "Poisson ratio 0.8",
                ),
            ),
        )
        unsupported_turn = FormulationTurn(
            assistant_message=(
                "The current solver does not support a three-dimensional domain."
            ),
            declared_state="unsupported",
            unsupported_features=("3D domain",),
        )

        repair = ConversationFormulator(
            CannedFormulationAgent([invalid_turn]),
            max_repair_attempts=0,
        ).start("Use Poisson ratio 0.8.")
        unsupported = ConversationFormulator(
            CannedFormulationAgent([unsupported_turn])
        ).start("Optimize a three-dimensional bracket.")

        self.assertEqual(repair.session.status, "repair_needed")
        self.assertEqual(repair.merge.issues[0].code, "invalid_value")
        self.assertEqual(unsupported.session.status, "unsupported")
        self.assertEqual(
            unsupported.session.unsupported_features,
            ("3D domain",),
        )

        negotiable = ConversationFormulator(
            CannedFormulationAgent(
                [
                    FormulationTurn(
                        assistant_message=(
                            "Point loads are unsupported, but we can use a "
                            "distributed traction segment."
                        ),
                        questions=(
                            "What segment width and traction magnitude should be used?",
                        ),
                        unsupported_features=("point load",),
                    )
                ]
            )
        ).start("Apply a point load at the free edge.")

        self.assertEqual(negotiable.session.status, "gathering")
        self.assertEqual(
            negotiable.session.unsupported_features,
            ("point load",),
        )

    def test_engine_repairs_rejected_patch_on_same_user_turn(self):
        user = "Use Poisson ratio 0.3."
        invalid = FormulationTurn(
            assistant_message="I recorded the material value.",
            updates=(
                update(
                    "material.poisson_ratio",
                    0.8,
                    "Poisson ratio 0.3",
                ),
            ),
        )
        corrected = FormulationTurn(
            assistant_message="I recorded Poisson ratio 0.3.",
            updates=(
                update(
                    "material.poisson_ratio",
                    0.3,
                    "Poisson ratio 0.3",
                ),
            ),
        )
        agent = CannedFormulationAgent([invalid, corrected])

        result = ConversationFormulator(
            agent,
            max_repair_attempts=1,
        ).start(user)

        self.assertEqual(len(agent.requests), 2)
        self.assertEqual(agent.requests[0].turn_number, 1)
        self.assertIsNone(agent.requests[0].repair)
        self.assertEqual(agent.requests[1].turn_number, 1)
        self.assertEqual(agent.requests[1].repair.attempt, 2)
        self.assertEqual(
            agent.requests[1].repair.issues[0].code,
            "invalid_value",
        )
        self.assertEqual(result.merge.issues, ())
        self.assertEqual(
            result.session.draft.fact("material.poisson_ratio").value,
            0.3,
        )
        self.assertEqual(result.session.draft.turn_count, 1)
        self.assertEqual(len(result.session.messages), 2)

    def test_engine_carries_adapter_state_through_repair(self):
        first_state = FormulationModelState(
            adapter="test-adapter",
            continuation_id="response-1",
        )
        repaired_state = FormulationModelState(
            adapter="test-adapter",
            continuation_id="response-2",
        )
        invalid = FormulationAgentResponse(
            turn=FormulationTurn(
                assistant_message="I encoded the value.",
                updates=(
                    update(
                        "material.poisson_ratio",
                        0.8,
                        "Poisson ratio 0.3",
                    ),
                ),
            ),
            model_state=first_state,
        )
        corrected = FormulationAgentResponse(
            turn=FormulationTurn(
                assistant_message="I corrected the encoding.",
                updates=(
                    update(
                        "material.poisson_ratio",
                        0.3,
                        "Poisson ratio 0.3",
                    ),
                ),
            ),
            model_state=repaired_state,
        )
        agent = CannedFormulationAgent([invalid, corrected])

        result = ConversationFormulator(agent).start(
            "Use Poisson ratio 0.3."
        )

        self.assertEqual(agent.requests[1].model_state, first_state)
        self.assertEqual(result.session.model_state, repaired_state)

    def test_blank_turn_and_nonsequential_merge_are_rejected(self):
        formulator = ConversationFormulator(CannedFormulationAgent([]))
        with self.assertRaises(ValueError):
            formulator.start(" ")
        with self.assertRaises(ValueError):
            merge_formulation_turn(
                ProblemDraft(),
                FormulationTurn(assistant_message="No changes."),
                user_message="message",
                turn_number=2,
            )
        with self.assertRaises(ValueError):
            ConversationFormulator(
                CannedFormulationAgent([]),
                max_repair_attempts=3,
            )


if __name__ == "__main__":
    unittest.main()
