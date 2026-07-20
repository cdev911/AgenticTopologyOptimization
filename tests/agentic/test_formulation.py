from __future__ import annotations

import json
import unittest

from agentic.compiler import compile_intent
from agentic.formulation import (
    ConversationFormulator,
    DraftNotReadyError,
    DraftUpdate,
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


class FormulationTests(unittest.TestCase):
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
            CannedFormulationAgent([invalid_turn])
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


if __name__ == "__main__":
    unittest.main()
