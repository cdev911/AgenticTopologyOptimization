from __future__ import annotations

import unittest

from pydantic import ValidationError

from agentic.boundary_draft import (
    BoundaryConfirm,
    BoundaryCreate,
    BoundaryDelete,
    BoundaryDraftState,
    BoundaryFieldInput,
    BoundaryPatch,
    BoundaryUpdate,
    assess_boundary_state,
    merge_boundary_patch,
)
from agentic.formulation import (
    DraftFact,
    FormulationTurn,
    ProblemDraft,
    assess_draft,
    merge_formulation_turn,
    migrate_legacy_boundary_facts,
)


def field(
    name,
    value,
    quote=None,
    *,
    basis="explicit",
    rationale="Captured boundary fact.",
):
    return BoundaryFieldInput(
        field=name,
        value=value,
        basis=basis,
        source_quote=quote,
        rationale=rationale,
    )


def create_support(quote="clamp the left edge"):
    return BoundaryCreate(
        local_ref="new_support",
        kind="support",
        fields=(
            field("support.kind", "fixed_all", quote),
            field("selector.kind", "whole_edge", quote),
            field("selector.edge", "left", quote),
        ),
    )


def create_load(quote="traction [0,-1] on the right edge"):
    return BoundaryCreate(
        local_ref="new_load",
        kind="load",
        fields=(
            field("load.kind", "traction_vector", quote),
            field("load.vector", [0, -1], quote),
            field("load.distribution", "uniform", quote, basis="derived"),
            field("selector.kind", "whole_edge", quote),
            field("selector.edge", "right", quote),
        ),
    )


class BoundaryDraftMergeTests(unittest.TestCase):
    def test_application_allocates_stable_support_and_load_ids(self):
        message = (
            "clamp the left edge and use traction [0,-1] on the right edge"
        )
        result = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(
                creates=(
                    create_support("clamp the left edge"),
                    create_load("traction [0,-1] on the right edge"),
                )
            ),
            user_message=message,
            turn_number=1,
        )

        self.assertEqual(result.issues, ())
        self.assertEqual(
            [item.bc_id for item in result.state.conditions],
            ["S1", "L1"],
        )
        self.assertEqual(
            [(item.local_ref, item.bc_id) for item in result.accepted],
            [("new_support", "S1"), ("new_load", "L1")],
        )
        self.assertEqual(result.state.next_support_number, 2)
        self.assertEqual(result.state.next_load_number, 2)

    def test_invalid_create_is_atomic_and_does_not_consume_id(self):
        invalid = BoundaryCreate(
            local_ref="new_bad",
            kind="support",
            fields=(
                field(
                    "load.kind",
                    "traction_vector",
                    "make a support",
                ),
            ),
        )
        failed = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(invalid,)),
            user_message="make a support",
            turn_number=1,
        )
        self.assertEqual(failed.state.conditions, ())
        self.assertEqual(failed.state.next_support_number, 1)
        self.assertEqual(failed.issues[0].code, "invalid_field")

        valid = merge_boundary_patch(
            failed.state,
            BoundaryPatch(creates=(create_support(),)),
            user_message="clamp the left edge",
            turn_number=2,
        )
        self.assertEqual(valid.state.conditions[0].bc_id, "S1")

    def test_field_update_preserves_other_fields_and_entities(self):
        initial = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(create_support(), create_load())),
            user_message=(
                "clamp the left edge and traction [0,-1] on the right edge"
            ),
            turn_number=1,
        ).state
        changed = merge_boundary_patch(
            initial,
            BoundaryPatch(updates=(
                BoundaryUpdate(
                    bc_id="L1",
                    field="load.vector",
                    value=[0, -3],
                    basis="explicit",
                    source_quote="Change L1 to [0,-3]",
                    rationale="User corrected only the vector.",
                ),
            )),
            user_message="Change L1 to [0,-3]; leave S1 alone.",
            turn_number=2,
        )

        self.assertEqual(changed.issues, ())
        self.assertEqual(
            changed.state.condition("L1").fact("load.vector").value,
            [0, -3],
        )
        self.assertEqual(
            changed.state.condition("L1").fact("selector.edge").value,
            "right",
        )
        self.assertIsNotNone(changed.state.condition("S1"))
        revision = changed.state.revisions[-1]
        self.assertEqual(revision.previous_value, [0, -1])
        self.assertEqual(revision.new_value, [0, -3])

    def test_missing_target_and_wrong_kind_field_fail_closed(self):
        state = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(create_support(),)),
            user_message="clamp the left edge",
            turn_number=1,
        ).state
        result = merge_boundary_patch(
            state,
            BoundaryPatch(updates=(
                BoundaryUpdate(
                    bc_id="L9",
                    field="load.vector",
                    value=[0, -1],
                    basis="explicit",
                    source_quote="change L9",
                    rationale="Missing target.",
                ),
                BoundaryUpdate(
                    bc_id="S1",
                    field="load.vector",
                    value=[0, -1],
                    basis="explicit",
                    source_quote="change S1",
                    rationale="Wrong field family.",
                ),
            )),
            user_message="change L9 and change S1",
            turn_number=2,
        )
        self.assertEqual(
            {issue.code for issue in result.issues},
            {"missing_target", "invalid_field"},
        )
        self.assertEqual(result.accepted, ())

    def test_assumption_creates_typed_pending_confirmation(self):
        create = BoundaryCreate(
            local_ref="new_load",
            kind="load",
            fields=(
                field(
                    "load.distribution",
                    "uniform",
                    basis="assumption",
                    rationale="Uniform distribution is proposed.",
                ),
            ),
        )
        result = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(create,)),
            user_message="Use a total force on the right.",
            turn_number=1,
        )
        pending = result.state.pending_confirmations()
        self.assertEqual(len(pending), 1)
        self.assertEqual(pending[0].bc_id, "L1")
        self.assertEqual(pending[0].field, "load.distribution")
        self.assertEqual(pending[0].value, "uniform")

    def test_complete_constant_traction_upgrades_uniform_assumption_to_derived(self):
        quote = "traction [2,0] on the upper quarter of the right edge"
        create = BoundaryCreate(
            local_ref="new_load",
            kind="load",
            fields=(
                field("load.kind", "traction_vector", quote),
                field("load.vector", [2, 0], quote),
                field(
                    "load.distribution",
                    "uniform",
                    basis="assumption",
                    rationale="The model proposed uniformity.",
                ),
                field("selector.kind", "fraction_interval", quote),
                field("selector.edge", "right", quote),
                field("selector.start", 0.75, quote),
                field("selector.end", 1.0, quote),
            ),
        )

        result = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(create,)),
            user_message=quote,
            turn_number=1,
        )

        distribution = result.state.condition("L1").fact(
            "load.distribution"
        )
        self.assertEqual(distribution.value, "uniform")
        self.assertEqual(distribution.basis, "derived")
        self.assertEqual(result.state.pending_confirmations(), ())
        self.assertTrue(
            assess_boundary_state(result.state).conditions[0].ready
        )

    def test_named_edge_center_survives_unresolved_extent_and_same_assumption(self):
        first_message = "Put a 1 N load on the right face center."
        first = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_load",
                    kind="load",
                    fields=(
                        field("load.magnitude", 1, "1 N"),
                        field("load.unit", "N", "1 N"),
                        field(
                            "selector.kind",
                            "unspecified_extent",
                            "right face center",
                            basis="derived",
                        ),
                        field(
                            "selector.edge",
                            "right",
                            "right face center",
                            basis="derived",
                        ),
                    ),
                ),
            )),
            user_message=first_message,
            turn_number=1,
        )
        center = first.state.condition("L1").fact("selector.center")
        self.assertEqual(center.value, 0.5)
        self.assertEqual(center.basis, "derived")

        second_message = "Use 10 percent of that edge."
        second = merge_boundary_patch(
            first.state,
            BoundaryPatch(updates=(
                BoundaryUpdate(
                    bc_id="L1",
                    field="selector.kind",
                    value="centered_fraction",
                    basis="assumption",
                    source_quote=None,
                    rationale="Propose the centered finite selector.",
                ),
                BoundaryUpdate(
                    bc_id="L1",
                    field="selector.center",
                    value=0.5,
                    basis="assumption",
                    source_quote=None,
                    rationale="Model repeated the already known center.",
                ),
                BoundaryUpdate(
                    bc_id="L1",
                    field="selector.span",
                    value=0.1,
                    basis="derived",
                    source_quote="10 percent",
                    rationale="Convert the stated percentage.",
                ),
            )),
            user_message=second_message,
            turn_number=2,
        )

        center = second.state.condition("L1").fact("selector.center")
        self.assertEqual(center.value, 0.5)
        self.assertEqual(center.basis, "derived")
        self.assertEqual(center.source_turn, 1)

    def test_bare_directional_load_location_does_not_become_whole_edge(self):
        result = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_load",
                    kind="load",
                    fields=(
                        field("load.magnitude", 10, "10"),
                        field(
                            "selector.kind",
                            "whole_edge",
                            "on the right",
                            basis="derived",
                        ),
                        field("selector.edge", "right", "right"),
                    ),
                ),
            )),
            user_message="The load on the right is 10.",
            turn_number=1,
        )

        selector = result.state.condition("L1").fact("selector.kind")
        self.assertEqual(selector.value, "unspecified_extent")
        self.assertEqual(selector.basis, "derived")

    def test_generic_yes_confirms_exactly_one_assumption(self):
        initial = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_load",
                    kind="load",
                    fields=(
                        field(
                            "load.distribution",
                            "uniform",
                            basis="assumption",
                        ),
                    ),
                ),
            )),
            user_message="Use a total force.",
            turn_number=1,
        ).state
        result = merge_boundary_patch(
            initial,
            BoundaryPatch(confirmations=(
                BoundaryConfirm(
                    bc_id="L1",
                    field="load.distribution",
                    source_quote="yes",
                    rationale="User confirmed the sole pending assumption.",
                ),
            )),
            user_message="yes",
            turn_number=2,
        )
        self.assertEqual(result.issues, ())
        fact = result.state.condition("L1").fact("load.distribution")
        self.assertEqual(fact.basis, "confirmed")
        self.assertEqual(result.state.pending_confirmations(), ())

    def test_generic_yes_rejects_ambiguous_multiple_assumptions(self):
        initial = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_load",
                    kind="load",
                    fields=(
                        field(
                            "load.distribution",
                            "uniform",
                            basis="assumption",
                        ),
                        field(
                            "selector.span",
                            0.1,
                            basis="assumption",
                        ),
                    ),
                ),
            )),
            user_message="Approximate the point load.",
            turn_number=1,
        ).state
        result = merge_boundary_patch(
            initial,
            BoundaryPatch(confirmations=(
                BoundaryConfirm(
                    bc_id="L1",
                    field="load.distribution",
                    source_quote="yes",
                    rationale="Attempted generic confirmation.",
                ),
            )),
            user_message="yes",
            turn_number=2,
        )
        self.assertEqual(result.issues[0].code, "invalid_confirmation")
        self.assertEqual(
            result.state.condition("L1").fact("load.distribution").basis,
            "assumption",
        )

    def test_explicit_confirm_all_can_confirm_multiple_assumptions(self):
        initial = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_load",
                    kind="load",
                    fields=(
                        field(
                            "load.distribution",
                            "uniform",
                            basis="assumption",
                        ),
                        field(
                            "selector.span",
                            0.1,
                            basis="assumption",
                        ),
                    ),
                ),
            )),
            user_message="Approximate the point load.",
            turn_number=1,
        ).state
        confirmations = tuple(
            BoundaryConfirm(
                bc_id=item.bc_id,
                field=item.field,
                source_quote="confirm all",
                rationale="User explicitly confirmed every listed assumption.",
            )
            for item in initial.pending_confirmations()
        )
        result = merge_boundary_patch(
            initial,
            BoundaryPatch(confirmations=confirmations),
            user_message="confirm all",
            turn_number=2,
        )
        self.assertEqual(result.issues, ())
        self.assertEqual(result.state.pending_confirmations(), ())

    def test_explicit_correction_replaces_assumption_without_confirmation(self):
        initial = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_load",
                    kind="load",
                    fields=(
                        field(
                            "selector.span",
                            0.1,
                            basis="assumption",
                        ),
                    ),
                ),
            )),
            user_message="Use a finite patch.",
            turn_number=1,
        ).state
        result = merge_boundary_patch(
            initial,
            BoundaryPatch(updates=(
                BoundaryUpdate(
                    bc_id="L1",
                    field="selector.span",
                    value=0.2,
                    basis="explicit",
                    source_quote="make L1 span 20 percent",
                    rationale="User replaced the proposal.",
                ),
            )),
            user_message="No, make L1 span 20 percent.",
            turn_number=2,
        )
        fact = result.state.condition("L1").fact("selector.span")
        self.assertEqual(fact.value, 0.2)
        self.assertEqual(fact.basis, "explicit")
        self.assertEqual(result.state.pending_confirmations(), ())

    def test_delete_preserves_other_entities_and_ids_are_not_reused(self):
        initial = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(create_load(),)),
            user_message="traction [0,-1] on the right edge",
            turn_number=1,
        ).state
        deleted = merge_boundary_patch(
            initial,
            BoundaryPatch(deletes=(
                BoundaryDelete(
                    bc_id="L1",
                    source_quote="Remove L1",
                    rationale="User removed the load.",
                ),
            )),
            user_message="Remove L1.",
            turn_number=2,
        ).state
        self.assertEqual(deleted.conditions, ())
        self.assertEqual(deleted.next_load_number, 2)

        added = merge_boundary_patch(
            deleted,
            BoundaryPatch(creates=(create_load(),)),
            user_message="traction [0,-1] on the right edge",
            turn_number=3,
        ).state
        self.assertEqual(added.conditions[0].bc_id, "L2")

    def test_patch_schema_rejects_update_confirmation_overlap(self):
        update = BoundaryUpdate(
            bc_id="L1",
            field="selector.span",
            value=0.2,
            basis="explicit",
            source_quote="20 percent",
            rationale="Update.",
        )
        confirmation = BoundaryConfirm(
            bc_id="L1",
            field="selector.span",
            source_quote="yes",
            rationale="Confirm.",
        )
        with self.assertRaisesRegex(
            ValidationError,
            "cannot update and confirm",
        ):
            BoundaryPatch(
                updates=(update,),
                confirmations=(confirmation,),
            )

    def test_readiness_reports_partial_assumed_and_capability_fields(self):
        partial = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_load",
                    kind="load",
                    fields=(
                        field(
                            "load.kind",
                            "resultant_magnitude",
                            "total force 10 N",
                        ),
                        field(
                            "load.magnitude",
                            10,
                            "total force 10 N",
                        ),
                        field(
                            "load.unit",
                            "N",
                            "10 N",
                        ),
                        field(
                            "selector.edge",
                            "right",
                            "right edge",
                        ),
                        field(
                            "selector.kind",
                            "unspecified_extent",
                            "right edge",
                            basis="derived",
                        ),
                        field(
                            "load.distribution",
                            "uniform",
                            basis="assumption",
                        ),
                    ),
                ),
            )),
            user_message="Use a total force 10 N on the right edge.",
            turn_number=1,
        ).state
        readiness = assess_boundary_state(partial)
        item = readiness.conditions[0]
        self.assertFalse(readiness.ready)
        self.assertNotIn("selector.extent", item.missing_fields)
        self.assertIn("load.direction", item.missing_fields)
        self.assertEqual(
            item.unconfirmed_fields,
            ("load.distribution",),
        )

        roller = merge_boundary_patch(
            BoundaryDraftState(),
            BoundaryPatch(creates=(
                BoundaryCreate(
                    local_ref="new_roller",
                    kind="support",
                    fields=(
                        field(
                            "support.kind",
                            "roller_normal",
                            "roller on bottom",
                        ),
                        field(
                            "selector.kind",
                            "whole_edge",
                            "bottom",
                        ),
                        field(
                            "selector.edge",
                            "bottom",
                            "bottom",
                        ),
                    ),
                ),
            )),
            user_message="Use a roller on bottom.",
            turn_number=1,
        ).state
        item = assess_boundary_state(roller).conditions[0]
        self.assertEqual(item.capability_limits, ())
        self.assertTrue(item.ready)


class BoundaryDraftIntegrationTests(unittest.TestCase):
    def test_formulation_turn_merges_optional_bc_patch(self):
        user = "clamp the left edge"
        turn = FormulationTurn(
            assistant_message="I retained support S1.",
            boundary_patch=BoundaryPatch(
                creates=(create_support(user),)
            ),
        )
        result = merge_formulation_turn(
            ProblemDraft(),
            turn,
            user_message=user,
            turn_number=1,
        )
        self.assertEqual(result.issues, ())
        self.assertEqual(
            result.draft.boundary_state.conditions[0].bc_id,
            "S1",
        )
        self.assertEqual(
            result.boundary_merge.accepted[0].local_ref,
            "new_support",
        )
        # Package 4 makes complete first-class BC entities authoritative.
        readiness = assess_draft(result.draft)
        self.assertNotIn("supports", readiness.missing_fields)
        self.assertIn("external_load", readiness.missing_fields)

    def test_empty_boundary_patch_preserves_existing_live_behavior(self):
        result = merge_formulation_turn(
            ProblemDraft(),
            FormulationTurn(assistant_message="No BC changes."),
            user_message="Continue.",
            turn_number=1,
        )
        self.assertEqual(result.draft.boundary_state, BoundaryDraftState())
        self.assertEqual(result.boundary_merge.accepted, ())
        self.assertEqual(result.boundary_merge.issues, ())

    def test_legacy_edge_segment_facts_migrate_with_provenance(self):
        draft = ProblemDraft(
            facts=(
                DraftFact(
                    path="support_edges",
                    value=["left"],
                    basis="derived",
                    source_turn=1,
                    source_quote="Fix its left side",
                    rationale="Relative full-edge clamp.",
                ),
                DraftFact(
                    path="tractions",
                    value=[{
                        "edge_segment": {
                            "edge": "right",
                            "center_fraction": 0.5,
                            "span_fraction": 0.1,
                        },
                        "vector": [0, -1],
                    }],
                    basis="derived",
                    source_turn=2,
                    source_quote="middle ten percent of the right edge",
                    rationale="Relative distributed traction.",
                ),
            ),
            turn_count=2,
        )
        migrated = migrate_legacy_boundary_facts(draft)
        self.assertEqual(
            [item.bc_id for item in migrated.boundary_state.conditions],
            ["S1", "L1"],
        )
        support = migrated.boundary_state.condition("S1")
        load = migrated.boundary_state.condition("L1")
        self.assertEqual(support.fact("selector.edge").value, "left")
        self.assertEqual(load.fact("selector.kind").value, "centered_fraction")
        self.assertEqual(load.fact("selector.span").value, 0.1)
        self.assertEqual(
            load.fact("selector.span").source_quote,
            "middle ten percent of the right edge",
        )
        self.assertEqual(migrated.values(), draft.values())

    def test_legacy_region_facts_migrate_to_expert_region(self):
        draft = ProblemDraft(
            facts=(
                DraftFact(
                    path="supports",
                    value=[{
                        "region": {
                            "op": "plane",
                            "axis": "x",
                            "value": 0,
                        }
                    }],
                    basis="explicit",
                    source_turn=1,
                    source_quote="x=0 is fixed",
                    rationale="Absolute support.",
                ),
            ),
            turn_count=1,
        )
        migrated = migrate_legacy_boundary_facts(draft)
        support = migrated.boundary_state.condition("S1")
        self.assertEqual(support.fact("selector.kind").value, "expert_region")
        self.assertEqual(
            support.fact("selector.region").value["op"],
            "plane",
        )
        with self.assertRaisesRegex(ValueError, "already populated"):
            migrate_legacy_boundary_facts(migrated)


if __name__ == "__main__":
    unittest.main()
