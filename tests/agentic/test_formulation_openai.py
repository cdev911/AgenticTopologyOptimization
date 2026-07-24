from __future__ import annotations

import json
import os
import unittest
from types import SimpleNamespace
from unittest import mock

from agentic.compiler import compile_formulation_draft
from agentic.formulation import (
    ConversationFormulator,
    ConversationMessage,
    DraftUpdate,
    FormulationModelState,
    FormulationRepair,
    FormulationRequest,
    FormulationTurn,
    PatchIssue,
    ProblemDraft,
)
from agentic.formulation_openai import (
    ADAPTER_ID,
    DEFAULT_MODEL,
    PROMPT_VERSION,
    FormulationAPIError,
    OpenAIBoundaryConfirm,
    OpenAIBoundaryCreate,
    OpenAIBoundaryFieldInput,
    OpenAIBoundaryPatch,
    OpenAIBoundaryUpdate,
    OpenAIDraftUpdate,
    OpenAIFormulationConfig,
    OpenAIFormulationTurn,
    OpenAIResponsesFormulationAgent,
    build_openai_client,
    config_from_environment,
    load_system_prompt,
)
from fenitop.tools.validate_config import validate_config_tool


class FakeResponses:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, outcomes):
        self.responses = FakeResponses(outcomes)


class MissingContinuationError(RuntimeError):
    status_code = 404


def model_response(response_id="resp-1", turn=None):
    return SimpleNamespace(
        id=response_id,
        status="completed",
        output=(),
        output_parsed=turn
        or FormulationTurn(
            assistant_message="I need more information.",
            questions=("What material should be used?",),
        ),
    )


def request(*, model_state=None, repair=None):
    return FormulationRequest(
        turn_number=2,
        user_message="Use Poisson ratio 0.3.",
        draft=ProblemDraft(turn_count=1),
        history=(
            ConversationMessage(
                turn=1,
                role="user",
                content="Make a rectangular cantilever.",
            ),
            ConversationMessage(
                turn=1,
                role="assistant",
                content="What are its dimensions?",
            ),
        ),
        model_state=model_state,
        repair=repair,
    )


def empty_boundary_patch():
    return OpenAIBoundaryPatch(
        creates=(),
        updates=(),
        deletes=(),
        confirmations=(),
    )


def openai_update(path, value, quote, *, basis="explicit"):
    return OpenAIDraftUpdate(
        path=path,
        value_json=json.dumps(value, separators=(",", ":")),
        basis=basis,
        source_quote=quote,
        rationale="Captured from the current request.",
    )


def openai_boundary_field(field, value, quote, *, basis="explicit"):
    return OpenAIBoundaryFieldInput(
        field=field,
        value_json=json.dumps(value, separators=(",", ":")),
        basis=basis,
        source_quote=quote,
        rationale="Captured from the current request.",
    )


class OpenAIFormulationAgentTests(unittest.TestCase):
    def config(self):
        return OpenAIFormulationConfig(
            model="test-model",
            reasoning_effort="high",
            timeout_seconds=45,
            max_output_tokens=2048,
            safety_identifier="test-user",
        )

    def test_first_turn_uses_responses_structured_output_and_explicit_policy(self):
        client = FakeClient([model_response()])
        agent = OpenAIResponsesFormulationAgent(
            client,
            config=self.config(),
            system_prompt="formulation contract",
        )

        result = agent.formulate(request())

        self.assertEqual(result.model_state.adapter, ADAPTER_ID)
        self.assertEqual(result.model_state.continuation_id, "resp-1")
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertEqual(call["instructions"], "formulation contract")
        self.assertIs(call["text_format"], OpenAIFormulationTurn)
        self.assertEqual(
            call["reasoning"],
            {"effort": "high", "context": "all_turns"},
        )
        self.assertTrue(call["store"])
        self.assertEqual(call["max_output_tokens"], 2048)
        self.assertEqual(call["safety_identifier"], "test-user")
        self.assertEqual(call["timeout"], 45)
        self.assertNotIn("temperature", call)
        self.assertNotIn("tools", call)
        self.assertNotIn("previous_response_id", call)
        payload = json.loads(call["input"][0]["content"])
        self.assertEqual(payload["user_message"], "Use Poisson ratio 0.3.")
        self.assertEqual(len(payload["history"]), 2)
        self.assertEqual(payload["boundary_catalog"], [])
        self.assertEqual(payload["pending_boundary_confirmations"], [])
        self.assertFalse(payload["context_recovery"])

    def test_continuation_uses_previous_response_and_canonical_draft(self):
        client = FakeClient([model_response("resp-2")])
        agent = OpenAIResponsesFormulationAgent(
            client,
            config=self.config(),
            system_prompt="contract",
        )
        state = FormulationModelState(
            adapter=ADAPTER_ID,
            continuation_id="resp-1",
        )

        result = agent.formulate(request(model_state=state))

        call = client.responses.calls[0]
        self.assertEqual(call["previous_response_id"], "resp-1")
        payload = json.loads(call["input"][0]["content"])
        self.assertEqual(payload["history"], [])
        self.assertEqual(payload["canonical_draft"]["turn_count"], 1)
        self.assertEqual(result.model_state.continuation_id, "resp-2")

    def test_repair_feedback_is_sent_as_a_complete_typed_payload(self):
        rejected = FormulationTurn(
            assistant_message="I recorded the value.",
            updates=(
                DraftUpdate(
                    path="material.poisson_ratio",
                    value=0.8,
                    basis="explicit",
                    source_quote="Poisson ratio 0.3",
                    rationale="Material value.",
                ),
            ),
        )
        repair = FormulationRepair(
            attempt=2,
            rejected_turn=rejected,
            issues=(
                PatchIssue(
                    code="invalid_value",
                    path="material.poisson_ratio",
                    message="must be less than 0.5",
                ),
            ),
        )
        client = FakeClient([model_response()])
        agent = OpenAIResponsesFormulationAgent(
            client,
            config=self.config(),
            system_prompt="contract",
        )

        agent.formulate(request(repair=repair))

        payload = json.loads(
            client.responses.calls[0]["input"][0]["content"]
        )
        self.assertEqual(payload["repair_feedback"]["attempt"], 2)
        self.assertEqual(
            payload["repair_feedback"]["issues"][0]["code"],
            "invalid_value",
        )
        self.assertEqual(
            payload["repair_feedback"]["rejected_turn"]["updates"][0][
                "value"
            ],
            0.8,
        )

    def test_missing_continuation_replays_full_application_context_once(self):
        client = FakeClient(
            [
                MissingContinuationError("expired provider state"),
                model_response("resp-recovered"),
            ]
        )
        agent = OpenAIResponsesFormulationAgent(
            client,
            config=self.config(),
            system_prompt="contract",
        )
        state = FormulationModelState(
            adapter=ADAPTER_ID,
            continuation_id="expired",
        )

        result = agent.formulate(request(model_state=state))

        self.assertEqual(len(client.responses.calls), 2)
        self.assertEqual(
            client.responses.calls[0]["previous_response_id"],
            "expired",
        )
        recovery = client.responses.calls[1]
        self.assertNotIn("previous_response_id", recovery)
        payload = json.loads(recovery["input"][0]["content"])
        self.assertTrue(payload["context_recovery"])
        self.assertEqual(len(payload["history"]), 2)
        self.assertEqual(
            result.model_state.continuation_id,
            "resp-recovered",
        )

    def test_provider_errors_refusals_and_missing_output_are_sanitized(self):
        provider = OpenAIResponsesFormulationAgent(
            FakeClient([RuntimeError("secret response body")]),
            config=self.config(),
            system_prompt="contract",
        )
        with self.assertRaises(FormulationAPIError) as caught:
            provider.formulate(request())
        self.assertEqual(caught.exception.kind, "provider")
        self.assertNotIn("secret response body", str(caught.exception))

        refusal_item = SimpleNamespace(type="refusal", refusal="no")
        refusal_response = SimpleNamespace(
            id="resp-refusal",
            status="completed",
            output=(SimpleNamespace(content=(refusal_item,)),),
            output_parsed=None,
        )
        refusal = OpenAIResponsesFormulationAgent(
            FakeClient([refusal_response]),
            config=self.config(),
            system_prompt="contract",
        )
        with self.assertRaises(FormulationAPIError) as caught:
            refusal.formulate(request())
        self.assertEqual(caught.exception.kind, "refusal")

        incomplete_response = SimpleNamespace(
            id="resp-incomplete",
            status="incomplete",
            output=(),
            output_parsed=None,
        )
        incomplete = OpenAIResponsesFormulationAgent(
            FakeClient([incomplete_response]),
            config=self.config(),
            system_prompt="contract",
        )
        with self.assertRaises(FormulationAPIError) as caught:
            incomplete.formulate(request())
        self.assertEqual(caught.exception.kind, "invalid_response")
        self.assertEqual(
            caught.exception.provider_error_type,
            "IncompleteResponse",
        )

    def test_transport_decodes_compact_json_value_without_open_schema(self):
        transport = OpenAIFormulationTurn(
            assistant_message="I recorded the domain.",
            updates=(
                OpenAIDraftUpdate(
                    path="domain.bounds",
                    value_json="[[0,0],[10,5]]",
                    basis="derived",
                    source_quote="10 by 5 at the origin",
                    rationale="Converted dimensions to bounds.",
                ),
            ),
            clears=(),
            boundary_patch=empty_boundary_patch(),
            questions=(),
            declared_state="gathering",
            unsupported_features=(),
        )

        domain_turn = transport.to_domain()

        self.assertEqual(
            domain_turn.updates[0].value,
            [[0, 0], [10, 5]],
        )
        schema = json.dumps(OpenAIFormulationTurn.model_json_schema())
        self.assertNotIn('"JsonValue": {}', schema)
        self.assertIn("value_json", schema)

        invalid = transport.model_copy(
            update={
                "updates": (
                    transport.updates[0].model_copy(
                        update={"value_json": "not-json"}
                    ),
                )
            }
        )
        self.assertIsNone(invalid.to_domain().updates[0].value)

        duplicate = transport.model_copy(
            update={
                "updates": (
                    transport.updates[0],
                    transport.updates[0],
                )
            }
        )
        response = model_response()
        response.output_parsed = duplicate
        agent = OpenAIResponsesFormulationAgent(
            FakeClient([response]),
            config=self.config(),
            system_prompt="contract",
        )
        with self.assertRaises(FormulationAPIError) as caught:
            agent.formulate(request())
        self.assertEqual(caught.exception.kind, "invalid_response")
        self.assertEqual(
            caught.exception.provider_error_type,
            "ValidationError",
        )

    def test_transport_decodes_first_class_boundary_operations(self):
        transport = OpenAIFormulationTurn(
            assistant_message="I retained the support and refined L1.",
            updates=(),
            clears=(),
            boundary_patch=OpenAIBoundaryPatch(
                creates=(
                    OpenAIBoundaryCreate(
                        local_ref="new_support_1",
                        kind="support",
                        fields=(
                            OpenAIBoundaryFieldInput(
                                field="support.kind",
                                value_json='"fixed_all"',
                                basis="derived",
                                source_quote="clamp the left",
                                rationale="A clamp is a full-vector support.",
                            ),
                            OpenAIBoundaryFieldInput(
                                field="selector.kind",
                                value_json='"whole_edge"',
                                basis="derived",
                                source_quote="clamp the left",
                                rationale="The full edge was requested.",
                            ),
                            OpenAIBoundaryFieldInput(
                                field="selector.edge",
                                value_json='"left"',
                                basis="explicit",
                                source_quote="clamp the left",
                                rationale="The user named the left edge.",
                            ),
                        ),
                    ),
                ),
                updates=(
                    OpenAIBoundaryUpdate(
                        bc_id="L1",
                        field="load.magnitude",
                        value_json="10",
                        basis="explicit",
                        source_quote="10 N",
                        rationale="The user supplied the magnitude.",
                    ),
                ),
                deletes=(),
                confirmations=(
                    OpenAIBoundaryConfirm(
                        bc_id="L1",
                        field="load.distribution",
                        source_quote="on L1",
                        rationale="The user confirmed the proposed distribution.",
                    ),
                ),
            ),
            questions=(),
            declared_state="gathering",
            unsupported_features=(),
        )

        domain = transport.to_domain()

        self.assertEqual(
            domain.boundary_patch.creates[0].fields[0].value,
            "fixed_all",
        )
        self.assertEqual(
            domain.boundary_patch.updates[0].value,
            10,
        )
        self.assertEqual(
            domain.boundary_patch.confirmations[0].bc_id,
            "L1",
        )
        schema = json.dumps(OpenAIFormulationTurn.model_json_schema())
        self.assertNotIn('"supports"', schema)
        self.assertNotIn('"support_edges"', schema)
        self.assertNotIn('"tractions"', schema)
        self.assertIn('"boundary_patch"', schema)
        self.assertLess(len(schema), 30_000)

    def test_live_transport_reaches_first_class_compile_and_validation(self):
        user = (
            "Minimize compliance on a 10 by 4 mm rectangle at the origin with "
            "E 100 MPa, nu 0.3, units mm N MPa, the left edge clamped, a total "
            "20 N downward uniformly over the centered half of the right edge, "
            "and 40 percent material."
        )
        turn = OpenAIFormulationTurn(
            assistant_message=(
                "I retained a full clamp S1 and total resultant L1."
            ),
            updates=(
                openai_update(
                    "problem_type",
                    "minimize_compliance",
                    user,
                    basis="derived",
                ),
                openai_update(
                    "domain.bounds",
                    [[0, 0], [10, 4]],
                    user,
                    basis="derived",
                ),
                openai_update("material.young_modulus", 100, user),
                openai_update("material.poisson_ratio", 0.3, user),
                openai_update("units.length", "mm", user),
                openai_update("units.force", "N", user),
                openai_update("units.stress", "MPa", user),
                openai_update(
                    "volume_fraction",
                    0.4,
                    user,
                    basis="derived",
                ),
            ),
            clears=(),
            boundary_patch=OpenAIBoundaryPatch(
                creates=(
                    OpenAIBoundaryCreate(
                        local_ref="new_support_1",
                        kind="support",
                        fields=(
                            openai_boundary_field(
                                "support.kind",
                                "fixed_all",
                                user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "selector.kind",
                                "whole_edge",
                                user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "selector.edge",
                                "left",
                                user,
                            ),
                        ),
                    ),
                    OpenAIBoundaryCreate(
                        local_ref="new_load_1",
                        kind="load",
                        fields=(
                            openai_boundary_field(
                                "load.kind",
                                "resultant_magnitude",
                                user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "load.magnitude",
                                20,
                                user,
                            ),
                            openai_boundary_field(
                                "load.direction",
                                "down",
                                user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "load.unit",
                                "N",
                                user,
                            ),
                            openai_boundary_field(
                                "load.distribution",
                                "uniform",
                                user,
                            ),
                            openai_boundary_field(
                                "selector.kind",
                                "centered_fraction",
                                user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "selector.edge",
                                "right",
                                user,
                            ),
                            openai_boundary_field(
                                "selector.center",
                                0.5,
                                user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "selector.span",
                                0.5,
                                user,
                                basis="derived",
                            ),
                        ),
                    ),
                ),
                updates=(),
                deletes=(),
                confirmations=(),
            ),
            questions=(),
            declared_state="ready",
            unsupported_features=(),
        )
        agent = OpenAIResponsesFormulationAgent(
            FakeClient([model_response("resp-ready", turn)]),
            config=self.config(),
            system_prompt="contract",
        )

        step = ConversationFormulator(agent).start(user)

        self.assertEqual(step.session.status, "ready_for_review")
        self.assertEqual(
            [
                condition.bc_id
                for condition in step.session.draft.boundary_state.conditions
            ],
            ["S1", "L1"],
        )
        compilation = compile_formulation_draft(step.finalized_draft)
        self.assertEqual(
            compilation.config.fem.boundary_conditions[1].kind,
            "uniform_resultant",
        )
        validation = validate_config_tool({"config": compilation.config})
        self.assertEqual(validation["status"], "ok", validation["errors"])

    def test_boundary_assumption_is_visible_then_confirmed_by_stable_id(self):
        first_user = "Apply a total 10 N downward on the entire right edge."
        first = OpenAIFormulationTurn(
            assistant_message=(
                "I recorded L1 and propose distributing it uniformly; "
                "please confirm that assumption."
            ),
            updates=(),
            clears=(),
            boundary_patch=OpenAIBoundaryPatch(
                creates=(
                    OpenAIBoundaryCreate(
                        local_ref="new_load_1",
                        kind="load",
                        fields=(
                            openai_boundary_field(
                                "load.kind",
                                "resultant_magnitude",
                                first_user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "load.magnitude",
                                10,
                                first_user,
                            ),
                            openai_boundary_field(
                                "load.direction",
                                "down",
                                first_user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "load.unit",
                                "N",
                                first_user,
                            ),
                            OpenAIBoundaryFieldInput(
                                field="load.distribution",
                                value_json='"uniform"',
                                basis="assumption",
                                source_quote=None,
                                rationale=(
                                    "A total force on a finite edge needs a "
                                    "distribution."
                                ),
                            ),
                            openai_boundary_field(
                                "selector.kind",
                                "whole_edge",
                                first_user,
                                basis="derived",
                            ),
                            openai_boundary_field(
                                "selector.edge",
                                "right",
                                first_user,
                            ),
                        ),
                    ),
                ),
                updates=(),
                deletes=(),
                confirmations=(),
            ),
            questions=("Should L1 be distributed uniformly?",),
            declared_state="gathering",
            unsupported_features=(),
        )
        second = OpenAIFormulationTurn(
            assistant_message="Uniform distribution for L1 is confirmed.",
            updates=(),
            clears=(),
            boundary_patch=OpenAIBoundaryPatch(
                creates=(),
                updates=(),
                deletes=(),
                confirmations=(
                    OpenAIBoundaryConfirm(
                        bc_id="L1",
                        field="load.distribution",
                        source_quote="yes",
                        rationale="Confirmed the sole pending BC assumption.",
                    ),
                ),
            ),
            questions=(),
            declared_state="gathering",
            unsupported_features=(),
        )
        client = FakeClient([
            model_response("resp-assumed", first),
            model_response("resp-confirmed", second),
        ])
        formulator = ConversationFormulator(
            OpenAIResponsesFormulationAgent(
                client,
                config=self.config(),
                system_prompt="contract",
            )
        )

        initial = formulator.start(first_user)
        confirmed = formulator.advance(initial.session, "yes")

        pending_payload = json.loads(
            client.responses.calls[1]["input"][0]["content"]
        )
        self.assertEqual(
            pending_payload["pending_boundary_confirmations"][0]["bc_id"],
            "L1",
        )
        fact = confirmed.session.draft.boundary_state.condition("L1").fact(
            "load.distribution"
        )
        self.assertEqual(fact.basis, "confirmed")
        self.assertEqual(fact.value, "uniform")

    def test_malformed_inner_json_uses_bounded_deterministic_repair(self):
        malformed = OpenAIFormulationTurn(
            assistant_message="I recorded the material value.",
            updates=(
                OpenAIDraftUpdate(
                    path="material.poisson_ratio",
                    value_json="not-json",
                    basis="explicit",
                    source_quote="Poisson ratio 0.3",
                    rationale="Material value.",
                ),
            ),
            clears=(),
            boundary_patch=empty_boundary_patch(),
            questions=(),
            declared_state="gathering",
            unsupported_features=(),
        )
        corrected = malformed.model_copy(
            update={
                "assistant_message": "I corrected the material encoding.",
                "updates": (
                    malformed.updates[0].model_copy(
                        update={"value_json": "0.3"}
                    ),
                ),
            }
        )
        client = FakeClient(
            [
                model_response("resp-invalid", malformed),
                model_response("resp-corrected", corrected),
            ]
        )
        agent = OpenAIResponsesFormulationAgent(
            client,
            config=self.config(),
            system_prompt="contract",
        )

        result = ConversationFormulator(agent).start(
            "Use Poisson ratio 0.3."
        )

        self.assertEqual(len(client.responses.calls), 2)
        repair_payload = json.loads(
            client.responses.calls[1]["input"][0]["content"]
        )
        self.assertEqual(
            repair_payload["repair_feedback"]["issues"][0]["code"],
            "invalid_value",
        )
        self.assertEqual(
            result.session.draft.fact("material.poisson_ratio").value,
            0.3,
        )
        self.assertEqual(
            result.session.model_state.continuation_id,
            "resp-corrected",
        )

    def test_environment_defaults_to_quality_first_formulation_settings(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            config = config_from_environment()
        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.reasoning_effort, "medium")
        self.assertEqual(config.timeout_seconds, 90)
        self.assertEqual(config.max_output_tokens, 5000)

        with mock.patch.dict(
            os.environ,
            {
                "OPENAI_FORMULATION_MODEL": "chosen-model",
                "OPENAI_FORMULATION_REASONING_EFFORT": "xhigh",
                "OPENAI_FORMULATION_TIMEOUT_SECONDS": "120",
                "OPENAI_FORMULATION_MAX_OUTPUT_TOKENS": "6000",
            },
            clear=True,
        ):
            config = config_from_environment()
        self.assertEqual(config.model, "chosen-model")
        self.assertEqual(config.reasoning_effort, "xhigh")
        self.assertEqual(config.timeout_seconds, 120)
        self.assertEqual(config.max_output_tokens, 6000)

    def test_client_retries_are_disabled_and_prompt_is_versioned(self):
        config = self.config()
        with mock.patch(
            "agentic.formulation_openai.OpenAI"
        ) as client_factory:
            build_openai_client(config)
        self.assertEqual(
            client_factory.call_args.kwargs,
            {"timeout": 45, "max_retries": 0},
        )

        prompt = load_system_prompt()
        self.assertEqual(PROMPT_VERSION, "formulation-system-v3")
        self.assertIn("Collaborate with the user over as many turns", prompt)
        self.assertIn("repair_feedback", prompt)
        self.assertIn("source_quote", prompt)
        self.assertIn("value_json", prompt)
        self.assertIn("Never claim that", prompt)
        self.assertIn("a solve, validation, conversion", prompt)
        self.assertIn("boundary_patch", prompt)
        self.assertIn("total resultant", prompt)
        self.assertIn("highest-information", prompt)
        self.assertIn("nonzero prescribed displacement", prompt)

    def test_configuration_rejects_unbounded_values(self):
        with self.assertRaises(ValueError):
            OpenAIFormulationConfig(model="")
        with self.assertRaises(ValueError):
            OpenAIFormulationConfig(reasoning_effort="none")
        with self.assertRaises(ValueError):
            OpenAIFormulationConfig(timeout_seconds=0)
        with self.assertRaises(ValueError):
            OpenAIFormulationConfig(max_output_tokens=100)


if __name__ == "__main__":
    unittest.main()
