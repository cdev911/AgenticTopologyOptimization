from __future__ import annotations

import json
import os
import unittest
from unittest import mock

from agentic.intent import InterpretationEnvelope
from agentic.interpreter import (
    IntentInterpreter,
    InterpretationError,
    InterpreterConfig,
    OpenAIInterpretationEnvelope,
    PROMPT_VERSION,
    build_crewai_llm,
    config_from_environment,
    load_system_prompt,
    openai_response_format,
)


def ready_envelope() -> dict:
    return {
        "result": {
            "status": "ready",
            "intent": {
                "problem_type": "minimize_compliance",
                "domain": {"bounds": [[0, 0], [10, 4]]},
                "material": {"young_modulus": 1.0, "poisson_ratio": 0.3},
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


class FakeLLM:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.messages = []

    def call(self, messages):
        self.messages.append(messages)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class InterpreterTests(unittest.TestCase):
    def config(self, max_attempts=2):
        return InterpreterConfig(model="test-model", max_attempts=max_attempts)

    def test_ready_result_and_user_request_are_strictly_wrapped(self):
        envelope = InterpretationEnvelope.model_validate(ready_envelope())
        llm = FakeLLM([envelope])
        interpreter = IntentInterpreter(
            llm,
            config=self.config(),
            system_prompt="system contract",
        )

        result = interpreter.interpret("  optimize this beam  ")

        self.assertEqual(result.status, "ready")
        self.assertEqual(len(llm.messages), 1)
        self.assertEqual(llm.messages[0][0]["content"], "system contract")
        self.assertEqual(
            json.loads(llm.messages[0][1]["content"]),
            {"user_request": "optimize this beam"},
        )

    def test_clarification_dict_and_unsupported_json_are_accepted(self):
        clarification = {
            "result": {
                "status": "needs_clarification",
                "missing_fields": ["material.young_modulus"],
                "questions": ["What Young's modulus should be used?"],
            }
        }
        unsupported = json.dumps(
            {
                "result": {
                    "status": "unsupported",
                    "unsupported_features": ["3D domain"],
                    "explanation": "v1 supports rectangular 2D domains.",
                }
            }
        )

        clarification_result = IntentInterpreter(
            FakeLLM([clarification]),
            config=self.config(),
            system_prompt="contract",
        ).interpret("A beam with no material properties")
        unsupported_result = IntentInterpreter(
            FakeLLM([unsupported]),
            config=self.config(),
            system_prompt="contract",
        ).interpret("A 3D bracket")

        self.assertEqual(clarification_result.status, "needs_clarification")
        self.assertEqual(unsupported_result.status, "unsupported")

    def test_invalid_structured_output_retries_once_then_succeeds(self):
        llm = FakeLLM([{"result": {"status": "ready"}}, ready_envelope()])
        result = IntentInterpreter(
            llm,
            config=self.config(max_attempts=2),
            system_prompt="contract",
        ).interpret("complete request")

        self.assertEqual(result.status, "ready")
        self.assertEqual(len(llm.messages), 2)
        self.assertEqual(llm.messages[0], llm.messages[1])

    def test_retry_budget_is_bounded_and_provider_details_are_sanitized(self):
        llm = FakeLLM(
            [
                RuntimeError("secret provider response one"),
                RuntimeError("secret provider response two"),
                ready_envelope(),
            ]
        )
        interpreter = IntentInterpreter(
            llm,
            config=self.config(max_attempts=2),
            system_prompt="contract",
        )

        with self.assertRaises(InterpretationError) as caught:
            interpreter.interpret("request")

        self.assertEqual(caught.exception.attempts, 2)
        self.assertEqual(caught.exception.last_error_type, "RuntimeError")
        self.assertNotIn("secret provider response", str(caught.exception))
        self.assertEqual(len(llm.messages), 2)

    def test_blank_request_fails_without_calling_the_model(self):
        llm = FakeLLM([ready_envelope()])
        interpreter = IntentInterpreter(
            llm,
            config=self.config(),
            system_prompt="contract",
        )

        with self.assertRaises(ValueError):
            interpreter.interpret(" \n ")

        self.assertEqual(llm.messages, [])

    def test_prompt_documents_capability_and_default_authority(self):
        prompt = load_system_prompt()

        self.assertEqual(PROMPT_VERSION, "intent-system-v2")
        self.assertIn("needs_clarification", prompt)
        self.assertIn("component-wise or roller supports", prompt)
        self.assertIn("Deterministic", prompt)
        self.assertIn("mesh.divisions", prompt)
        self.assertIn("centered 10% of the right edge", prompt)
        self.assertIn("center_fraction=0.5", prompt)
        self.assertIn("Never add paths", prompt)

    def test_environment_and_crewai_configuration_are_pinned(self):
        with mock.patch.dict(os.environ, {"OPENAI_MODEL": "chosen-model"}):
            config = config_from_environment()
        self.assertEqual(config.model, "chosen-model")

        with mock.patch("agentic.interpreter.LLM") as llm_factory:
            build_crewai_llm(config)

        kwargs = llm_factory.call_args.kwargs
        self.assertEqual(kwargs["model"], "openai/chosen-model")
        self.assertEqual(kwargs["reasoning_effort"], "low")
        self.assertEqual(kwargs["max_retries"], 0)
        self.assertIs(kwargs["response_format"], OpenAIInterpretationEnvelope)
        self.assertNotIn("temperature", kwargs)

    def test_openai_schema_normalizes_strict_arrays_and_objects(self):
        response_format = openai_response_format()
        missing_items = []
        missing_additional_properties = []
        prefix_items = []

        def walk(value, path=()):
            if isinstance(value, dict):
                if value.get("type") == "array" and "items" not in value:
                    missing_items.append(path)
                if (
                    value.get("type") == "object" or "properties" in value
                ) and value.get("additionalProperties") is not False:
                    missing_additional_properties.append(path)
                if "prefixItems" in value:
                    prefix_items.append(path)
                for key, child in value.items():
                    walk(child, (*path, key))
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, (*path, index))

        walk(response_format)

        self.assertEqual(missing_items, [])
        self.assertEqual(missing_additional_properties, [])
        self.assertEqual(prefix_items, [])

    def test_configuration_rejects_unbounded_or_unrecorded_settings(self):
        with self.assertRaises(ValueError):
            InterpreterConfig(model="", max_attempts=2)
        with self.assertRaises(ValueError):
            InterpreterConfig(model="model", max_attempts=0)
        with self.assertRaises(ValueError):
            InterpreterConfig(model="model", max_attempts=4)
        with self.assertRaises(ValueError):
            InterpreterConfig(model="model", reasoning_effort="high")


if __name__ == "__main__":
    unittest.main()
