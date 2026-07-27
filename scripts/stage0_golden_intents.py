"""Run the Stage 0 intent-classification gate against candidate models.

This is an occasional billed evaluation, not a production interpreter prompt:

    docker compose run --rm -T fenitop \
        python scripts/stage0_golden_intents.py
"""

from __future__ import annotations

import argparse
import json
from typing import Literal

from crewai import LLM
from pydantic import BaseModel, ConfigDict, Field


class GoldenResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_id: Literal["supported_ready", "ambiguous", "unsupported"]
    status: Literal["ready", "needs_clarification", "unsupported"]
    reason: str = Field(min_length=1, max_length=240)


class GoldenBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[GoldenResult] = Field(min_length=3, max_length=3)


EXPECTED = {
    "supported_ready": "ready",
    "ambiguous": "needs_clarification",
    "unsupported": "unsupported",
}

SYSTEM_PROMPT = """\
Classify structural topology-optimization requests for this exact v1 capability.

Supported: rectangular 2D plane-strain compliance or compliant-mechanism problems;
unit thickness; distributed boundary traction; full-vector zero clamps; explicit
material, mesh, volume, filter, support, and load information.

Return ready only when the problem-defining physics is explicit or safely derived.
Return needs_clarification when a supported request omits or ambiguously describes
required physics. Return unsupported when it requests capabilities outside v1,
including 3D, component-wise/roller supports, nonzero prescribed displacement, or
other unsupported physics. Never invent missing values.

Classify every supplied scenario exactly once and preserve each scenario_id.
"""

SCENARIOS = [
    {
        "scenario_id": "supported_ready",
        "request": (
            "Minimize compliance of a 10 by 4 rectangular 2D plane-strain beam "
            "using a 40 by 16 mesh. Fully clamp the entire left edge and apply "
            "a distributed traction vector [0, -1] to the entire right edge. "
            "Use Young's modulus 1, Poisson ratio 0.3, volume fraction 0.4, "
            "and filter radius 0.3."
        ),
    },
    {
        "scenario_id": "ambiguous",
        "request": (
            "Optimize a 10 by 4 beam. Fix the left side, put a load on the "
            "right, and use 40 percent material."
        ),
    },
    {
        "scenario_id": "unsupported",
        "request": (
            "Optimize a three-dimensional cantilever with a roller support and "
            "a prescribed nonzero displacement."
        ),
    },
]


def evaluate(model: str) -> dict[str, object]:
    llm = LLM(
        model=f"openai/{model}",
        response_format=GoldenBatch,
        reasoning_effort="low",
        timeout=90,
    )
    raw = llm.call(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps({"scenarios": SCENARIOS}),
            },
        ]
    )
    batch = raw if isinstance(raw, GoldenBatch) else GoldenBatch.model_validate_json(raw)
    observed = {item.scenario_id: item.status for item in batch.results}
    passed = observed == EXPECTED
    return {
        "model": model,
        "passed": passed,
        "expected": EXPECTED,
        "observed": observed,
        "results": [item.model_dump(mode="json") for item in batch.results],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-5.6-terra", "gpt-5.6-luna"],
    )
    args = parser.parse_args()

    evaluations = [evaluate(model) for model in args.models]
    print(json.dumps({"evaluations": evaluations}, indent=2, sort_keys=True))
    if not all(result["passed"] for result in evaluations):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
