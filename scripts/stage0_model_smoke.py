"""Small billed Stage 0 check for CrewAI structured output.

Run inside the project container:

    docker compose run --rm -T fenitop python scripts/stage0_model_smoke.py
"""

from __future__ import annotations

import os
from typing import Literal

from crewai import LLM
from pydantic import BaseModel, ConfigDict


class SmokeResult(BaseModel):
    """Deliberately tiny schema used only to verify the model boundary."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    answer: Literal[4]


def main() -> None:
    model = os.environ.get("OPENAI_MODEL", "gpt-5.6-terra")
    llm = LLM(
        model=f"openai/{model}",
        response_format=SmokeResult,
        reasoning_effort="low",
        timeout=60,
    )
    raw = llm.call(
        [
            {
                "role": "system",
                "content": "Return the requested structured result only.",
            },
            {
                "role": "user",
                "content": "Set status to ok and answer to two plus two.",
            },
        ]
    )
    result = (
        raw
        if isinstance(raw, SmokeResult)
        else SmokeResult.model_validate_json(raw)
    )
    print(
        "crewai_structured_output=ok "
        f"model={model} status={result.status} answer={result.answer}"
    )


if __name__ == "__main__":
    main()
