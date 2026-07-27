"""Transport-clean JSON CLI plumbing shared by all public tools."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

from fenitop.tools.contracts import (
    AnalyzeResultsResponse,
    RunTopoptResponse,
    ValidateConfigResponse,
)
from fenitop.tools.schema import FieldError, error_envelope

logger = logging.getLogger(__name__)
_RESPONSE_MODELS = {
    "validate_config": ValidateConfigResponse,
    "run_topopt": RunTopoptResponse,
    "analyze_results": AnalyzeResultsResponse,
}


def _transport_error(tool_name: str, code: str, message: str) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if tool_name == "validate_config":
        payload.update(
            checked={"structural": False, "resource": False, "geometry": False},
            normalized_config=None,
        )
    response = error_envelope(
        tool_name,
        [FieldError("<root>", code, message)],
        stage="transport",
        **payload,
    )
    _RESPONSE_MODELS[tool_name].model_validate(response)
    return response


def _write_response(response: dict[str, Any], output: str | None) -> None:
    text = json.dumps(response, indent=2, allow_nan=False)
    if output is None:
        sys.stdout.write(text)
        sys.stdout.write("\n")
        sys.stdout.flush()
        return
    Path(output).write_text(text + "\n", encoding="utf-8")


def run_cli(
    tool_fn: Callable[[dict[str, Any]], dict[str, Any]],
    description: str,
    *,
    tool_name: str,
) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input", default=None, help="Path to a JSON request file. Defaults to stdin."
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON response. Defaults to stdout.",
    )
    args = parser.parse_args()

    try:
        if args.input is None:
            request = json.load(sys.stdin)
        else:
            with Path(args.input).open("r", encoding="utf-8") as handle:
                request = json.load(handle)
    except json.JSONDecodeError as exc:
        response = _transport_error(
            tool_name,
            "malformed_json",
            f"Request is not valid JSON: line {exc.lineno}, column {exc.colno}.",
        )
        transport_failure = True
    except (OSError, UnicodeError) as exc:
        response = _transport_error(
            tool_name,
            "input_read_failed",
            f"Could not read the JSON request: {exc}",
        )
        transport_failure = True
    else:
        transport_failure = False
        try:
            response = tool_fn(request)
            _RESPONSE_MODELS[tool_name].model_validate(response)
            json.dumps(response, allow_nan=False)
        except Exception:
            logger.exception("%s CLI: unexpected tool/serialization failure", tool_name)
            response = _transport_error(
                tool_name,
                "internal_error",
                "The tool failed unexpectedly; inspect local logs.",
            )

    try:
        _write_response(response, args.output)
    except (OSError, UnicodeError, TypeError, ValueError):
        logger.exception("%s CLI: could not write response", tool_name)
        return 2
    if transport_failure:
        return 2
    return 0 if response["status"] == "ok" else 1
