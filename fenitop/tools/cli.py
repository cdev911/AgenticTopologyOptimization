"""Shared --input/--output JSON CLI plumbing for the three agent tools.

Each tool's own module defines main() and calls run_cli(its_tool_fn, ...) so
this module never needs to import dolfinx-dependent tool modules itself.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict


def run_cli(tool_fn: Callable[[Dict[str, Any]], Dict[str, Any]], description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--input", default=None,
                         help="Path to a JSON request file. Defaults to stdin.")
    parser.add_argument("--output", default=None,
                         help="Path to write the JSON response. Defaults to stdout.")
    args = parser.parse_args()

    if args.input is not None:
        with open(args.input, "r", encoding="utf-8") as handle:
            request = json.load(handle)
    else:
        request = json.load(sys.stdin)

    response = tool_fn(request)
    text = json.dumps(response, indent=2)

    if args.output is not None:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)

    return 0 if response.get("status") == "ok" else 1
