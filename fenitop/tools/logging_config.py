"""Shared logging setup for the agent-tool layer.

Each tool module gets its own named logger via get_logger(__name__), so log
records are attributable to a specific tool, but the format/handler is
configured once, here.

Deliberately logs to stderr, never stdout: fenitop/tools/cli.py prints each
tool's JSON response to stdout, and a log line interleaved into that stream
would corrupt it for anything parsing the CLI's output. The MCP server
(stdio transport) has the same constraint for the same reason.
"""
import logging
import sys

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            stream=sys.stderr,
        )
        _CONFIGURED = True
    return logging.getLogger(name)
