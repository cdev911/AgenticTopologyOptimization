"""Parse fenitop's <prefix>_run.log -- pure stdlib, no dolfinx/MPI needed.

topopt.py's FlushFileHandler (topopt.py's _configure_logger) flushes every
line immediately, and each "history {...}" line is only written after that
iteration's FEM solve/sensitivity/update calls already succeeded. That makes
this file durably consistent up to the last completed iteration even if the
process later dies mid-iteration -- log-tailing is a reliable way to recover
"last known good" state, not a hack.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Union

_HISTORY_RE = re.compile(r"history (\{.*\})\s*$")


def read_history(run_log_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Return every "history {...}" JSON record in the log, in file order.

    Missing file returns an empty list rather than raising, since Tool 2's
    failure path may call this before any log line was ever written.
    """
    path = Path(run_log_path)
    if not path.is_file():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = _HISTORY_RE.search(line)
            if not match:
                continue
            try:
                records.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                continue
    return records


def count_warnings(run_log_path: Union[str, Path], marker: str) -> int:
    """Count log lines containing `marker` as a plain substring."""
    path = Path(run_log_path)
    if not path.is_file():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if marker in line:
                count += 1
    return count
