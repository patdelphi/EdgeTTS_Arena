from __future__ import annotations

import json
import sys
from typing import Any

from edgetts_arena.core.worker_runtime import run_isolated_model, run_isolated_repeated_model

RESULT_PREFIX = "__EDGETTS_ARENA_RESULT__="


def _emit(payload: dict[str, Any]) -> None:
    print(f"{RESULT_PREFIX}{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1 or args[0] not in {"single", "repeated"}:
        print("usage: python -m edgetts_arena.core.external_worker [single|repeated]", file=sys.stderr)
        return 2

    try:
        task = json.load(sys.stdin)
        if not isinstance(task, dict):
            raise ValueError("worker task must be a JSON object")
    except Exception as exc:
        print(f"invalid worker task: {exc}", file=sys.stderr)
        return 2

    payload = run_isolated_model(task) if args[0] == "single" else run_isolated_repeated_model(task)
    _emit(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
