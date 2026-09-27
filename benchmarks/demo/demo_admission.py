#!/usr/bin/env python3
"""Interview demo 5: the overload incident, as FROZEN evidence
(docs/interview-demo.md). The real A/B takes ~55 min of GPU — an interview
shows the saved raw artifacts instead of rerunning them.

Run A (unbounded): 16 requests, 8K prompt, 8 concurrent, mlx-lm 9B-4bit
    on the M5. Every request hung to its 900 s client timeout while the
    server answered /health with 200 the whole time. 0/16 delivered.
Run B (admission ON): the identical workload, token/work-budget admission
    at the gateway. 16/16 delivered.

This reruns nothing; it prints the frozen summary and the tails of the two
committed console transcripts.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
DOCS = REPO / "docs" / "results"

HEADER = f"""\
overload incident — 8K prompt x c8, 16 requests, out=256
mlx-community/Qwen3.5-9B-4bit on Apple M5 (24 GB), via the gateway

  run A  unbounded          0/16 delivered   wall 1800 s  (silent: /health 200 throughout)
  run B  admission ON      16/16 delivered   ITL p50 ~60 ms

mechanism: every request is priced (estimated prefill + decode ms from the
calibrated service model); requests admit only while active estimated work
fits the budget; overflow queues (interactive-first) and answers
429 + Retry-After past the overload ceiling. Cancellation releases budget.
"""


def tail(path: Path, n: int = 14) -> str:
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def main() -> int:
    print(HEADER)
    for name, label in [("run-A-console.txt", "run A — unbounded (frozen console)"),
                        ("run-B-console.txt", "run B — admission ON (frozen console)")]:
        p = DOCS / name
        if not p.exists():
            print(f"missing artifact: {p}", file=sys.stderr)
            return 1
        print(f"--- {label} ({name}, tail) ---")
        print(tail(p))
        print()
    print("""\
full study:      docs/results/admission-saturation.md
raw artifacts:   benchmarks/results/admission-saturation-20260924-214429/ (gitignored)
reproduce:       make experiment-admission-saturation  (~55 min, real GPU — not for interviews)
live companion:  make failcheck  (deterministic 429 + Retry-After backpressure, ~2 min)
git tag:         phase-3-admission-control
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
