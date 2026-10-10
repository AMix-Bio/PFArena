#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 RESULT_DIRECTORY" >&2
    exit 1
fi

python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

files = sorted(Path(sys.argv[1]).glob("T*/evaluation/summary_metrics.json"))
if len(files) != 4:
    raise SystemExit(f"Expected 4 task summaries, found {len(files)}")

scores = []
for path in files:
    task = path.parent.parent.name
    metrics = json.loads(path.read_text())["metrics"]
    key = "NMS@40" if task.startswith("T1_") else "NMS@5"
    scores.append(metrics[key])
    print(f"{task}: {scores[-1]:.6f}")

print(f"Average NMS: {sum(scores) / 4:.4f}")
PY
