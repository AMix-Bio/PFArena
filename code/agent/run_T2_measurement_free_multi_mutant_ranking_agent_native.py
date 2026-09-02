#!/usr/bin/env python3
"""Run T2 with one autonomous Biomni A1 session per measurement-free multi-mutant set."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent_native_common import main_for_task


if __name__ == "__main__":
    raise SystemExit(main_for_task("T2_measurement_free_multi_mutant_ranking"))
