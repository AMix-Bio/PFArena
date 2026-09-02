#!/usr/bin/env python3
"""Portable entry point for the T1--T4 Biomni benchmark tasks."""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parent
for import_root in (HERE, CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))
from router import DATASET_INPUTS
from agent_native_common import main_for_task

TASKS = tuple(task_name for task_name, _ in DATASET_INPUTS.values())

def main() -> int:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--provider", choices=("deepseek", "gpt", "claude"))
    known, rest = p.parse_known_args()
    if known.provider:
        presets = {
            "deepseek": ("deepseek-chat", "DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"),
            "gpt": ("gpt-4o", "OPENAI_BASE_URL", "OPENAI_API_KEY"),
            "claude": ("claude-3-5-sonnet-latest", "CLAUDE_BASE_URL", "CLAUDE_API_KEY"),
        }
        model, url_env, key_env = presets[known.provider]
        os.environ.setdefault("BIOMNI_AGENT_LLM", model)
        os.environ.setdefault("BIOMNI_AGENT_BASE_URL", os.getenv(url_env, ""))
        os.environ.setdefault("BIOMNI_AGENT_API_KEY_ENV", key_env)
    sys.argv = [sys.argv[0], *rest]
    return main_for_task(known.task)

if __name__ == "__main__":
    raise SystemExit(main())
