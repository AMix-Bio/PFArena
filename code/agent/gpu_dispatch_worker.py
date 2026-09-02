"""Subprocess worker that executes one GPU-dispatched Biomni function call."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import pickle
import re
import subprocess
import sys
import traceback
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any


ROOT = Path(os.getenv("BIOMNI_PROJECT_ROOT", Path(__file__).resolve().parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "biomni") not in sys.path:
    sys.path.insert(0, str(ROOT / "biomni"))

try:
    from biomni_benchmark.multi_turn_agent.gpu_runtime_env import (  # noqa: E402
        apply_runtime_environment, inject_python_namespace, install_transformers_local_redirects)
except ImportError:
    from gpu_runtime_env import (apply_runtime_environment, inject_python_namespace,
                                  install_transformers_local_redirects)

DEFAULT_SESSION_MAX_BYTES = 256 * 1024 * 1024
DEFAULT_VALUE_MAX_BYTES = 32 * 1024 * 1024
DEFAULT_GPU_DISPATCH_JOBS_DIR = ROOT / "agent_artifacts" / "gpu_dispatch_jobs"
BLOCKED_SESSION_MODULE_PREFIXES = (
    "torch",
    "transformers",
    "tokenizers",
    "esm",
    "openfold",
    "accelerate",
)
BLOCKED_SESSION_TYPE_NAMES = (
    "Module",
    "Tensor",
    "Parameter",
    "Tokenizer",
    "Pipeline",
)
SKIPPED_SESSION_KEY_PREFIXES = (
    "BIOMNI_",
    "HF_",
    "HUGGINGFACE_",
    "TRANSFORMERS_",
)
FORWARDED_ENV_KEYS = {
    "BIOMNI_LLM",
    "BIOMNI_LLM_MODEL",
    "BIOMNI_SOURCE",
    "BIOMNI_CUSTOM_BASE_URL",
    "BIOMNI_CUSTOM_API_KEY",
    "BIOMNI_LLM_TIMEOUT_SECONDS",
    "BIOMNI_LLM_MAX_RETRIES",
    "BIOMNI_AGENT_ARTIFACT_DIR",
    "BIOMNI_ALPHAFOLD_ARTIFACT_DIR",
    "BIOMNI_GPU_DISPATCH_JOBS_DIR",
    "PYTHONPATH",
}


@contextmanager
def pushd(path: str | None):
    original = os.getcwd()
    if path and Path(path).exists():
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def _bind_arguments(function: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> inspect.BoundArguments | None:
    try:
        signature = inspect.signature(function)
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return bound
    except Exception:
        return None


def _coerce_gpu_arguments(function: Any, module_name: str, function_name: str, args: tuple[Any, ...], kwargs: dict[str, Any]):
    """Prefer the server-assigned GPU for functions whose defaults otherwise use CPU/GPU0."""
    gpu_id = os.getenv("BIOMNI_DISPATCH_GPU_ID")
    if gpu_id is None:
        return args, kwargs

    bound = _bind_arguments(function, args, kwargs)
    if bound is None:
        return args, kwargs

    if module_name == "biomni.tool.systems_biology" and function_name == "query_chatnt":
        if bound.arguments.get("device", -1) in {-1, None, "cpu"}:
            bound.arguments["device"] = 0

    if module_name == "biomni.tool.pharmacology" and function_name == "run_diffdock_with_smiles":
        if bound.arguments.get("use_gpu", True):
            bound.arguments["gpu_device"] = int(gpu_id)

    if module_name == "biomni.tool.genomics" and function_name == "generate_transcriptformer_embeddings":
        bound.arguments["num_gpus"] = 1

    return (), dict(bound.arguments)


def _is_pickleable(value: Any) -> bool:
    try:
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception:
        return False


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _apply_payload_env(payload: dict[str, Any]) -> None:
    forwarded = payload.get("env") or {}
    if not isinstance(forwarded, dict):
        return
    for key, value in forwarded.items():
        if key in FORWARDED_ENV_KEYS and value is not None:
            os.environ[key] = str(value)


def _payload_env_overlay(payload: dict[str, Any]) -> dict[str, str]:
    forwarded = payload.get("env") or {}
    if not isinstance(forwarded, dict):
        return {}
    return {
        key: str(value)
        for key, value in forwarded.items()
        if key in FORWARDED_ENV_KEYS and value is not None
    }


def _payload_cwd(payload: dict[str, Any]) -> str | None:
    cwd = payload.get("cwd")
    if cwd and Path(str(cwd)).exists():
        return str(cwd)
    forwarded = payload.get("env") or {}
    if isinstance(forwarded, dict):
        artifact_dir = forwarded.get("BIOMNI_AGENT_ARTIFACT_DIR")
        if artifact_dir and Path(str(artifact_dir)).exists():
            return str(artifact_dir)
    return None


def _blocked_type_reason(value: Any) -> str | None:
    value_type = type(value)
    module_name = getattr(value_type, "__module__", "") or ""
    type_name = getattr(value_type, "__name__", "") or ""

    if inspect.ismodule(value):
        return "module"
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isbuiltin(value):
        return "callable"
    if inspect.isclass(value):
        return "class"
    if module_name.startswith(BLOCKED_SESSION_MODULE_PREFIXES):
        return f"blocked_module:{module_name}"
    if any(token in type_name for token in BLOCKED_SESSION_TYPE_NAMES):
        return f"blocked_type:{type_name}"
    return None


def _container_block_reason(value: Any, *, depth: int = 3, max_items: int = 200) -> str | None:
    reason = _blocked_type_reason(value)
    if reason:
        return reason
    if depth <= 0:
        return None

    if isinstance(value, dict):
        for index, item in enumerate(value.values()):
            if index >= max_items:
                return None
            reason = _container_block_reason(item, depth=depth - 1, max_items=max_items)
            if reason:
                return f"contains:{reason}"
    elif isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            if index >= max_items:
                return None
            reason = _container_block_reason(item, depth=depth - 1, max_items=max_items)
            if reason:
                return f"contains:{reason}"
    return None


def _known_size_bytes(value: Any) -> int | None:
    nbytes = getattr(value, "nbytes", None)
    if isinstance(nbytes, int):
        return nbytes
    memory_usage = getattr(value, "memory_usage", None)
    if callable(memory_usage):
        try:
            usage = memory_usage(deep=True)
            if hasattr(usage, "sum"):
                return int(usage.sum())
            return int(usage)
        except Exception:
            return None
    return None


def _session_value_pickle(value: Any, *, max_bytes: int) -> tuple[bytes | None, str | None]:
    reason = _container_block_reason(value)
    if reason:
        return None, reason

    known_size = _known_size_bytes(value)
    if known_size is not None and known_size > max_bytes:
        return None, f"known_size>{max_bytes}"

    try:
        data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        return None, f"not_pickleable:{type(exc).__name__}"
    if len(data) > max_bytes:
        return None, f"pickle_size>{max_bytes}"
    return data, None


def _load_session_state(state_path: Path, *, max_bytes: int) -> tuple[dict[str, Any], str | None]:
    if not state_path.exists():
        return {}, None
    try:
        state_size = state_path.stat().st_size
    except OSError as exc:
        return {}, f"could not stat session state {state_path}: {exc}"
    if state_size > max_bytes:
        return {}, f"skipped loading oversized session state {state_path} ({state_size} bytes > {max_bytes})"
    try:
        with state_path.open("rb") as handle:
            loaded = pickle.load(handle)
        if isinstance(loaded, dict):
            return loaded, None
        return {}, f"ignored non-dict session state {state_path}"
    except Exception as exc:
        return {}, f"could not load session state {state_path}: {exc}"


def _save_session_state(namespace: dict[str, Any], state_path: Path) -> dict[str, Any]:
    max_value_bytes = _env_int("BIOMNI_GPU_SESSION_VALUE_MAX_BYTES", DEFAULT_VALUE_MAX_BYTES)
    persistable: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    total_pickle_bytes = 0

    for key, value in namespace.items():
        if key.startswith("__"):
            continue
        if key in {"BIOMNI_MODEL_PATHS"} or key.startswith(SKIPPED_SESSION_KEY_PREFIXES):
            continue
        data, reason = _session_value_pickle(value, max_bytes=max_value_bytes)
        if data is None:
            skipped[key] = reason or "filtered"
            continue
        total_pickle_bytes += len(data)
        persistable[key] = value

    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("wb") as handle:
        pickle.dump(persistable, handle, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(state_path)

    metadata = {
        "state_path": str(state_path),
        "saved_keys": sorted(persistable),
        "skipped": skipped,
        "max_value_bytes": max_value_bytes,
        "estimated_value_pickle_bytes": total_pickle_bytes,
        "state_file_bytes": state_path.stat().st_size if state_path.exists() else 0,
    }
    with state_path.with_suffix(".meta.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return metadata


def _run_python_code(payload: dict[str, Any]) -> str:
    apply_runtime_environment()
    _apply_payload_env(payload)
    apply_runtime_environment()
    install_transformers_local_redirects()

    session_id = payload.get("session_id") or "default"
    jobs_dir = Path(os.getenv("BIOMNI_GPU_DISPATCH_JOBS_DIR", str(DEFAULT_GPU_DISPATCH_JOBS_DIR)))
    session_dir = jobs_dir / "_sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    state_path = session_dir / f"{session_id}.pkl"
    max_session_bytes = _env_int("BIOMNI_GPU_SESSION_MAX_BYTES", DEFAULT_SESSION_MAX_BYTES)
    namespace, load_warning = _load_session_state(state_path, max_bytes=max_session_bytes)
    inject_python_namespace(namespace)

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_buffer = StringIO()
    stderr_buffer = StringIO()
    sys.stdout = stdout_buffer
    sys.stderr = stderr_buffer
    try:
        if load_warning:
            print(f"[gpu-dispatch] {load_warning}")
        exec(str(payload.get("code", "")).strip("```").strip(), namespace)
    except Exception as exc:
        print(f"Error: {exc}", file=stderr_buffer)
        traceback.print_exc(file=stderr_buffer)
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

    session_metadata = _save_session_state(namespace, state_path)
    skipped_count = len(session_metadata.get("skipped", {}))
    if skipped_count:
        print(
            f"[gpu-dispatch] session saved {len(session_metadata.get('saved_keys', []))} keys; "
            f"skipped {skipped_count} large/nonportable keys; "
            f"state_file_bytes={session_metadata.get('state_file_bytes')}",
            file=stderr_buffer,
        )

    output = stdout_buffer.getvalue()
    errors = stderr_buffer.getvalue()
    if errors:
        output = output + ("\n" if output else "") + errors
    return output


def _run_bash_code(payload: dict[str, Any]) -> str:
    apply_runtime_environment()
    _apply_payload_env(payload)
    apply_runtime_environment()
    install_transformers_local_redirects()

    session_id = payload.get("session_id") or "default"
    jobs_dir = Path(os.getenv("BIOMNI_GPU_DISPATCH_JOBS_DIR", str(DEFAULT_GPU_DISPATCH_JOBS_DIR)))
    script_dir = jobs_dir / "_bash_scripts"
    script_dir.mkdir(parents=True, exist_ok=True)
    script_path = script_dir / f"{session_id}_{payload.get('job_id', 'job')}.sh"
    script = str(payload.get("code", "")).strip("` \n\t")
    script = re.sub(r"^#!BASH|^# Bash script|^#!CLI", "", script, count=1).strip()
    script_path.write_text(script + "\n", encoding="utf-8")

    env = os.environ.copy()
    env.update(_payload_env_overlay(payload))
    env.setdefault("PYTHONNOUSERSITE", "1")
    env["BIOMNI_GPU_PYTHON_EXECUTABLE"] = sys.executable
    python_bin_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = python_bin_dir + os.pathsep + env.get("PATH", "")
    timeout_seconds = _env_int("BIOMNI_GPU_BASH_TIMEOUT_SECONDS", int(payload.get("timeout_seconds") or 14_400))
    try:
        completed = subprocess.run(
            ["bash", str(script_path)],
            cwd=os.getcwd(),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        errors = exc.stderr or ""
        return (
            f"Error running Bash script: exceeded timeout after {timeout_seconds} seconds\n"
            + (output or "")
            + (("\n" if output and errors else "") + errors if errors else "")
        )

    output = completed.stdout or ""
    errors = completed.stderr or ""
    if completed.returncode != 0:
        prefix = f"Error running Bash script (exit code {completed.returncode}):\n"
        return prefix + output + (("\n" if output and errors else "") + errors if errors else "")
    if errors:
        return output + ("\n" if output else "") + errors
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Biomni GPU dispatch function call.")
    parser.add_argument("--payload", required=True)
    parser.add_argument("--status", required=True)
    args = parser.parse_args()

    payload_path = Path(args.payload)
    status_path = Path(args.status)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    os.environ["BIOMNI_GPU_DISPATCH_ROLE"] = "server"

    try:
        with payload_path.open("rb") as handle:
            payload = pickle.load(handle)
        _apply_payload_env(payload)
        apply_runtime_environment()
        install_transformers_local_redirects()
        module_name = payload["module_name"]
        function_name = payload["function_name"]

        with pushd(_payload_cwd(payload)):
            if module_name == "__python_repl__" and function_name == "run_python_code":
                if str(payload.get("language") or "python").lower() == "bash":
                    result = _run_bash_code(payload)
                else:
                    result = _run_python_code(payload)
            else:
                module = importlib.import_module(module_name)
                function = getattr(module, function_name)
                call_args = tuple(payload.get("args", ()))
                call_kwargs = dict(payload.get("kwargs", {}))
                call_args, call_kwargs = _coerce_gpu_arguments(function, module_name, function_name, call_args, call_kwargs)
                result = function(*call_args, **call_kwargs)

        result_path = Path(payload["result_path"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with result_path.open("wb") as handle:
            pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)

        status = {
            "status": "success",
            "result_path": str(result_path),
        }
    except BaseException as exc:
        status = {
            "status": "error",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    with status_path.open("w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
    return 0 if status["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
