"""Client-side hooks for dispatching selected Biomni tools to a GPU worker."""

from __future__ import annotations

import functools
import importlib
import os
import pickle
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import requests


ROOT = Path(os.getenv("BIOMNI_PROJECT_ROOT", Path(__file__).resolve().parent))
for import_root in (ROOT, ROOT / "biomni"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

DEFAULT_GPU_DISPATCH_BASE_URL = os.getenv("BIOMNI_GPU_DISPATCH_BASE_URL", "http://127.0.0.1:25000")
DEFAULT_GPU_DISPATCH_JOBS_DIR = ROOT / "agent_artifacts" / "gpu_dispatch_jobs"
DEFAULT_GPU_DISPATCH_TIMEOUT_SECONDS = 14_400
DEFAULT_GPU_DISPATCH_POLL_SECONDS = 5

DEFAULT_GPU_DISPATCH_FUNCTIONS = (
    "biomni.tool.genomics.generate_gene_embeddings_with_ESM_models",
    "biomni.tool.genomics.generate_embeddings_with_state",
    "biomni.tool.genomics.generate_transcriptformer_embeddings",
    "biomni.tool.genomics.get_uce_embeddings_scRNA",
    "biomni.tool.systems_biology.query_chatnt",
    "biomni.tool.systems_biology.compare_protein_structures",
    "biomni.tool.pharmacology.run_diffdock_with_smiles",
    "biomni.tool.pharmacology.predict_binding_affinity_protein_1d_sequence",
    "biomni.tool.bioimaging.segment_with_nn_unet",
)
FORWARDED_ENV_KEYS = (
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
)

_PATCHED_FUNCTIONS: dict[str, Any] = {}
_ORIGINAL_RUN_PYTHON_REPL: Any | None = None
_ORIGINAL_RUN_BASH_SCRIPT: Any | None = None
_CODE_DISPATCH_CONFIG: dict[str, Any] = {}

HEAVY_CODE_PATTERNS = (
    r"\bimport\s+esm\b",
    r"\bfrom\s+esm\b",
    r"esm\.pretrained",
    r"\bimport\s+torch\b",
    r"\bfrom\s+torch\b",
    r"torch\.cuda",
    r"\bDiffDock\b",
    r"\bdiffdock\b",
    r"\bnnUNet\b",
    r"\bnnunet\b",
    r"\bcellpose\b",
    r"\bCellpose\b",
    r"\bopenmm\b",
    r"\bDeepPurpose\b",
    r"\bDTI\.model_pretrained\b",
    r"\bpredict_from_folder\b",
    r"\btransformers\s+import\s+pipeline\b",
    r"\bpipeline\s*\(",
)
LOCAL_PYTHON_CODE_PATTERNS = (
    r"\bfrom\s+biomni\.tool\.database\b",
    r"\bimport\s+biomni\.tool\.database\b",
    r"\bbiomni\.tool\.database\b",
    r"\bquery_uniprot\b",
    r"\bquery_alphafold\b",
    r"\bquery_interpro\b",
    r"\bquery_pubmed\b",
    r"\bquery_pmc\b",
    r"\brequests\.",
    r"\bimport\s+requests\b",
    r"\bfrom\s+requests\b",
    r"\bhttpx\.",
    r"\bimport\s+httpx\b",
    r"\burllib\.request\b",
    r"\bfrom\s+urllib\b",
    r"\bBio\.Entrez\b",
    r"\bEntrez\.",
    r"https?://",
)
LOCAL_BASH_CODE_PATTERNS = (
    r"\bcurl\b",
    r"\bwget\b",
    r"\bpython(?:3)?\b.*\b(?:requests|httpx|urllib|Bio\.Entrez|query_uniprot|query_pubmed|query_pmc|query_interpro)\b",
    r"https?://",
)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_function_list(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_GPU_DISPATCH_FUNCTIONS
    return tuple(item.strip() for item in value.split(",") if item.strip())


def build_job_id(function_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = function_name.replace(".", "_")
    return f"{safe_name}_{timestamp}_{uuid4().hex[:8]}"


def collect_forwarded_env() -> dict[str, str]:
    env = {key: os.environ[key] for key in FORWARDED_ENV_KEYS if os.environ.get(key)}
    env.setdefault("BIOMNI_GPU_DISPATCH_JOBS_DIR", str(DEFAULT_GPU_DISPATCH_JOBS_DIR))
    return env


class GPUDispatchClient:
    """Submit a Python function call to the remote GPU dispatcher."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        jobs_dir: str | Path | None = None,
        request_timeout_seconds: int = 30,
        poll_interval_seconds: int = DEFAULT_GPU_DISPATCH_POLL_SECONDS,
    ) -> None:
        self.base_url = (base_url or os.getenv("BIOMNI_GPU_DISPATCH_BASE_URL") or DEFAULT_GPU_DISPATCH_BASE_URL).rstrip(
            "/"
        )
        self.jobs_dir = Path(jobs_dir or os.getenv("BIOMNI_GPU_DISPATCH_JOBS_DIR") or DEFAULT_GPU_DISPATCH_JOBS_DIR)
        self.request_timeout_seconds = request_timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds

    def submit_function(
        self,
        *,
        module_name: str,
        function_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        timeout_seconds: int,
    ) -> Any:
        job_id = build_job_id(f"{module_name}.{function_name}")
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        payload_path = job_dir / "payload.pkl"
        result_path = job_dir / "result.pkl"

        payload = {
            "job_id": job_id,
            "module_name": module_name,
            "function_name": function_name,
            "args": args,
            "kwargs": kwargs,
            "cwd": os.getcwd(),
            "result_path": str(result_path),
            "env": collect_forwarded_env(),
        }
        with payload_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        response = requests.post(
            f"{self.base_url}/jobs",
            json={
                "job_id": job_id,
                "payload_path": str(payload_path),
                "timeout_seconds": int(timeout_seconds),
            },
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()

        status = self.wait_for_job(job_id, timeout_seconds=timeout_seconds + 300)
        if status.get("status") != "success":
            message = status.get("error_message") or f"GPU dispatch job {job_id} failed"
            raise RuntimeError(message)
        with result_path.open("rb") as handle:
            return pickle.load(handle)

    def submit_python_code(
        self,
        *,
        code: str,
        session_id: str,
        cwd: str,
        timeout_seconds: int,
        language: str = "python",
    ) -> str:
        job_id = build_job_id("python_repl")
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        payload_path = job_dir / "payload.pkl"
        result_path = job_dir / "result.pkl"

        payload = {
            "job_id": job_id,
            "module_name": "__python_repl__",
            "function_name": "run_python_code",
            "code": code,
            "language": language,
            "session_id": session_id,
            "cwd": cwd,
            "timeout_seconds": int(timeout_seconds),
            "result_path": str(result_path),
            "env": collect_forwarded_env(),
        }
        with payload_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

        response = requests.post(
            f"{self.base_url}/jobs",
            json={
                "job_id": job_id,
                "payload_path": str(payload_path),
                "timeout_seconds": int(timeout_seconds),
            },
            timeout=self.request_timeout_seconds,
        )
        response.raise_for_status()

        status = self.wait_for_job(job_id, timeout_seconds=timeout_seconds + 300)
        if status.get("status") != "success":
            message = status.get("error_message") or f"GPU dispatch job {job_id} failed"
            raise RuntimeError(message)
        with result_path.open("rb") as handle:
            return pickle.load(handle)

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        response = requests.get(f"{self.base_url}/jobs/{job_id}", timeout=self.request_timeout_seconds)
        response.raise_for_status()
        return response.json()

    def wait_for_job(self, job_id: str, *, timeout_seconds: int) -> dict[str, Any]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            status = self.get_job_status(job_id)
            if status.get("status") in {"success", "error", "timeout"}:
                return status
            time.sleep(self.poll_interval_seconds)
        raise TimeoutError(f"Timed out waiting for GPU dispatch job {job_id}")


def _make_remote_wrapper(
    original: Any,
    *,
    module_name: str,
    function_name: str,
    base_url: str,
    jobs_dir: str | Path,
    timeout_seconds: int,
) -> Any:
    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        client = GPUDispatchClient(base_url=base_url, jobs_dir=jobs_dir)
        print(
            f"[gpu-dispatch] forwarding {module_name}.{function_name} "
            f"to {client.base_url}",
            flush=True,
        )
        return client.submit_function(
            module_name=module_name,
            function_name=function_name,
            args=args,
            kwargs=kwargs,
            timeout_seconds=timeout_seconds,
        )

    wrapper._biomni_gpu_dispatch_wrapper = True  # type: ignore[attr-defined]
    wrapper._biomni_gpu_dispatch_original = original  # type: ignore[attr-defined]
    return wrapper


def should_dispatch_python_code(code: str) -> bool:
    return any(re.search(pattern, code, flags=re.IGNORECASE) for pattern in HEAVY_CODE_PATTERNS)


def should_prefer_local_python_code(code: str) -> bool:
    return any(re.search(pattern, code, flags=re.IGNORECASE) for pattern in LOCAL_PYTHON_CODE_PATTERNS)


def should_prefer_local_bash_code(script: str) -> bool:
    return any(re.search(pattern, script, flags=re.IGNORECASE | re.DOTALL) for pattern in LOCAL_BASH_CODE_PATTERNS)


def default_remote_execute_enabled() -> bool:
    value = os.getenv("BIOMNI_GPU_DISPATCH_DEFAULT_REMOTE_EXECUTE")
    if value is None:
        return True
    return value.strip().lower() in {"1", "true", "yes", "on"}


def should_run_python_locally(code: str) -> bool:
    """Keep stateless light snippets and network/database I/O on the dev machine."""
    stripped = code.strip("` \n\t")
    if not stripped:
        return True
    if should_prefer_local_python_code(stripped) and not should_dispatch_python_code(stripped):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if all(line.startswith("#") for line in lines):
        return True
    if len(lines) == 1 and lines[0] == "pass":
        return True
    if len(lines) <= 3 and all(
        re.fullmatch(r"print\s*\(\s*(?:[rubfRUBF]*(['\"]).*?\1|\d+(?:\.\d+)?|True|False|None)\s*\)", line)
        for line in lines
    ):
        return True
    return False


def should_run_bash_locally(script: str) -> bool:
    """Keep simple snippets and network/database I/O local; run compute-heavy shell remotely."""
    stripped = script.strip()
    if not stripped:
        return True
    if should_prefer_local_bash_code(stripped) and not should_dispatch_python_code(stripped):
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return True
    if len(lines) > 3:
        return False
    safe_command = re.compile(r"^(?:pwd|date|true|false|echo(?:\s+[^;&|`$<>]*)?)$")
    return all(safe_command.fullmatch(line) for line in lines)


def install_python_repl_dispatch_hook(
    *,
    enabled: bool,
    base_url: str | None,
    jobs_dir: str | Path | None,
    timeout_seconds: int,
    session_id: str,
) -> bool:
    """Route A1 execute blocks to the GPU worker by default."""
    global _ORIGINAL_RUN_BASH_SCRIPT, _ORIGINAL_RUN_PYTHON_REPL
    if not enabled or os.getenv("BIOMNI_GPU_DISPATCH_ROLE") == "server":
        return False

    import biomni.agent.a1 as a1_module

    if _ORIGINAL_RUN_PYTHON_REPL is None:
        _ORIGINAL_RUN_PYTHON_REPL = a1_module.run_python_repl
    if _ORIGINAL_RUN_BASH_SCRIPT is None:
        _ORIGINAL_RUN_BASH_SCRIPT = a1_module.run_bash_script

    _CODE_DISPATCH_CONFIG.update(
        {
            "base_url": base_url or os.getenv("BIOMNI_GPU_DISPATCH_BASE_URL") or DEFAULT_GPU_DISPATCH_BASE_URL,
            "jobs_dir": jobs_dir or os.getenv("BIOMNI_GPU_DISPATCH_JOBS_DIR") or DEFAULT_GPU_DISPATCH_JOBS_DIR,
            "timeout_seconds": timeout_seconds,
            "session_id": session_id,
        }
    )

    @functools.wraps(_ORIGINAL_RUN_PYTHON_REPL)
    def run_python_repl_with_gpu_dispatch(command: str) -> str:
        if not default_remote_execute_enabled() and not should_dispatch_python_code(command):
            return _ORIGINAL_RUN_PYTHON_REPL(command)
        if default_remote_execute_enabled() and should_run_python_locally(command):
            if should_prefer_local_python_code(command):
                print("[gpu-dispatch] running Python execute block locally (network/database I/O)", flush=True)
            return _ORIGINAL_RUN_PYTHON_REPL(command)

        client = GPUDispatchClient(
            base_url=_CODE_DISPATCH_CONFIG["base_url"],
            jobs_dir=_CODE_DISPATCH_CONFIG["jobs_dir"],
        )
        print(
            f"[gpu-dispatch] forwarding Python execute block to {client.base_url}",
            flush=True,
        )
        return client.submit_python_code(
            code=command.strip("```").strip(),
            session_id=str(_CODE_DISPATCH_CONFIG["session_id"]),
            cwd=os.getcwd(),
            timeout_seconds=int(_CODE_DISPATCH_CONFIG["timeout_seconds"]),
        )

    run_python_repl_with_gpu_dispatch._biomni_gpu_dispatch_wrapper = True  # type: ignore[attr-defined]
    a1_module.run_python_repl = run_python_repl_with_gpu_dispatch

    @functools.wraps(_ORIGINAL_RUN_BASH_SCRIPT)
    def run_bash_script_with_gpu_dispatch(script: str) -> str:
        if not default_remote_execute_enabled() and not should_dispatch_python_code(script):
            return _ORIGINAL_RUN_BASH_SCRIPT(script)
        if default_remote_execute_enabled() and should_run_bash_locally(script):
            if should_prefer_local_bash_code(script):
                print("[gpu-dispatch] running Bash execute block locally (network/database I/O)", flush=True)
            return _ORIGINAL_RUN_BASH_SCRIPT(script)

        client = GPUDispatchClient(
            base_url=_CODE_DISPATCH_CONFIG["base_url"],
            jobs_dir=_CODE_DISPATCH_CONFIG["jobs_dir"],
        )
        print(
            f"[gpu-dispatch] forwarding Bash execute block to {client.base_url}",
            flush=True,
        )
        return client.submit_python_code(
            code=script.strip("```").strip(),
            session_id=str(_CODE_DISPATCH_CONFIG["session_id"]),
            cwd=os.getcwd(),
            timeout_seconds=int(_CODE_DISPATCH_CONFIG["timeout_seconds"]),
            language="bash",
        )

    run_bash_script_with_gpu_dispatch._biomni_gpu_dispatch_wrapper = True  # type: ignore[attr-defined]
    a1_module.run_bash_script = run_bash_script_with_gpu_dispatch
    return True


def install_gpu_dispatch_hooks(
    *,
    enabled: bool,
    base_url: str | None = None,
    jobs_dir: str | Path | None = None,
    timeout_seconds: int = DEFAULT_GPU_DISPATCH_TIMEOUT_SECONDS,
    functions: tuple[str, ...] | None = None,
) -> list[str]:
    """Monkey-patch selected Biomni tool functions so A1 calls run on the GPU worker."""
    if not enabled or os.getenv("BIOMNI_GPU_DISPATCH_ROLE") == "server":
        return []

    selected = functions or parse_function_list(os.getenv("BIOMNI_GPU_DISPATCH_FUNCTIONS"))
    resolved_base_url = base_url or os.getenv("BIOMNI_GPU_DISPATCH_BASE_URL") or DEFAULT_GPU_DISPATCH_BASE_URL
    resolved_jobs_dir = jobs_dir or os.getenv("BIOMNI_GPU_DISPATCH_JOBS_DIR") or DEFAULT_GPU_DISPATCH_JOBS_DIR
    installed: list[str] = []

    for qualified_name in selected:
        try:
            module_name, function_name = qualified_name.rsplit(".", 1)
            module = importlib.import_module(module_name)
            current = getattr(module, function_name)
        except Exception as exc:
            print(f"[gpu-dispatch] skip {qualified_name}: {exc}", flush=True)
            continue

        if getattr(current, "_biomni_gpu_dispatch_wrapper", False):
            installed.append(qualified_name)
            continue

        _PATCHED_FUNCTIONS.setdefault(qualified_name, current)
        setattr(
            module,
            function_name,
            _make_remote_wrapper(
                current,
                module_name=module_name,
                function_name=function_name,
                base_url=resolved_base_url,
                jobs_dir=resolved_jobs_dir,
                timeout_seconds=timeout_seconds,
            ),
        )
        installed.append(qualified_name)

    if installed:
        print(
            f"[gpu-dispatch] installed {len(installed)} remote tool hooks; "
            f"base_url={resolved_base_url} jobs_dir={resolved_jobs_dir}",
            flush=True,
        )
    return installed
