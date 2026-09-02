"""Runtime environment helpers for Biomni GPU-dispatched jobs."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(os.getenv("BIOMNI_PROJECT_ROOT", Path(__file__).resolve().parent))
DEFAULT_AGENT_ARTIFACT_DIR = ROOT / "agent_artifacts"
DEFAULT_PROENV_CONFIG = Path(os.getenv("BIOMNI_PROENV_CONFIG", ""))
DEFAULT_ALPHAFOLD_ARTIFACT_DIR = DEFAULT_AGENT_ARTIFACT_DIR / "alphafold"
FALLBACK_ESM_REPOS = ()
EXPORTED_MODEL_ENV_KEYS = (
    "BIOMNI_PROENV_CONFIG",
    "BIOMNI_ESM2_MODEL_PATH",
    "BIOMNI_ESM1V_MODEL_DIR",
    "BIOMNI_ESM1V_MODEL_LOCATIONS",
    "BIOMNI_ESM_REPO",
    "BIOMNI_ESMFOLD_MODEL_PATH",
    "BIOMNI_CHATNT_MODEL_PATH",
    "BIOMNI_AGENT_ARTIFACT_DIR",
    "BIOMNI_ALPHAFOLD_ARTIFACT_DIR",
    "BIOMNI_HF_HOME",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_DATASETS_OFFLINE",
)


def _clean_yaml_scalar(value: str) -> str:
    value = value.split("#", 1)[0].strip()
    if value in {"", "null", "None"}:
        return ""
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def _parse_simple_yaml(path: Path) -> dict[tuple[str, str], Any]:
    """Parse the scalar/list subset used by ProEnv's distributed.yml."""
    if not path.exists():
        return {}

    parsed: dict[tuple[str, str], Any] = {}
    current_section: str | None = None
    current_key: str | None = None
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()
        if indent == 0 and stripped.endswith(":"):
            current_section = stripped[:-1]
            current_key = None
            continue
        if current_section and indent == 2 and ":" in stripped:
            key, raw_value = stripped.split(":", 1)
            current_key = key.strip()
            value = _clean_yaml_scalar(raw_value)
            parsed[(current_section, current_key)] = value if value else []
            continue
        if current_section and current_key and indent >= 4 and stripped.startswith("-"):
            value = _clean_yaml_scalar(stripped[1:])
            if value:
                existing = parsed.setdefault((current_section, current_key), [])
                if isinstance(existing, list):
                    existing.append(value)
    return parsed


def _first_existing_path(*values: str | None) -> str | None:
    for value in values:
        if value and Path(value).exists():
            return value
    return None


def _first_existing_repo(*values: str | None) -> str | None:
    for value in values:
        if value and Path(value).exists():
            return value
    for path in FALLBACK_ESM_REPOS:
        if path.exists():
            return str(path)
    return None


def collect_runtime_model_env(config_path: str | Path | None = None) -> dict[str, str]:
    """Collect shared model/cache paths without downloading anything on the GPU host."""
    resolved_config = Path(
        config_path
        or os.getenv("BIOMNI_PROENV_CONFIG")
        or DEFAULT_PROENV_CONFIG
    )
    parsed = _parse_simple_yaml(resolved_config)

    def get(section: str, key: str) -> Any:
        return parsed.get((section, key))

    env: dict[str, str] = {"BIOMNI_PROENV_CONFIG": str(resolved_config)}

    esm2_path = _first_existing_path(
        os.getenv("BIOMNI_ESM2_MODEL_PATH"),
        get("fitness_score", "esm2_name_or_path"),
        get("bert_score", "esm2_name_or_path"),
    )
    if esm2_path:
        env["BIOMNI_ESM2_MODEL_PATH"] = esm2_path

    esm1v_dir = _first_existing_path(
        os.getenv("BIOMNI_ESM1V_MODEL_DIR"),
        get("fitness_score", "esm1v_model_dir"),
    )
    if esm1v_dir:
        env["BIOMNI_ESM1V_MODEL_DIR"] = esm1v_dir

    locations = get("fitness_score", "esm1v_model_locations")
    if isinstance(locations, list):
        existing_locations = [item for item in locations if Path(item).exists()]
        if existing_locations:
            env["BIOMNI_ESM1V_MODEL_LOCATIONS"] = os.pathsep.join(existing_locations)

    esm_repo = _first_existing_repo(
        os.getenv("BIOMNI_ESM_REPO"),
        get("fitness_score", "esm1v_esm_repo"),
    )
    if esm_repo:
        env["BIOMNI_ESM_REPO"] = esm_repo

    esmfold_path = _first_existing_path(
        os.getenv("BIOMNI_ESMFOLD_MODEL_PATH"),
        get("foldability", "esm_fold_name_or_path"),
        get("tm_score", "esm_fold_name_or_path"),
        get("motif_rmsd", "esm_fold_name_or_path"),
    )
    if esmfold_path:
        env["BIOMNI_ESMFOLD_MODEL_PATH"] = esmfold_path

    chatnt_path = _first_existing_path(
        os.getenv("BIOMNI_CHATNT_MODEL_PATH"),
        get("chatnt", "name_or_path"),
    )
    if chatnt_path:
        env["BIOMNI_CHATNT_MODEL_PATH"] = chatnt_path

    hf_home = _first_existing_path(
        os.getenv("BIOMNI_HF_HOME"),
        os.getenv("HF_HOME"),
        get("fitness_score", "aido_rag_hf_home"),
        get("fitness_score", "venusrem_cache_dir"),
    )
    if hf_home:
        env["BIOMNI_HF_HOME"] = hf_home
        env["HF_HOME"] = hf_home
        env["TRANSFORMERS_CACHE"] = os.getenv("TRANSFORMERS_CACHE") or hf_home
        hub_cache = Path(hf_home) / "hub"
        env["HUGGINGFACE_HUB_CACHE"] = os.getenv("HUGGINGFACE_HUB_CACHE") or str(hub_cache)

    env["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1")
    env["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "1")
    env["HF_DATASETS_OFFLINE"] = os.getenv("HF_DATASETS_OFFLINE", "1")
    env["TOKENIZERS_PARALLELISM"] = os.getenv("TOKENIZERS_PARALLELISM", "false")
    env["BIOMNI_ALPHAFOLD_ARTIFACT_DIR"] = os.getenv(
        "BIOMNI_ALPHAFOLD_ARTIFACT_DIR",
        str(DEFAULT_ALPHAFOLD_ARTIFACT_DIR),
    )
    env["BIOMNI_AGENT_ARTIFACT_DIR"] = os.getenv(
        "BIOMNI_AGENT_ARTIFACT_DIR",
        str(DEFAULT_AGENT_ARTIFACT_DIR),
    )
    return env


def apply_runtime_environment(config_path: str | Path | None = None, *, override: bool = False) -> dict[str, str]:
    env = collect_runtime_model_env(config_path)
    for key, value in env.items():
        if override or not os.environ.get(key):
            os.environ[key] = value

    for artifact_key in ("BIOMNI_ALPHAFOLD_ARTIFACT_DIR", "BIOMNI_AGENT_ARTIFACT_DIR"):
        artifact_dir = os.environ.get(artifact_key)
        if artifact_dir:
            Path(artifact_dir).mkdir(parents=True, exist_ok=True)

    esm_repo = os.environ.get("BIOMNI_ESM_REPO")
    if esm_repo and Path(esm_repo).exists() and esm_repo not in sys.path:
        sys.path.insert(0, esm_repo)
    return env


def inject_python_namespace(namespace: dict[str, Any]) -> None:
    model_paths = {key: os.environ[key] for key in EXPORTED_MODEL_ENV_KEYS if os.environ.get(key)}
    namespace["BIOMNI_MODEL_PATHS"] = model_paths
    for key, value in model_paths.items():
        namespace[key] = value


def _redirect_pretrained_name(pretrained_model_name_or_path: Any) -> Any:
    try:
        name = os.fspath(pretrained_model_name_or_path)
    except TypeError:
        return pretrained_model_name_or_path

    if name.startswith("facebook/esmfold"):
        return os.environ.get("BIOMNI_ESMFOLD_MODEL_PATH") or pretrained_model_name_or_path
    if name.startswith("facebook/esm"):
        return os.environ.get("BIOMNI_ESM2_MODEL_PATH") or pretrained_model_name_or_path
    if name == "InstaDeepAI/ChatNT":
        return os.environ.get("BIOMNI_CHATNT_MODEL_PATH") or pretrained_model_name_or_path
    return pretrained_model_name_or_path


def _patch_transformers_class(cls: Any) -> None:
    original = getattr(cls, "from_pretrained", None)
    if original is None or getattr(original, "_biomni_runtime_redirect", False):
        return

    def patched_from_pretrained(pretrained_model_name_or_path: Any, *args: Any, **kwargs: Any) -> Any:
        redirected = _redirect_pretrained_name(pretrained_model_name_or_path)
        if redirected != pretrained_model_name_or_path:
            kwargs.setdefault("local_files_only", True)
        elif isinstance(redirected, (str, os.PathLike)) and Path(redirected).exists():
            kwargs.setdefault("local_files_only", True)
        return original(redirected, *args, **kwargs)

    patched_from_pretrained._biomni_runtime_redirect = True  # type: ignore[attr-defined]
    setattr(cls, "from_pretrained", staticmethod(patched_from_pretrained))


def install_transformers_local_redirects() -> None:
    """Redirect common ESM HuggingFace model ids to shared local model paths."""
    try:
        import transformers
    except Exception:
        return

    original_pipeline = getattr(transformers, "pipeline", None)
    if original_pipeline is not None and not getattr(original_pipeline, "_biomni_runtime_redirect", False):

        def patched_pipeline(*args: Any, **kwargs: Any) -> Any:
            if "model" in kwargs:
                redirected = _redirect_pretrained_name(kwargs["model"])
                if redirected != kwargs["model"]:
                    kwargs["model"] = redirected
                    kwargs.setdefault("local_files_only", True)
            elif len(args) >= 2:
                redirected = _redirect_pretrained_name(args[1])
                if redirected != args[1]:
                    args = (args[0], redirected, *args[2:])
                    kwargs.setdefault("local_files_only", True)
            return original_pipeline(*args, **kwargs)

        patched_pipeline._biomni_runtime_redirect = True  # type: ignore[attr-defined]
        transformers.pipeline = patched_pipeline

    for class_name in (
        "AutoTokenizer",
        "AutoModel",
        "AutoModelForCausalLM",
        "AutoModelForSeq2SeqLM",
        "AutoModelForMaskedLM",
        "AutoConfig",
        "EsmModel",
        "EsmForMaskedLM",
        "EsmTokenizer",
    ):
        cls = getattr(transformers, class_name, None)
        if cls is not None:
            _patch_transformers_class(cls)
