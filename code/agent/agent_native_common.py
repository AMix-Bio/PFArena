#!/usr/bin/env python3
"""Shared agent-native runners for the T1--T4 mutation benchmark tasks.

Unlike the legacy wrappers, these runners do not ask A1 to execute one fixed
batch Python script that precomputes a hard-coded tool context. Each benchmark
query is sent to Biomni A1 as its own task. A1 is instructed to reason over
multiple turns, inspect tool outputs, decide whether more tools are needed, and
then directly emit the ranking JSON.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import signal
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ["BIOMNI_PROJECT_ROOT"]) if os.environ.get("BIOMNI_PROJECT_ROOT") else next(
    (p for p in (_HERE, *_HERE.parents) if (p / "biomni_benchmark").is_dir()), _HERE
)
CODE_ROOT = _HERE.parent
for import_root in (ROOT, ROOT / "biomni", CODE_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from router import DATASET_INPUTS  # noqa: E402

def format_candidates(candidates: tuple[str, ...] | list[str]) -> str:
    return "\n".join(f"{index}. {candidate}" for index, candidate in enumerate(candidates, 1))


DATA_ROOT = Path(os.getenv("BIOMNI_DATA_ROOT", "data"))
TASK_BY_ID = {
    task_id: task_name
    for task_id, (task_name, _) in DATASET_INPUTS.items()
}
_TASK_DESCRIPTIONS = {
    "1": "single-mutant de-novo generation",
    "2": "measurement-free multi-mutant closed-set ranking",
    "3": "anchor-informed multi-mutant closed-set ranking",
    "4": "mutation-informed multi-mutant closed-set ranking",
}
TASK_NAMES = {
    TASK_BY_ID[task_id]: f"T{task_id} {_TASK_DESCRIPTIONS[task_id]}"
    for task_id in TASK_BY_ID
}
EVAL_SETTINGS = {task_name: task_name for task_name in TASK_BY_ID.values()}
DEFAULT_DIRS = {
    task_name: DATA_ROOT / task_name
    for task_name in TASK_BY_ID.values()
}
DEFAULT_A1_DATA = ROOT / "biomni_benchmark/a1_data"
DEFAULT_RESULT_ROOT = Path(os.getenv("BIOMNI_RESULT_ROOT", str(ROOT / "results")))
DEFAULT_AGENT_ARTIFACT_DIR = Path(os.getenv("BIOMNI_AGENT_ARTIFACT_DIR", str(ROOT / "agent_artifacts")))
DEFAULT_ALPHAFOLD_ARTIFACT_DIR = DEFAULT_AGENT_ARTIFACT_DIR / "alphafold"
DEFAULT_GPU_DISPATCH_JOBS_DIR = DEFAULT_AGENT_ARTIFACT_DIR / "gpu_dispatch_jobs"
AA = "ACDEFGHIKLMNPQRSTVWY"
EPI_RE = re.compile(r"epi_(\d+)\.csv$")
SITE_RE = re.compile(r"site_(\d+)\.csv$")
MUT_RE = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])")
TOOL_NAME_RE = re.compile(
    r"\b("
    r"query_uniprot|query_alphafold|query_interpro|query_pubmed|query_pmc|"
    r"find_n_glycosylation_motifs|predict_o_glycosylation_hotspots|"
    r"list_local_protocols|read_local_protocol|"
    r"blast|hmmer|foldseek|alphafold|interpro|uniprot|pubmed"
    r")\b",
    re.IGNORECASE,
)


class FlushWriter:
    """Small stdout/stderr proxy that flushes redirected A1 logs immediately."""

    def __init__(self, handle: Any):
        self.handle = handle

    def write(self, text: str) -> int:
        written = self.handle.write(text)
        self.handle.flush()
        return written

    def flush(self) -> None:
        self.handle.flush()

    def isatty(self) -> bool:
        return False


SECRET_LINE_RE = re.compile(
    r"(?i)(api\s*key|api_key|authorization|bearer\s+token|access\s*token)(\s*[:=]\s*)(?:bearer\s+)?\S+"
)
SECRET_TOKEN_RE = re.compile(r"\b(?:sk|sk-proj|sk-ant|xai)-[A-Za-z0-9._-]{12,}\b")


def collect_log_secrets(*values: str | None) -> tuple[str, ...]:
    secrets: list[str] = []
    for value in values:
        if value and value not in secrets:
            secrets.append(value)
    for env_name in (
        "BIOMNI_CUSTOM_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "DEEPSEEK_API_KEY",
    ):
        value = os.environ.get(env_name)
        if value and value not in secrets:
            secrets.append(value)
    return tuple(secrets)


def redact_log_text(text: str, secrets: tuple[str, ...]) -> str:
    redacted = text
    for secret in sorted(secrets, key=len, reverse=True):
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    redacted = SECRET_LINE_RE.sub(lambda match: f"{match.group(1)}{match.group(2)}[REDACTED]", redacted)
    redacted = SECRET_TOKEN_RE.sub("[REDACTED]", redacted)
    return redacted


class RedactingFlushWriter(FlushWriter):
    """FlushWriter variant that removes API credentials from persisted logs."""

    def __init__(self, handle: Any, secrets: tuple[str, ...]):
        super().__init__(handle)
        self.secrets = secrets

    def write(self, text: str) -> int:
        return super().write(redact_log_text(text, self.secrets))


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def ensure_pythonpath_entries(*entries: Path) -> None:
    current = [item for item in os.environ.get("PYTHONPATH", "").split(os.pathsep) if item]
    additions = [str(entry) for entry in entries if str(entry) not in current]
    if additions:
        os.environ["PYTHONPATH"] = os.pathsep.join(additions + current)


@contextlib.contextmanager
def wall_time_limit(seconds: int, label: str):
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _raise_timeout(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"{label} timed out after {seconds} seconds")

    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, _raise_timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, previous_timer[0], previous_timer[1])
        signal.signal(signal.SIGALRM, previous_handler)


@dataclass(frozen=True)
class AssayRecord:
    assay_id: str
    uniprot_id: str
    primary_task_class: str
    fitness_type: str
    readout_subclass: str
    wildtype_sequence: str


@dataclass(frozen=True)
class ClosedSetSample:
    assay_id: str
    sample_path: Path
    candidates: tuple[str, ...]
    candidate_group_id: str


@dataclass(frozen=True)
class ConditionPairGroup:
    source_assay_id: str
    candidate_group_id: str
    group_path: Path
    candidates: tuple[str, ...]
    condition_a: str
    condition_b: str
    assay_id_a: str
    assay_id_b: str


@dataclass(frozen=True)
class AnchorInformedGroup:
    source_assay_id: str
    candidate_group_id: str
    group_path: Path
    candidates: tuple[str, ...]
    anchor_mutant: str
    anchor_dms_score: str


@dataclass(frozen=True)
class MutationInformedGroup:
    source_assay_id: str
    candidate_group_id: str
    group_path: Path
    candidates: tuple[str, ...]
    single_context: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class NativeQuery:
    index: int
    task: str
    assay: str
    record: AssayRecord
    candidates: tuple[str, ...]
    measured_mutants: frozenset[str]
    source_path: str
    source_assay: str | None = None
    candidate_group_id: str | None = None
    condition_a: str | None = None
    condition_b: str | None = None
    target_mutation: str | None = None
    candidate_descriptions: tuple[str, ...] = ()
    anchor_mutant: str | None = None
    anchor_dms_score: str | None = None
    single_context: tuple[tuple[str, str], ...] = ()

    @property
    def is_closed_set(self) -> bool:
        return bool(self.candidates)


def normalize_openai_base_url(url: str | None) -> str | None:
    if not url:
        return None
    cleaned = url.rstrip("/")
    return cleaned if cleaned.endswith("/v1") else cleaned + "/v1"


def resolve_api_key(name: str | None) -> tuple[str | None, str | None]:
    names: list[str] = []
    if name:
        names.append(name)
    names.extend(["BIOMNI_CUSTOM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
    seen: set[str] = set()
    for env_name in names:
        if env_name in seen:
            continue
        seen.add(env_name)
        value = os.environ.get(env_name)
        if value:
            return value, env_name
    return None, None


def key_fingerprint(value: str | None) -> str:
    if not value:
        return "none"
    digest = hashlib.sha256(value.encode()).hexdigest()[:12]
    return f"sha256:{digest},len:{len(value)}"


def prime_provider_env() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        os.environ["ANTHROPIC_API_KEY"] = os.environ["ANTHROPIC_AUTH_TOKEN"]


def configure_biomni_default_llm(
    *,
    model: str,
    source: str | None,
    base_url: str | None,
    api_key: str | None,
) -> None:
    """Keep Biomni's internal tool LLM config aligned with the benchmark agent."""
    os.environ["BIOMNI_LLM"] = model
    os.environ["BIOMNI_LLM_MODEL"] = model
    if source:
        os.environ["BIOMNI_SOURCE"] = source
    if base_url:
        os.environ["BIOMNI_CUSTOM_BASE_URL"] = base_url
    if api_key:
        os.environ["BIOMNI_CUSTOM_API_KEY"] = api_key

    try:
        from biomni.config import default_config
    except Exception:
        return

    default_config.llm = model
    default_config.source = source
    default_config.base_url = base_url
    default_config.api_key = api_key or "EMPTY"


def is_deepseek_model(model: str | None) -> bool:
    return bool(model and "deepseek" in model.lower())


def openai_timeout_kwargs() -> dict[str, float | int]:
    return {
        "timeout": float(os.getenv("BIOMNI_LLM_TIMEOUT_SECONDS", "120")),
        "max_retries": int(os.getenv("BIOMNI_LLM_MAX_RETRIES", "1")),
    }


def create_custom_chat_openai(
    *,
    model: str,
    stop_sequences: list[str] | None,
    base_url: str | None,
    api_key: str | None,
    max_tokens: int,
    temperature: float | None,
    deepseek_thinking: bool,
    deepseek_thinking_type: str,
    deepseek_reasoning_effort: str,
) -> Any:
    from langchain_openai import ChatOpenAI

    chat_openai_cls = ChatOpenAI
    if model.startswith("gpt-5"):
        class _CustomChatOpenAINoStop(ChatOpenAI):
            def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
                payload = super()._get_request_payload(input_, stop=stop, **kwargs)
                payload.pop("stop", None)
                return payload

        chat_openai_cls = _CustomChatOpenAINoStop

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "stop_sequences": stop_sequences,
        "base_url": base_url,
        "api_key": api_key or "EMPTY",
        **openai_timeout_kwargs(),
    }
    if not deepseek_thinking and temperature is not None:
        kwargs["temperature"] = temperature

    if deepseek_thinking:
        kwargs["reasoning_effort"] = deepseek_reasoning_effort
        kwargs["extra_body"] = {"thinking": {"type": deepseek_thinking_type}}

    try:
        return chat_openai_cls(**kwargs)
    except TypeError:
        # Older langchain-openai versions may not expose `extra_body` directly.
        extra_body = kwargs.pop("extra_body", None)
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        model_kwargs: dict[str, Any] = {}
        if isinstance(extra_body, dict) and isinstance(extra_body.get("thinking"), dict):
            model_kwargs["thinking"] = extra_body["thinking"]
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        return chat_openai_cls(**kwargs)


def install_custom_get_llm_patch(a1_module: Any, args: argparse.Namespace):
    original_get_llm = getattr(a1_module, "get_llm")
    if args.agent_source != "Custom":
        return original_get_llm, False

    def patched_get_llm(
        model: str | None = None,
        temperature: float | None = None,
        stop_sequences: list[str] | None = None,
        source: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        config: Any = None,
    ) -> Any:
        resolved_model = model or args.agent_llm
        resolved_source = source or args.agent_source
        resolved_base_url = base_url or args.agent_base_url
        resolved_api_key = api_key
        if config is not None:
            resolved_base_url = resolved_base_url or getattr(config, "base_url", None)
            resolved_api_key = resolved_api_key or getattr(config, "api_key", None)
        if resolved_source == "Custom":
            use_deepseek_thinking = is_deepseek_model(resolved_model) and args.deepseek_thinking
            return create_custom_chat_openai(
                model=resolved_model,
                stop_sequences=stop_sequences,
                base_url=resolved_base_url,
                api_key=resolved_api_key,
                max_tokens=args.agent_max_tokens,
                temperature=temperature,
                deepseek_thinking=use_deepseek_thinking,
                deepseek_thinking_type=args.deepseek_thinking_type,
                deepseek_reasoning_effort=args.deepseek_reasoning_effort,
            )
        return original_get_llm(
            model=model,
            temperature=temperature,
            stop_sequences=stop_sequences,
            source=source,
            base_url=base_url,
            api_key=api_key,
            config=config,
        )

    a1_module.get_llm = patched_get_llm
    return original_get_llm, True


def count_text_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1024 * 1024), b""))


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def render_progress(done: int, total: int, elapsed: float) -> str:
    width = 28
    ratio = min(1.0, done / total) if total else 0.0
    filled = int(width * ratio)
    bar = "#" * filled + "-" * (width - filled)
    return f"\rprogress [{bar}] {done}/{total} queries {ratio * 100:5.1f}% elapsed {format_duration(elapsed)}"


def safe_path_component(value: str, *, max_length: int = 80) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not cleaned:
        cleaned = "unknown"
    if len(cleaned) <= max_length:
        return cleaned
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[: max_length - 11]}_{digest}"


def query_artifact_dir(query: NativeQuery, root: Path) -> Path:
    assay_part = safe_path_component(query.assay, max_length=72)
    task_part = safe_path_component(query.task, max_length=64)
    digest = hashlib.sha1(f"{query.task}\0{query.index}\0{query.assay}".encode("utf-8")).hexdigest()[:10]
    return root / "queries" / task_part / f"index_{query.index:04d}_{assay_part}_{digest}"


def build_artifact_run_id(task: str) -> str:
    override = os.environ.get("BIOMNI_AGENT_ARTIFACT_RUN_ID")
    if override:
        return safe_path_component(override, max_length=96)
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    return safe_path_component(f"{task}_{timestamp}_pid{os.getpid()}", max_length=120)


def resolve_artifact_run_root(base_root: Path, run_id: str) -> Path:
    return base_root.resolve() / "runs" / safe_path_component(run_id, max_length=120)


def clean_sequence(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in {"nan", "none", "null"}:
        return ""
    return "".join(ch for ch in text.upper() if ch in AA)


def first_nonempty(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = row.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return "unknown"


def load_assay_metadata_rows(data_dir: Path, task_label: str) -> list[dict[str, str]]:
    metadata_path = data_dir / "info.csv"
    if not metadata_path.exists():
        metadata_path = data_dir / "assay.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing {task_label} info.csv or assay.csv under: {data_dir}")
    with metadata_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_assay_records(data_dir: Path, task_label: str) -> list[AssayRecord]:
    records: list[AssayRecord] = []
    for row in load_assay_metadata_rows(data_dir, task_label):
        assay_id = str(row.get("assay_id") or "").strip()
        seq = clean_sequence(row.get("wildtype_sequence"))
        if not assay_id or not seq:
            continue
        records.append(
            AssayRecord(
                assay_id=assay_id,
                uniprot_id=first_nonempty(row, ["uniprot_id", "uniprot_primary_accession"]),
                primary_task_class=first_nonempty(row, ["primary_task_class"]),
                fitness_type=first_nonempty(row, ["fitness_type"]),
                readout_subclass=first_nonempty(
                    row,
                    ["readout_subclass", "fitness_subtype", "assay_modality", "phenotype", "raw_DMS_phenotype_name"],
                ),
                wildtype_sequence=seq,
            )
        )
    return records


def load_measured_mutants(data_dir: Path, assay_id: str) -> set[str]:
    path = data_dir / "norm_data" / f"{assay_id}.csv"
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            str(row.get("mutant") or "").strip()
            for row in csv.DictReader(handle)
            if str(row.get("mutant") or "").strip()
        }


def parse_json_ranking(text: str) -> list[str]:
    text = text.strip()
    if "```" in text:
        text = re.sub(r"^```(?:json)?", "", text)
        text = re.sub(r"```$", "", text).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            return []
        try:
            obj = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    ranking = obj.get("ranking", []) if isinstance(obj, dict) else []
    return [str(x).strip() for x in ranking if str(x).strip()]


def valid_mutation(mutant: str, sequence: str) -> bool:
    match = MUT_RE.fullmatch(mutant)
    if not match:
        return False
    ref, pos_text, alt = match.groups()
    pos = int(pos_text)
    return 1 <= pos <= len(sequence) and sequence[pos - 1] == ref and alt != ref


def complete_closed_set_ranking(
    assay_id: str,
    sequence: str,
    candidates: tuple[str, ...],
    llm_ranking: list[str],
) -> tuple[list[str], dict[str, int]]:
    del assay_id, sequence
    candidate_set = set(candidates)
    seen: set[str] = set()
    ranking: list[str] = []
    invalid_candidates = sum(1 for mutant in candidates if not mutant)
    duplicate = 0
    outside_candidate_set = 0

    for mutant in llm_ranking:
        if mutant in seen:
            duplicate += 1
            continue
        if mutant not in candidate_set:
            outside_candidate_set += 1
            continue
        seen.add(mutant)
        ranking.append(mutant)

    from_llm = len(ranking)
    return ranking[: len(candidates)], {
        "invalid_candidates": invalid_candidates,
        "duplicate": duplicate,
        "outside_candidate_set": outside_candidate_set,
        "from_llm": from_llm,
    }


def epi_sort_key(path: Path) -> tuple[int, str]:
    match = EPI_RE.fullmatch(path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def site_sort_key(path: Path) -> tuple[int, str]:
    match = SITE_RE.fullmatch(path.name)
    return (int(match.group(1)) if match else 10**9, path.name)


def load_one_flat_sample(path: Path) -> ClosedSetSample:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    group_ids = [
        group_id
        for row in rows
        for group_id in [(row.get("candidate_group_id") or "").strip()]
        if group_id
    ]
    candidate_group_id = group_ids[0] if group_ids else path.stem
    candidates = tuple((row.get("mutant") or "").strip() for row in rows if (row.get("mutant") or "").strip())
    if not candidates:
        raise ValueError(f"{path} has no candidate mutants")
    return ClosedSetSample(
        assay_id=path.stem,
        sample_path=path,
        candidates=candidates,
        candidate_group_id=candidate_group_id,
    )


def load_flat_samples(
    data_dir: Path,
    task_label: str,
    *,
    max_assays: int | None = None,
    start_index: int = 0,
) -> tuple[list[AssayRecord], list[ClosedSetSample]]:
    records = load_assay_records(data_dir, task_label)
    selected_records = records[start_index : None if max_assays is None else start_index + max_assays]
    selected_ids = {record.assay_id for record in selected_records}
    order_index = {record.assay_id: index for index, record in enumerate(selected_records)}
    samples: list[ClosedSetSample] = []
    norm_dir = data_dir / "norm_data"
    for path in sorted(norm_dir.glob("*.csv")):
        if path.stem in selected_ids:
            samples.append(load_one_flat_sample(path))
    samples.sort(key=lambda sample: order_index[sample.assay_id])
    return selected_records, samples


def load_measurement_free_multi_mutant_samples(
    data_dir: Path,
    *,
    max_assays: int | None = None,
    start_index: int = 0,
) -> tuple[list[AssayRecord], list[ClosedSetSample]]:
    records = load_assay_records(data_dir, TASK_BY_ID["2"])
    selected_records = records[start_index : None if max_assays is None else start_index + max_assays]
    selected_ids = {record.assay_id for record in selected_records}
    order_index = {record.assay_id: index for index, record in enumerate(selected_records)}
    samples: list[ClosedSetSample] = []
    norm_dir = data_dir / "norm_data"
    for path in sorted(norm_dir.glob("*_random_multi.csv")):
        assay_id = path.stem[: -len("_random_multi")]
        if assay_id in selected_ids:
            loaded = load_one_flat_sample(path)
            samples.append(
                ClosedSetSample(
                    assay_id=assay_id,
                    sample_path=loaded.sample_path,
                    candidates=loaded.candidates,
                    candidate_group_id=loaded.candidate_group_id,
                )
            )
    samples.sort(key=lambda sample: order_index[sample.assay_id])
    return selected_records, samples


def first_unique_value(rows: list[dict[str, str]], names: tuple[str, ...]) -> str:
    for name in names:
        values = [(row.get(name) or "").strip() for row in rows]
        values = [value for value in values if value]
        if values:
            return values[0]
    return ""


def load_one_anchor_informed_group(path: Path, source_assay_id: str) -> AnchorInformedGroup:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    group_ids = sorted(
        {
            group_id
            for row in rows
            for group_id in [(row.get("candidate_group_id") or "").strip()]
            if group_id
        }
    )
    if len(group_ids) != 1:
        raise ValueError(f"{path} must contain exactly one candidate_group_id, got {group_ids}")
    candidates = tuple((row.get("mutant") or "").strip() for row in rows if (row.get("mutant") or "").strip())
    if not candidates:
        raise ValueError(f"{path} has no candidate mutants")
    anchor_mutant = first_unique_value(rows, ("anchor_mutant",))
    anchor_dms_score = first_unique_value(rows, ("model_visible_anchor_DMS_score", "anchor_DMS_score"))
    if not anchor_mutant or not anchor_dms_score:
        raise ValueError(f"{path} is missing visible anchor_mutant or anchor_DMS_score")
    return AnchorInformedGroup(
        source_assay_id=source_assay_id,
        candidate_group_id=group_ids[0],
        group_path=path,
        candidates=candidates,
        anchor_mutant=anchor_mutant,
        anchor_dms_score=anchor_dms_score,
    )


def load_anchor_informed_groups(
    data_dir: Path,
    *,
    max_assays: int | None = None,
    start_assay_index: int = 0,
    max_groups: int | None = None,
    start_group_index: int = 0,
) -> tuple[list[AssayRecord], list[AnchorInformedGroup]]:
    records = load_assay_records(data_dir, TASK_BY_ID["3"])
    selected_records = records[
        start_assay_index : None if max_assays is None else start_assay_index + max_assays
    ]
    groups: list[AnchorInformedGroup] = []
    for record in selected_records:
        assay_dir = data_dir / "norm_data" / record.assay_id
        if not assay_dir.exists():
            continue
        for group_path in sorted(assay_dir.glob("anchor_*.csv")):
            groups.append(load_one_anchor_informed_group(group_path, record.assay_id))
    selected_groups = groups[
        start_group_index : None if max_groups is None else start_group_index + max_groups
    ]
    return selected_records, selected_groups


def load_visible_single_context(path: Path) -> tuple[tuple[str, str], ...]:
    if not path.exists():
        raise FileNotFoundError(f"Missing visible single-mutant context file: {path}")
    context: list[tuple[str, str]] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            mutant = (row.get("mutant") or "").strip()
            score = (row.get("model_visible_DMS_score") or "").strip()
            if mutant and score:
                context.append((mutant, score))
    if not context:
        raise ValueError(f"{path} has no visible single-mutant context rows with model_visible_DMS_score")
    return tuple(context)


def load_one_mutation_informed_group(path: Path, source_assay_id: str, data_dir: Path) -> MutationInformedGroup:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    group_ids = sorted(
        {
            group_id
            for row in rows
            for group_id in [(row.get("candidate_group_id") or "").strip()]
            if group_id
        }
    )
    if len(group_ids) != 1:
        raise ValueError(f"{path} must contain exactly one candidate_group_id, got {group_ids}")
    candidates = tuple((row.get("mutant") or "").strip() for row in rows if (row.get("mutant") or "").strip())
    if not candidates:
        raise ValueError(f"{path} has no combo candidate mutants")
    context_rel = first_unique_value(rows, ("single_mutant_context_path",))
    if context_rel:
        context_path = data_dir.parent / context_rel
    else:
        context_path = path.parent / "single_mutant_context.csv"
    return MutationInformedGroup(
        source_assay_id=source_assay_id,
        candidate_group_id=group_ids[0],
        group_path=path,
        candidates=candidates,
        single_context=load_visible_single_context(context_path),
    )


def load_mutation_informed_groups(
    data_dir: Path,
    *,
    max_assays: int | None = None,
    start_assay_index: int = 0,
    max_groups: int | None = None,
    start_group_index: int = 0,
) -> tuple[list[AssayRecord], list[MutationInformedGroup]]:
    records = load_assay_records(data_dir, TASK_BY_ID["4"])
    selected_records = records[
        start_assay_index : None if max_assays is None else start_assay_index + max_assays
    ]
    groups: list[MutationInformedGroup] = []
    for record in selected_records:
        assay_dir = data_dir / "norm_data" / record.assay_id
        if not assay_dir.exists():
            continue
        for group_path in sorted(assay_dir.glob("single_context_combo*.csv")):
            groups.append(load_one_mutation_informed_group(group_path, record.assay_id, data_dir))
    selected_groups = groups[
        start_group_index : None if max_groups is None else start_group_index + max_groups
    ]
    return selected_records, selected_groups


def build_queries(task: str, data_dir: Path, args: argparse.Namespace) -> list[NativeQuery]:
    if task == TASK_BY_ID["2"]:
        records, samples = load_measurement_free_multi_mutant_samples(
            data_dir,
            max_assays=args.max_items,
            start_index=args.start_index,
        )
        record_by_assay = {record.assay_id: record for record in records}
        return [
            NativeQuery(
                index=args.start_index + offset,
                task=task,
                assay=sample.assay_id,
                record=record_by_assay[sample.assay_id],
                candidates=sample.candidates,
                measured_mutants=frozenset(),
                source_path=str(sample.sample_path),
                source_assay=sample.assay_id,
                candidate_group_id=sample.candidate_group_id,
            )
            for offset, sample in enumerate(samples)
        ]

    if task == TASK_BY_ID["3"]:
        start_group = args.start_group_index if args.start_group_index is not None else args.start_index
        records, groups = load_anchor_informed_groups(
            data_dir,
            max_assays=args.max_assays,
            start_assay_index=args.start_assay_index,
            max_groups=args.max_items,
            start_group_index=start_group,
        )
        record_by_assay = {record.assay_id: record for record in records}
        return [
            NativeQuery(
                index=start_group + offset,
                task=task,
                assay=group.candidate_group_id,
                record=record_by_assay[group.source_assay_id],
                candidates=group.candidates,
                measured_mutants=frozenset(),
                source_path=str(group.group_path),
                source_assay=group.source_assay_id,
                candidate_group_id=group.candidate_group_id,
                anchor_mutant=group.anchor_mutant,
                anchor_dms_score=group.anchor_dms_score,
            )
            for offset, group in enumerate(groups)
        ]

    if task == TASK_BY_ID["4"]:
        start_group = args.start_group_index if args.start_group_index is not None else args.start_index
        records, groups = load_mutation_informed_groups(
            data_dir,
            max_assays=args.max_assays,
            start_assay_index=args.start_assay_index,
            max_groups=args.max_items,
            start_group_index=start_group,
        )
        record_by_assay = {record.assay_id: record for record in records}
        return [
            NativeQuery(
                index=start_group + offset,
                task=task,
                assay=group.candidate_group_id,
                record=record_by_assay[group.source_assay_id],
                candidates=group.candidates,
                measured_mutants=frozenset(),
                source_path=str(group.group_path),
                source_assay=group.source_assay_id,
                candidate_group_id=group.candidate_group_id,
                single_context=group.single_context,
            )
            for offset, group in enumerate(groups)
        ]

    raise ValueError(f"Unsupported task guidance for {query.task}")


def format_condition_candidates(candidates: tuple[str, ...], descriptions: tuple[str, ...]) -> str:
    lines: list[str] = []
    for index, candidate in enumerate(candidates, start=1):
        description = descriptions[index - 1] if index - 1 < len(descriptions) else candidate
        lines.append(f"{index}. Condition assay ID: `{candidate}`\n   Description: {description}")
    return "\n".join(lines)


def format_single_context(context: tuple[tuple[str, str], ...]) -> str:
    return "\n".join(
        f"{index}. {mutant}: model-visible single-mutant DMS_score = {score}"
        for index, (mutant, score) in enumerate(context, start=1)
    )


def build_agent_prompt(query: NativeQuery, max_tool_rounds: int, require_tool_use: bool = True) -> str:
    record = query.record
    prompt_artifact_dir = os.environ.get("BIOMNI_AGENT_ARTIFACT_DIR", str(DEFAULT_AGENT_ARTIFACT_DIR))
    aa_alphabet = ", ".join(AA)
    candidate_block = ""
    if query.candidates:
        candidate_heading = "CANDIDATE MUTATION STRINGS TO RANK"
        if query.task in {
            TASK_BY_ID["2"],
            TASK_BY_ID["3"],
            TASK_BY_ID["4"],
        }:
            candidate_heading = "CANDIDATE MULTI-MUTATIONS TO RANK"
        candidate_block = f"""

### {candidate_heading} ({len(query.candidates)} total)
{format_candidates(query.candidates)}
"""
    condition_block = ""
    anchor_block = ""
    if query.task == TASK_BY_ID["3"]:
        anchor_block = f"""

### ANCHOR MUTANT CONTEXT
- Anchor Mutant: {query.anchor_mutant or "unknown"}
- Anchor DMS Score: {query.anchor_dms_score or "unknown"}
"""
    single_context_block = ""
    if query.task == TASK_BY_ID["4"] and query.single_context:
        single_context_block = f"""

### SINGLE MUTANT CONTEXT
The ground-truth single-mutant DMS scores of every component appearing in any
candidate are intentionally visible model inputs. Use them as context for
ranking combo candidates, while keeping combo candidate fitness hidden.
Candidates may contain different numbers of component mutations. Do not rank
combos by simply summing these single-mutant scores; that would mostly reward
longer candidates. Use the single-mutant scores as features and adjust for
combo length, per-mutation quality, weak components, and likely epistatic or
structural incompatibilities.

{format_single_context(query.single_context)}
"""

    final_count = 40 if query.task == TASK_BY_ID["1"] else len(query.candidates)
    if query.task == TASK_BY_ID["1"]:
        mutation_constraints = f"""
- Return distinct valid single substitutions only, in 1-indexed WTposMUT format, such as `H24R` or `A15V`.
- Both WT and MUT must be one of the 20 standard amino-acid single-letter codes: {aa_alphabet}.
- Every mutation position must satisfy `1 <= position <= {len(record.wildtype_sequence)}`.
- The WT letter must strictly equal `wildtype_sequence[position - 1]` from the sequence printed above.
- MUT must be different from WT; do not output synonymous/no-op mutations.
- Do not output multi-site mutations, insertions, deletions, stop codons, HGVS notation, scores, or annotations.
- Consider all valid single amino-acid substitutions across the full wild-type sequence, then return only the top {final_count}.
- Before finalizing, explicitly validate the full ranking against the supplied wild-type sequence and remove or replace any invalid item.
""".strip()
    elif False:
        mutation_constraints = (
            "Copy each condition assay ID exactly as supplied, include every candidate exactly once, and do not invent candidates."
        )
    elif query.task == TASK_BY_ID["3"]:
        mutation_constraints = f"""
- Closed-set rule: rank only the supplied multi-mutation candidates; do not introduce new mutations, insertions, deletions, wild-type strings, or reformatted candidates.
- Treat each supplied string as one opaque closed-set multi-mutation candidate that includes the anchor.
- Copy each supplied candidate string exactly as provided.
- Include every supplied candidate exactly once; no duplicates and no omissions.
- Within each component mutation, positions must be within `1 <= position <= {len(record.wildtype_sequence)}`, WT must match `wildtype_sequence[position - 1]`, and MUT must differ from WT.
- Before finalizing, verify that the output ranking is a permutation of the supplied candidate list.
""".strip()
    elif query.task == TASK_BY_ID["4"]:
        mutation_constraints = f"""
- Closed-set rule: rank only the supplied multi-mutation combo candidates; do not introduce new mutations, insertions, deletions, wild-type strings, or reformatted candidates.
- Treat each supplied string as one opaque closed-set combo candidate.
- Copy each supplied candidate string exactly as provided.
- Include every supplied candidate exactly once; no duplicates and no omissions.
- Within each component mutation, positions must be within `1 <= position <= {len(record.wildtype_sequence)}`, WT must match `wildtype_sequence[position - 1]`, and MUT must differ from WT.
- Do not make the final order a raw sum of visible single-mutant scores; use a length-aware and epistasis-aware ranking.
- Before finalizing, verify that the output ranking is a permutation of the supplied candidate list.
""".strip()
    else:
        mutation_constraints = f"""
- Closed-set rule: rank only the supplied candidate strings; do not introduce new mutations, insertions, deletions, wild-type strings, or reformatted candidates.
- Copy each candidate string exactly as supplied.
- Include every supplied candidate exactly once; no duplicates and no omissions.
- If a candidate is a mutation string, each component position must be within `1 <= position <= {len(record.wildtype_sequence)}`, WT must match `wildtype_sequence[position - 1]`, and MUT must differ from WT.
- Before finalizing, verify that the output ranking is a permutation of the supplied candidate list.
""".strip()
    tool_requirement = (
        "You must use at least one `<execute>` block and observe its result before the final `<solution>`."
        if require_tool_use and max_tool_rounds > 0
        else "Use `<execute>` blocks when they provide useful evidence before the final `<solution>`."
    )

    return f"""
You are Biomni A1 acting as the sole ranking agent for one mutation benchmark query.

### IMPORTANT RUNNING MODE
Do not write one fixed batch script that calls a preset list of tools and then delegates ranking to another worker LLM.
You must do autonomous multi-turn reasoning for this single query:
1. Decide which Biomni/database/domain/wetlab/protein tools are useful.
2. Use focused Python <execute> blocks to call tools or inspect public tool outputs.
3. After each observation, decide whether another tool call is needed.
4. Stop tool use when the evidence is sufficient, then produce the ranking yourself.
{tool_requirement}

You may call Biomni tools from Python, for example tools under biomni.tool.database,
biomni.tool.glycoengineering, biomni.tool.protocols, biomni.tool.literature, or other
available biomni.tool modules when useful. Tool choices are up to you. Use at most
{max_tool_rounds} execute/observation rounds. One `<execute>` block may call multiple tools
or run a compact multi-step script when that is more efficient.
If you use torch/ESM/deep-learning tools, explicitly use CUDA when available
(`device = "cuda" if torch.cuda.is_available() else "cpu"`, move models and tensors
to that device, and use `torch.no_grad()` for inference).
When scoring amino acids with a Transformers ESM tokenizer, never assume amino
acid letters occupy logits indices 0-19. Convert each residue with
`tokenizer.convert_tokens_to_ids(residue)` and use those token ids for logits.
The GPU worker is run-only and offline: do not install packages or download models inside
`<execute>` blocks. Use existing shared model/cache paths exposed in environment variables:
`BIOMNI_ESM2_MODEL_PATH`, `BIOMNI_ESM1V_MODEL_DIR`, `BIOMNI_ESM_REPO`,
`BIOMNI_ESMFOLD_MODEL_PATH`, `BIOMNI_CHATNT_MODEL_PATH`, and `HF_HOME`. For ESM2 with Transformers, load from
`os.environ["BIOMNI_ESM2_MODEL_PATH"]` with `local_files_only=True` instead of
`facebook/esm*` model ids.
For ESM-1v `.pt` checkpoints under `BIOMNI_ESM1V_MODEL_DIR`, PyTorch 2.6+ may
require `torch.load(path, map_location="cpu", weights_only=False)` or
`esm.pretrained.load_model_and_alphabet(path)`. Avoid spending repeated rounds
on checkpoint-loading variants once one reliable method is available.
The current working directory for execute blocks is
`os.environ["BIOMNI_AGENT_ARTIFACT_DIR"]` ({prompt_artifact_dir}).
Keep scratch files and downloaded/generated structure files there.
Keep AlphaFold-related downloaded/generated files under
`os.environ["BIOMNI_ALPHAFOLD_ARTIFACT_DIR"]`.
When calling `query_alphafold(..., download=True)`, omit `output_dir` or pass a
subdirectory inside that artifact directory; do not create `af_*` or `AF_*` files in
the project root.

### STRICT DATA BOUNDARIES
- Do not read local benchmark files, norm_data files, labels, DMS_score, DMS_score_raw, candidate_label, or hidden answer columns.
- The only exception is the T3 anchor DMS score or T4 single-mutant DMS scores already printed in this prompt; do not inspect local files to recover any additional scores.
- Do not call any external LLM or worker model.
- Use only the assay fields shown below, the supplied candidate/context fields if present, and public/derived evidence from tools you actually call.
- Never claim tool evidence unless it appeared in an observation.

### ASSAY CONTEXT
- UniProt ID: {record.uniprot_id}
- Primary Task Class: {record.primary_task_class}
- Fitness Metric Type: {record.fitness_type}
- Assay Readout Subclass: {record.readout_subclass}
- Sequence Length: {len(record.wildtype_sequence)}

### WILD-TYPE SEQUENCE
`{record.wildtype_sequence}`
{condition_block}
{anchor_block}
{single_context_block}
{candidate_block}
### TASK GUIDANCE
{task_specific_guidance(query)}

### OUTPUT CONSTRAINTS
- Return exactly {final_count} items in final ranking.
{mutation_constraints}
- Return no scores, explanations, labels, or markdown in the final answer.

### FINAL RESPONSE FORMAT
When ready, respond with one <solution> block containing only this valid JSON object:
<solution>
{{"ranking": ["best_mutation_or_candidate", "second_best", "..."]}}
</solution>
""".strip()


def extract_tool_summary(log_items: list[Any], final: str) -> dict[str, Any]:
    text = "\n\n".join(str(item) for item in log_items) + "\n\n" + str(final)
    tools = sorted({match.group(1) for match in TOOL_NAME_RE.finditer(text)})
    return {
        "num_execute_blocks": len(
            re.findall(r"<execute\b|<invoke\s+name=[\"']execute[\"']", text, flags=re.IGNORECASE)
        ),
        "num_observation_blocks": len(re.findall(r"<observation\b", text, flags=re.IGNORECASE)),
        "num_solution_blocks": len(re.findall(r"<solution>", text, flags=re.IGNORECASE)),
        "tool_mentions": tools,
    }


def run_agent_go(
    agent: Any,
    prompt: str,
    *,
    max_tool_rounds: int,
    enforce_max_tool_rounds: bool,
    require_tool_use: bool,
) -> tuple[list[Any], str]:
    if not enforce_max_tool_rounds:
        return agent.go(prompt)

    from biomni.utils import pretty_print
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    agent.critic_count = 0
    agent.user_task = prompt

    if agent.use_tool_retriever:
        selected_resources_names = agent._prepare_resources_for_retrieval(prompt)
        agent.update_system_prompt_with_selected_resources(selected_resources_names)

    def normalize_response_content(content: Any) -> str:
        if isinstance(content, list):
            text_parts: list[str] = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = block.get("type")
                if block_type in ("text", "output_text", "redacted_text"):
                    part = block.get("text") or block.get("content") or ""
                    if isinstance(part, str):
                        text_parts.append(part)
            return "".join(text_parts)
        return str(content)

    def close_incomplete_tags(text: str) -> str:
        for tag in ("execute", "solution", "think"):
            if f"<{tag}>" in text and f"</{tag}>" not in text:
                text += f"</{tag}>"
        return text

    def extract_execute_payload(text: str) -> str | None:
        execute_match = re.search(r"<execute>(.*?)</execute>", text, re.DOTALL | re.IGNORECASE)
        if execute_match:
            return execute_match.group(1)

        invoke_match = re.search(
            r"<invoke\s+name=[\"']execute[\"']\s*>\s*"
            r"<parameter\s+name=[\"']code[\"']\s*>(.*?)</parameter>\s*"
            r"</invoke>",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if invoke_match:
            return invoke_match.group(1)

        code_block_match = re.search(r"```(?:python|bash|r)?\s*(.*?)```", text, re.DOTALL)
        if code_block_match and not re.search(r"<solution>(.*?)</solution>", text, re.DOTALL | re.IGNORECASE):
            return code_block_match.group(1)

        return None

    def normalize_execute_payload(code: str) -> str:
        stripped = code.strip()
        while re.match(r"^<execute\b[^>]*>", stripped, flags=re.IGNORECASE):
            stripped = re.sub(r"^<execute\b[^>]*>", "", stripped, count=1, flags=re.IGNORECASE).strip()
        while re.search(r"</execute>\s*$", stripped, flags=re.IGNORECASE):
            stripped = re.sub(r"</execute>\s*$", "", stripped, count=1, flags=re.IGNORECASE).strip()
        return stripped

    def requires_user_role_observation(llm: Any) -> bool:
        model_name = str(getattr(llm, "model_name", getattr(llm, "model", ""))).lower()
        llm_type = str(type(llm)).lower()
        return "claude" in model_name or "anthropic" in model_name or "bedrock" in llm_type

    def execute_code(code: str) -> str:
        import biomni.agent.a1 as a1_module

        timeout = agent.timeout_seconds
        stripped = normalize_execute_payload(code)
        if (
            stripped.startswith("#!R")
            or stripped.startswith("# R code")
            or stripped.startswith("# R script")
        ):
            r_code = re.sub(r"^#!R|^# R code|^# R script", "", stripped, count=1).strip()
            result = a1_module.run_with_timeout(a1_module.run_r_code, [r_code], timeout=timeout)
        elif (
            stripped.startswith("#!BASH")
            or stripped.startswith("# Bash script")
            or stripped.startswith("#!CLI")
        ):
            if stripped.startswith("#!CLI"):
                cli_command = re.sub(r"^#!CLI", "", stripped, count=1).strip().replace("\n", " ")
                result = a1_module.run_with_timeout(a1_module.run_bash_script, [cli_command], timeout=timeout)
            else:
                bash_script = re.sub(r"^#!BASH|^# Bash script", "", stripped, count=1).strip()
                result = a1_module.run_with_timeout(a1_module.run_bash_script, [bash_script], timeout=timeout)
        else:
            agent._clear_execution_plots()
            agent._inject_custom_functions_to_repl()
            result = a1_module.run_with_timeout(a1_module.run_python_repl, [stripped], timeout=timeout)

        if len(result) > 10000:
            result = "The output is too long to be added to context. Here are the first 10K characters...\n" + result[:10000]
        return result

    messages: list[Any] = [HumanMessage(content=prompt)]
    agent.log = []
    final = ""
    execute_count = 0
    forced_final = False
    forced_final_attempts = 0
    early_solution_attempts = 0
    parsing_errors = 0
    max_generation_steps = max(8, 2 * max_tool_rounds + 8)
    action_tag_instructions = (
        "IMPORTANT RESPONSE FORMAT: Every assistant response must contain exactly one actionable tag: "
        "either <execute>...</execute> to run code, or <solution>...</solution> to finish. "
        "Do not return only thinking, analysis, markdown, or <details> blocks."
    )
    if require_tool_use and max_tool_rounds > 0:
        action_tag_instructions += (
            " You must run at least one <execute> block and receive an <observation> before producing <solution>."
        )

    for _ in range(max_generation_steps):
        system_prompt = agent.system_prompt + "\n\n" + action_tag_instructions
        if hasattr(agent.llm, "model_name") and (
            "gpt" in str(agent.llm.model_name).lower() or "openai" in str(type(agent.llm)).lower()
        ):
            system_prompt += " Do not use markdown code blocks (```) - use <execute> tags instead."

        response = agent.llm.invoke([SystemMessage(content=system_prompt)] + messages)
        msg = close_incomplete_tags(normalize_response_content(response.content)).strip()

        answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)
        execute_payload = extract_execute_payload(msg)

        ai_message = AIMessage(content=msg)
        messages.append(ai_message)
        agent.log.append(pretty_print(ai_message))

        if answer_match:
            parsing_errors = 0
            if require_tool_use and max_tool_rounds > 0 and execute_count == 0:
                early_solution_attempts += 1
                if early_solution_attempts > 3:
                    raise RuntimeError(
                        "A1 attempted to finish without any required execute/observation round."
                    )
                messages.append(
                    HumanMessage(
                        content=(
                            "Your previous <solution> was rejected because this agent run requires tool use. "
                            "Run one focused <execute> block now and wait for the <observation> before finalizing. "
                            "At minimum, use Python to validate the proposed mutations against the supplied "
                            "wild-type sequence and output constraints; use Biomni/public tools too if useful."
                        )
                    )
                )
                continue
            final = msg
            agent._conversation_state = {"messages": messages, "next_step": "end"}
            return agent.log, final

        if execute_payload is not None:
            parsing_errors = 0
            if execute_count >= max_tool_rounds:
                forced_final_attempts += 1
                if forced_final_attempts > 2:
                    raise RuntimeError(
                        f"A1 exceeded the hard execute round limit ({max_tool_rounds}) "
                        "and did not provide a solution after being instructed to stop tool use."
                    )
                messages.append(
                    HumanMessage(
                        content=(
                            f"You have already used the maximum allowed {max_tool_rounds} execute/observation rounds. "
                            "Do not emit another <execute> block. Produce the final <solution> JSON now, using the "
                            "evidence already observed. If evidence is incomplete, use your best scientific judgment."
                        )
                    )
                )
                forced_final = True
                continue

            execute_count += 1
            result = execute_code(execute_payload)
            observation_message_cls = HumanMessage if requires_user_role_observation(agent.llm) else AIMessage
            observation = observation_message_cls(content=f"\n<observation>{result}</observation>".strip())
            messages.append(observation)
            agent.log.append(pretty_print(observation))

            if execute_count >= max_tool_rounds and not forced_final:
                messages.append(
                    HumanMessage(
                        content=(
                            f"You have now used all {max_tool_rounds} allowed execute/observation rounds. "
                            "Stop tool use. Return the final <solution> JSON only, with the required ranking length. "
                            "Use the evidence already observed; if it is incomplete, use your best scientific judgment."
                        )
                    )
                )
                forced_final = True
            continue

        parsing_errors += 1
        if parsing_errors >= 4:
            raise RuntimeError(
                "A1 produced repeated responses without actionable <execute> or <solution> tags."
            )
        if execute_count > 0 and parsing_errors >= 2:
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous responses contained no actionable tag. Stop free-form thinking now. "
                        "Use the observations already available and produce the final <solution> JSON only. "
                        "Do not emit markdown, <details>, or another thinking-only response."
                    )
                )
            )
            forced_final = True
            continue
        messages.append(
            HumanMessage(
                content=(
                    "Your previous response had no actionable tag. Reply with exactly one of these forms: "
                    "<execute>code to run</execute> or <solution>{\"ranking\": [...]}</solution>. "
                    "If you are ready or tool budget is exhausted, produce <solution> JSON now."
                )
            )
        )

    raise RuntimeError(
        f"A1 did not finish within the controlled generation loop after {max_generation_steps} generation steps."
    )


def complete_prediction(query: NativeQuery, llm_ranking: list[str], top_k: int) -> tuple[list[str], dict[str, int]]:
    if query.task == TASK_BY_ID["1"]:
        seen: set[str] = set()
        ranking: list[str] = []
        invalid = duplicate = not_measured = 0
        for mutant in llm_ranking:
            if mutant in seen:
                duplicate += 1
            elif not valid_mutation(mutant, query.record.wildtype_sequence):
                invalid += 1
            elif query.measured_mutants and mutant not in query.measured_mutants:
                not_measured += 1
            else:
                seen.add(mutant)
                ranking.append(mutant)
            if len(ranking) >= top_k:
                break
        return ranking[:top_k], {"invalid": invalid, "duplicate": duplicate,
                                "not_measured": not_measured, "from_llm": len(ranking)}
    raise ValueError(f"Unsupported task completion for {query.task}")


def run_one_query(query: NativeQuery, args: argparse.Namespace, api_key: str | None, log_handle: Any) -> dict[str, Any]:
    from biomni.agent import A1
    import biomni.agent.a1 as a1_module
    try:
        from biomni_benchmark.multi_turn_agent.gpu_dispatch import (
            install_gpu_dispatch_hooks,
            install_python_repl_dispatch_hook,
            parse_function_list,
        )
        from biomni_benchmark.multi_turn_agent.gpu_runtime_env import apply_runtime_environment
    except ImportError:
        from gpu_dispatch import (
            install_gpu_dispatch_hooks,
            install_python_repl_dispatch_hook,
            parse_function_list,
        )
        from gpu_runtime_env import apply_runtime_environment

    previous_agent_artifact_dir = os.environ.get("BIOMNI_AGENT_ARTIFACT_DIR")
    previous_alphafold_artifact_dir = os.environ.get("BIOMNI_ALPHAFOLD_ARTIFACT_DIR")
    artifact_base_root = Path(getattr(args, "agent_artifact_base_dir", None) or previous_agent_artifact_dir or DEFAULT_AGENT_ARTIFACT_DIR)
    artifact_run_root = resolve_artifact_run_root(
        artifact_base_root,
        getattr(args, "artifact_run_id", build_artifact_run_id(query.task)),
    )
    artifact_dir = query_artifact_dir(query, artifact_run_root).resolve()
    alphafold_artifact_dir = artifact_dir / "alphafold"
    os.environ["BIOMNI_AGENT_ARTIFACT_DIR"] = str(artifact_dir)
    os.environ["BIOMNI_ALPHAFOLD_ARTIFACT_DIR"] = str(alphafold_artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    alphafold_artifact_dir.mkdir(parents=True, exist_ok=True)
    ensure_pythonpath_entries(ROOT, ROOT / "biomni")

    prompt = build_agent_prompt(
        query,
        max_tool_rounds=args.max_tool_rounds,
        require_tool_use=args.require_tool_use,
    )
    final = ""
    log_items: list[Any] = []
    error: str | None = None
    agent: Any | None = None
    started = time.monotonic()

    try:
        with wall_time_limit(args.timeout_seconds, f"query index={query.index} assay={query.assay}"):
            print("\n" + "=" * 80, file=log_handle, flush=True)
            print(f"QUERY index={query.index} task={query.task} assay={query.assay}", file=log_handle, flush=True)
            print("=" * 80, file=log_handle, flush=True)
            flushing_log = FlushWriter(log_handle)
            with (
                contextlib.redirect_stdout(flushing_log),
                contextlib.redirect_stderr(flushing_log),
                working_directory(artifact_dir),
            ):
                apply_runtime_environment()
                print(f"AGENT_ARTIFACT_DIR {artifact_dir}", flush=True)
                installed_gpu_hooks = install_gpu_dispatch_hooks(
                    enabled=args.gpu_dispatch,
                    base_url=args.gpu_dispatch_base_url,
                    jobs_dir=args.gpu_dispatch_jobs_dir,
                    timeout_seconds=args.gpu_dispatch_timeout_seconds or args.timeout_seconds,
                    functions=parse_function_list(args.gpu_dispatch_functions),
                )
                python_repl_hook = install_python_repl_dispatch_hook(
                    enabled=args.gpu_dispatch and args.gpu_dispatch_code,
                    base_url=args.gpu_dispatch_base_url,
                    jobs_dir=args.gpu_dispatch_jobs_dir,
                    timeout_seconds=args.gpu_dispatch_timeout_seconds or args.timeout_seconds,
                    session_id=(
                        f"{getattr(args, 'artifact_run_id', 'run')}_"
                        f"{query.task}_{query.index}_{os.getpid()}"
                    ),
                )
                if installed_gpu_hooks:
                    print(
                        "GPU_DISPATCH_HOOKS "
                        + ", ".join(installed_gpu_hooks),
                        flush=True,
                    )
                if python_repl_hook:
                    print("GPU_DISPATCH_PYTHON_REPL_HOOK enabled", flush=True)
                print("A1_INIT_START", flush=True)
                original_get_llm, patched_get_llm = install_custom_get_llm_patch(a1_module, args)
                if patched_get_llm:
                    deepseek_patch = is_deepseek_model(args.agent_llm) and args.deepseek_thinking
                    print(
                        "LLM_INFERENCE custom_max_tokens_patch=True "
                        f"max_tokens={args.agent_max_tokens} "
                        f"deepseek_thinking={deepseek_patch}"
                        + (
                            " "
                            f"thinking_type={args.deepseek_thinking_type} "
                            f"reasoning_effort={args.deepseek_reasoning_effort} "
                            "temperature=omitted"
                            if deepseek_patch
                            else ""
                        ),
                        flush=True,
                    )
                try:
                    agent = A1(
                        path=str(Path(args.a1_data_path).resolve()),
                        llm=args.agent_llm,
                        source=args.agent_source,
                        base_url=args.agent_base_url,
                        api_key=api_key,
                        expected_data_lake_files=[],
                        use_tool_retriever=args.use_tool_retriever,
                        timeout_seconds=args.timeout_seconds,
                    )
                finally:
                    if patched_get_llm:
                        a1_module.get_llm = original_get_llm
                print("A1_INIT_DONE", flush=True)
                print("AGENT_GO_START", flush=True)
                log_items, final = run_agent_go(
                    agent,
                    prompt,
                    max_tool_rounds=args.max_tool_rounds,
                    enforce_max_tool_rounds=args.enforce_max_tool_rounds,
                    require_tool_use=args.require_tool_use,
                )
                print("AGENT_GO_DONE", flush=True)
    except KeyboardInterrupt:
        raise
    except BaseException:
        if agent is not None and getattr(agent, "log", None):
            log_items = list(agent.log)
        error = traceback.format_exc()
        print("\nERROR TRACEBACK:", file=log_handle, flush=True)
        print(error, file=log_handle, flush=True)
        if args.fail_fast:
            raise
    finally:
        if previous_agent_artifact_dir is None:
            os.environ.pop("BIOMNI_AGENT_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMNI_AGENT_ARTIFACT_DIR"] = previous_agent_artifact_dir
        if previous_alphafold_artifact_dir is None:
            os.environ.pop("BIOMNI_ALPHAFOLD_ARTIFACT_DIR", None)
        else:
            os.environ["BIOMNI_ALPHAFOLD_ARTIFACT_DIR"] = previous_alphafold_artifact_dir

    llm_ranking = parse_json_ranking(final)
    final_ranking, quality = complete_prediction(query, llm_ranking, top_k=args.top_k)
    tool_summary = extract_tool_summary(log_items, final)
    elapsed = time.monotonic() - started

    return {
        "index": query.index,
        "task": query.task,
        "eval_setting": EVAL_SETTINGS[query.task],
        "assay": query.assay,
        "source_assay": query.source_assay,
        "candidate_group_id": query.candidate_group_id,
        "condition_a": query.condition_a,
        "condition_b": query.condition_b,
        "target_mutation": query.target_mutation,
        "anchor_mutant": query.anchor_mutant,
        "anchor_dms_score": query.anchor_dms_score,
        "num_single_context": len(query.single_context),
        "artifact_dir": str(artifact_dir),
        "alphafold_artifact_dir": str(alphafold_artifact_dir),
        "source_path": query.source_path,
        "num_candidates": len(query.candidates),
        "num_prompt_ranking": len(llm_ranking),
        "num_final": len(final_ranking),
        "quality": quality,
        "agent_llm": args.agent_llm,
        "agent_source": args.agent_source,
        "agent_inference": {
            "max_tokens": args.agent_max_tokens,
            "deepseek_thinking": bool(args.deepseek_thinking and is_deepseek_model(args.agent_llm)),
            "deepseek_thinking_type": args.deepseek_thinking_type if is_deepseek_model(args.agent_llm) else None,
            "deepseek_reasoning_effort": args.deepseek_reasoning_effort if is_deepseek_model(args.agent_llm) else None,
            "temperature": None if args.deepseek_thinking and is_deepseek_model(args.agent_llm) else "biomni_default",
            "top_p": None,
            "stream": False,
        },
        "max_tool_rounds": args.max_tool_rounds,
        "enforce_max_tool_rounds": args.enforce_max_tool_rounds,
        "require_tool_use": args.require_tool_use,
        "elapsed_seconds": round(elapsed, 3),
        "agent_tool_summary": tool_summary,
        "error": error,
        "raw_response": final,
        "_final_ranking": final_ranking,
    }


def default_output_paths(task: str) -> tuple[Path, Path, Path]:
    result_dir = DEFAULT_RESULT_ROOT / task
    prefix = result_dir / f"{task}_agent_native_deepseek_v4_pro"
    return (
        prefix.with_name(prefix.name + "_predictions.csv"),
        prefix.with_name(prefix.name + "_audit.jsonl"),
        prefix.with_name(prefix.name + "_agent_log.txt"),
    )


def parse_args(task: str) -> argparse.Namespace:
    default_pred, default_audit, default_log = default_output_paths(task)
    parser = argparse.ArgumentParser(description=f"Agent-native Biomni runner for {task}: {TASK_NAMES[task]}")
    parser.add_argument(f"--{task}-dir", dest="task_dir", default=str(DEFAULT_DIRS[task]))
    parser.add_argument("--data-dir", dest="task_dir_override", default=None)
    parser.add_argument("--output-csv", default=str(default_pred))
    parser.add_argument("--audit-jsonl", default=str(default_audit))
    parser.add_argument("--a1-log", default=str(default_log))
    parser.add_argument("--a1-data-path", default=str(DEFAULT_A1_DATA))
    parser.add_argument(
        "--agent-artifact-dir",
        dest="agent_artifact_base_dir",
        default=os.getenv("BIOMNI_AGENT_ARTIFACT_DIR", str(DEFAULT_AGENT_ARTIFACT_DIR)),
        help="Base artifact directory. Each process writes under runs/<artifact-run-id>/queries/...",
    )
    parser.add_argument(
        "--artifact-run-id",
        default=os.getenv("BIOMNI_AGENT_ARTIFACT_RUN_ID"),
        help="Optional stable run id for artifact isolation. Defaults to task_timestamp_pid.",
    )
    parser.add_argument("--agent-llm", default=os.getenv("BIOMNI_AGENT_LLM", "deepseek-v4-pro"))
    parser.add_argument("--agent-source", default=os.getenv("BIOMNI_AGENT_SOURCE", "Custom"))
    parser.add_argument("--agent-base-url", default=os.getenv("BIOMNI_AGENT_BASE_URL"))
    parser.add_argument("--agent-api-key-env", default=os.getenv("BIOMNI_AGENT_API_KEY_ENV", "OPENAI_API_KEY"))
    parser.add_argument(
        "--agent-max-tokens",
        type=int,
        default=int(os.getenv("BIOMNI_AGENT_MAX_TOKENS", "8192")),
        help="Maximum output tokens for OpenAI-compatible Custom LLM calls.",
    )
    parser.add_argument(
        "--deepseek-thinking",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("BIOMNI_DEEPSEEK_THINKING", "1").strip().lower() in {"1", "true", "yes", "on"},
        help="For Custom DeepSeek models, enable thinking mode via extra_body.thinking.",
    )
    parser.add_argument(
        "--deepseek-thinking-type",
        choices=("enabled", "disabled"),
        default=os.getenv("BIOMNI_DEEPSEEK_THINKING_TYPE", "enabled"),
    )
    parser.add_argument(
        "--deepseek-reasoning-effort",
        choices=("high", "max"),
        default=os.getenv("BIOMNI_DEEPSEEK_REASONING_EFFORT", "high"),
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-tool-rounds", type=int, default=40)
    parser.add_argument(
        "--enforce-max-tool-rounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Hard-limit the A1 graph recursion to approximately --max-tool-rounds "
            "execute/observation cycles plus one final solution generation."
        ),
    )
    parser.add_argument(
        "--require-tool-use",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require at least one execute/observation round before accepting the final solution "
            "when the controlled loop is enabled."
        ),
    )
    parser.add_argument("--timeout-seconds", type=int, default=1200)
    parser.add_argument("--progress-interval", type=float, default=5.0)
    parser.add_argument("--use-tool-retriever", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gpu-dispatch",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("BIOMNI_GPU_DISPATCH_ENABLED", "0").strip().lower()
        in {"1", "true", "yes", "on"},
        help="Forward known GPU-heavy Biomni tools to the remote GPU dispatcher.",
    )
    parser.add_argument(
        "--gpu-dispatch-base-url",
        default=os.getenv("BIOMNI_GPU_DISPATCH_BASE_URL", "http://127.0.0.1:25000"),
    )
    parser.add_argument(
        "--gpu-dispatch-jobs-dir",
        default=os.getenv(
            "BIOMNI_GPU_DISPATCH_JOBS_DIR",
            str(DEFAULT_GPU_DISPATCH_JOBS_DIR),
        ),
    )
    parser.add_argument(
        "--gpu-dispatch-timeout-seconds",
        type=int,
        default=int(os.getenv("BIOMNI_GPU_DISPATCH_TIMEOUT_SECONDS", "0")),
        help="Remote GPU job timeout. Use 0 to inherit --timeout-seconds.",
    )
    parser.add_argument(
        "--gpu-dispatch-functions",
        default=os.getenv("BIOMNI_GPU_DISPATCH_FUNCTIONS"),
        help="Comma-separated fully-qualified Biomni functions to dispatch; defaults to built-in GPU-heavy tools.",
    )
    parser.add_argument(
        "--gpu-dispatch-code",
        action=argparse.BooleanOptionalAction,
        default=os.getenv("BIOMNI_GPU_DISPATCH_CODE_ENABLED", "1").strip().lower()
        in {"1", "true", "yes", "on"},
        help=(
            "Forward A1 Python/Bash execute blocks to the GPU dispatcher by default, "
            "except light snippets and network/database I/O."
        ),
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Append to existing outputs and skip queries already present in the audit JSONL.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    if task in {TASK_BY_ID["3"], TASK_BY_ID["4"]}:
        parser.add_argument("--start-group-index", type=int, default=None)
        parser.add_argument("--max-groups", dest="max_items", type=int, default=None)
        parser.add_argument("--max-assays", type=int, default=None)
        parser.add_argument("--start-assay-index", type=int, default=0)
    else:
        parser.set_defaults(start_group_index=None, max_assays=None, start_assay_index=0)
    args = parser.parse_args()
    if args.task_dir_override:
        args.task_dir = args.task_dir_override
    args.agent_base_url = normalize_openai_base_url(
        args.agent_base_url or os.environ.get("ANTHROPIC_BASE_URL")
    )
    if not args.artifact_run_id:
        args.artifact_run_id = build_artifact_run_id(task)
    else:
        args.artifact_run_id = safe_path_component(args.artifact_run_id, max_length=120)
    return args


def main_for_task(task: str) -> int:
    if task not in DEFAULT_DIRS:
        raise ValueError(f"Unsupported task: {task}")
    prime_provider_env()
    args = parse_args(task)
    data_dir = Path(args.task_dir)
    queries = build_queries(task, data_dir, args)

    print(f"task={task} setting={EVAL_SETTINGS[task]}")
    print(f"data_dir={data_dir}")
    print(f"num_queries={len(queries)}")
    print(f"agent_llm={args.agent_llm} source={args.agent_source} base_url={args.agent_base_url}")
    print(
        f"agent_max_tokens={args.agent_max_tokens} "
        f"deepseek_thinking={args.deepseek_thinking if is_deepseek_model(args.agent_llm) else 'n/a'} "
        f"deepseek_thinking_type={args.deepseek_thinking_type if is_deepseek_model(args.agent_llm) else 'n/a'} "
        f"deepseek_reasoning_effort={args.deepseek_reasoning_effort if is_deepseek_model(args.agent_llm) else 'n/a'}"
    )
    api_key, api_key_env = resolve_api_key(args.agent_api_key_env)
    configure_biomni_default_llm(
        model=args.agent_llm,
        source=args.agent_source,
        base_url=args.agent_base_url,
        api_key=api_key,
    )
    print(f"agent_api_key_configured={'yes' if api_key else 'no'}")
    print(
        f"use_tool_retriever={args.use_tool_retriever} "
        f"max_tool_rounds={args.max_tool_rounds} "
        f"require_tool_use={args.require_tool_use}"
    )
    artifact_run_root = resolve_artifact_run_root(Path(args.agent_artifact_base_dir), args.artifact_run_id)
    print(f"artifact_run_id={args.artifact_run_id}")
    print(f"artifact_run_root={artifact_run_root}")
    print(
        f"gpu_dispatch={args.gpu_dispatch} "
        f"base_url={args.gpu_dispatch_base_url if args.gpu_dispatch else 'disabled'} "
        f"jobs_dir={args.gpu_dispatch_jobs_dir} "
        f"gpu_timeout={args.gpu_dispatch_timeout_seconds or args.timeout_seconds} "
        f"gpu_code_dispatch={args.gpu_dispatch_code}"
    )

    if args.dry_run:
        for query in queries[:3]:
            dry_artifact_dir = query_artifact_dir(query, artifact_run_root).resolve()
            dry_alphafold_dir = dry_artifact_dir / "alphafold"
            print(f"dry_run_query index={query.index} assay={query.assay} candidates={len(query.candidates)}")
            with temporary_env(
                {
                    "BIOMNI_AGENT_ARTIFACT_DIR": str(dry_artifact_dir),
                    "BIOMNI_ALPHAFOLD_ARTIFACT_DIR": str(dry_alphafold_dir),
                }
            ):
                print(
                    build_agent_prompt(
                        query,
                        max_tool_rounds=args.max_tool_rounds,
                        require_tool_use=args.require_tool_use,
                    )[:2500]
                )
            print("---")
        return 0

    output_csv = Path(args.output_csv).resolve()
    audit_jsonl = Path(args.audit_jsonl).resolve()
    a1_log = Path(args.a1_log).resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    a1_log.parent.mkdir(parents=True, exist_ok=True)

    completed_existing = 0
    if args.resume_existing:
        completed_existing = min(count_text_lines(audit_jsonl), len(queries))
        if completed_existing:
            print(
                f"resume_existing completed={completed_existing}/{len(queries)} "
                f"next_index={queries[completed_existing].index if completed_existing < len(queries) else 'done'}",
                flush=True,
            )
        if completed_existing >= len(queries):
            print("resume_existing chunk already complete")
            print(f"predictions={output_csv}")
            print(f"audit={audit_jsonl}")
            print(f"a1_log={a1_log}")
            return 0
    else:
        output_csv.unlink(missing_ok=True)
        audit_jsonl.unlink(missing_ok=True)

    started = time.monotonic()
    pred_mode = "a" if args.resume_existing and output_csv.exists() and output_csv.stat().st_size > 0 else "w"
    audit_mode = "a" if args.resume_existing else "w"
    log_mode = "a" if args.resume_existing else "w"
    remaining_queries = queries[completed_existing:]

    prediction_item_column = "mutant"
    log_secrets = collect_log_secrets(api_key)
    with output_csv.open(pred_mode, newline="") as pred_handle, audit_jsonl.open(audit_mode) as audit_handle, a1_log.open(log_mode) as log_handle:
        safe_log_handle = RedactingFlushWriter(log_handle, log_secrets)
        writer = csv.DictWriter(pred_handle, fieldnames=["assay", prediction_item_column, "rank"])
        if pred_mode == "w":
            writer.writeheader()
        if log_mode == "a" and a1_log.exists() and a1_log.stat().st_size > 0:
            safe_log_handle.write(
                "\n\n"
                f"===== RESUME at {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"completed={completed_existing}/{len(queries)} =====\n"
            )
            safe_log_handle.flush()
        for offset, query in enumerate(remaining_queries, start=1):
            done = completed_existing + offset
            print(
                f"query_start {done}/{len(queries)} index={query.index} "
                f"task={query.task} assay={query.assay}",
                flush=True,
            )
            audit_row = run_one_query(query, args, api_key, safe_log_handle)
            final_ranking = audit_row.pop("_final_ranking")
            for rank, item in enumerate(final_ranking, start=1):
                writer.writerow({"assay": query.assay, prediction_item_column: item, "rank": rank})
            audit_handle.write(json.dumps(audit_row, ensure_ascii=False) + "\n")
            audit_handle.flush()
            pred_handle.flush()
            if args.progress_interval >= 0:
                print(render_progress(done, len(queries), time.monotonic() - started), end="", flush=True)
        print()

    print(f"predictions={output_csv}")
    print(f"audit={audit_jsonl}")
    print(f"a1_log={a1_log}")
    return 0
