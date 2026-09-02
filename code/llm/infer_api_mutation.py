import argparse
import asyncio
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm

from openai import AsyncOpenAI


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from router import get_default_input_paths, get_prompt_template
from llm.settings import API_BASE_URL, API_KEY, API_MAX_RETRIES, API_TIMEOUT

DEFAULT_INPUT_ROOT = Path(__file__).resolve().parents[2] / "PFArena"
DEFAULT_INPUT_PATHS = get_default_input_paths(DEFAULT_INPUT_ROOT)
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"


@dataclass(frozen=True)
class AssayContext:
    assay_id: str
    uniprot_id: str
    primary_task_class: str
    fitness_type: str
    readout_subclass: str
    wildtype_sequence: str
    sequence_length: int


@dataclass(frozen=True)
class MutationTask:
    input_path: Path
    output_path: Path
    context: AssayContext
    mutants: tuple[str, ...]
    ground_truth_path: Path | None = None
    anchor_mutant: str = ""
    anchor_dms_score: str = ""
    single_mutant_dms_scores: str = ""


def build_prompt(task: MutationTask, args: argparse.Namespace) -> str:
    template = get_prompt_template(args.task_id)

    candidate_mutants = "\n".join(task.mutants)
    return template.format(
        uniprot_id=task.context.uniprot_id,
        primary_task_class=task.context.primary_task_class,
        fitness_type=task.context.fitness_type,
        readout_subclass=task.context.readout_subclass,
        sequence_length=task.context.sequence_length,
        wildtype_sequence=task.context.wildtype_sequence,
        anchor_mutant=task.anchor_mutant,
        anchor_DMS_score=task.anchor_dms_score,
        single_mutant_dms_scores=task.single_mutant_dms_scores,
        num_candidates=len(task.mutants),
        candidate_mutants=candidate_mutants,
    )


async def create_chat_completion(client: AsyncOpenAI, task: MutationTask, args: argparse.Namespace):
    return await client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": build_prompt(task, args),
            }
        ],
        max_completion_tokens=args.max_completion_tokens,
        stream=False,
    )


def extract_balanced_json(text: str) -> object | None:
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*({[\s\S]*?})\s*```", text, flags=re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    start = None
    stack: list[str] = []
    for index, char in enumerate(text):
        if char == "{":
            if start is None:
                start = index
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    try:
                        return json.loads(text[start : index + 1])
                    except Exception:
                        start = None
                        stack = []

    try:
        return json.loads(text)
    except Exception:
        print("[Exception] Response json format error.")
        return None


def parse_rank_mode_mutants(response_text: str) -> tuple[str, ...] | None:
    parsed = extract_balanced_json(response_text)
    if isinstance(parsed, dict):
        value = parsed.get("ranking")
        if isinstance(value, list):
            return tuple(str(item).strip() for item in value)
    return None


def parse_generation_mode_mutants(response_text: str) -> tuple[str, ...] | None:
    parsed = extract_balanced_json(response_text)
    if isinstance(parsed, dict):
        value = parsed.get("ranking")
        if isinstance(value, list):
            return tuple(str(item).strip() for item in value)
    return None


def parse_prediction(response_text: str, args: argparse.Namespace) -> tuple[str, ...] | None:
    if args.task_id == "1":
        return parse_generation_mode_mutants(response_text)
    elif args.task_id in ("2", "3", "4"):
        return parse_rank_mode_mutants(response_text)


def read_mutants(path: Path) -> tuple[str, ...]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        mutants = tuple(row["mutant"].strip() for row in reader if row["mutant"].strip())
    if not mutants:
        raise ValueError(f"{path} has no candidate mutants")
    return mutants


def context_from_assay_row(row: dict[str, str], assay_csv_path: Path) -> AssayContext:
    return AssayContext(
        assay_id=row["assay_id"].strip(),
        uniprot_id=row["uniprot_id"].strip(),
        primary_task_class=row["primary_task_class"].strip(),
        fitness_type=row["fitness_type"].strip(),
        readout_subclass=row["readout_subclass"].strip(),
        wildtype_sequence=row["wildtype_sequence"].strip(),
        sequence_length=int(float(row["sequence_length"])),
    )


def read_assay_contexts(assay_csv_path: Path) -> dict[str, AssayContext]:
    contexts: dict[str, AssayContext] = {}
    with assay_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "assay_id" not in reader.fieldnames:
            raise ValueError(f"{assay_csv_path} missing required assay_id column")
        for row in reader:
            context = context_from_assay_row(row, assay_csv_path)
            contexts[context.assay_id] = context
    if not contexts:
        raise ValueError(f"{assay_csv_path} has no assay rows")
    return contexts


def relative_output_path(input_path: Path, input_dir: Path, args: argparse.Namespace) -> Path:
    relative_path = input_path.relative_to(input_dir)
    return args.output_root / args.model / output_task_prefix(input_dir) / relative_path


def output_task_prefix(input_dir: Path) -> Path:
    path = input_dir.resolve()
    parts = path.parts
    task_index = next((index for index, part in enumerate(parts) if re.fullmatch(r"T\d+.*", part)), None)
    if task_index is None:
        raise ValueError(f"Could not find task directory beginning with T<X> in input path: {input_dir}")

    task_parts = parts[task_index:]
    if path.suffix.lower() == ".csv":
        task_parts = task_parts[:-1]
    if not task_parts:
        raise ValueError(f"Could not derive output task prefix from input path: {input_dir}")
    return Path(*task_parts)


def mutation_generation_relative_path(row: dict[str, str], context: AssayContext) -> Path:
    dms_filename = (row.get("DMS_filename") or "").strip()
    if dms_filename:
        return Path(dms_filename)
    return Path("norm_data") / f"{context.assay_id}.csv"


def mutation_generation_output_path(assay_csv_path: Path, row: dict[str, str], context: AssayContext, args: argparse.Namespace) -> Path:
    relative_path = mutation_generation_relative_path(row, context)
    return args.output_root / args.model / output_task_prefix(assay_csv_path) / relative_path


def read_candidate_assay_index(assay_csv_path: Path) -> dict[Path, dict[str, str]]:
    with assay_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            (assay_csv_path.parent / row["DMS_filename"].strip()).resolve(): row
            for row in reader
        }


def read_single_mutant_dms_scores(path: Path) -> str:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        lines = [
            f"{row['mutant'].strip()}: {row['DMS_score'].strip()}"
            for row in reader
        ]
    return "\n".join(lines)


def rank_assay_csv_path(input_dir: Path) -> Path:
    parts = input_dir.parts
    if "norm_data" not in parts:
        raise ValueError(f"Ranking tasks expects norm_data or an assay directory under norm_data: {input_dir}")
    norm_data_index = parts.index("norm_data")
    return Path(*parts[:norm_data_index]) / "assay.csv"


def discover_generation_tasks(args: argparse.Namespace) -> list[MutationTask]:
    assay_csv_path = args.input_dir.resolve()
    if not assay_csv_path.exists():
        raise FileNotFoundError(f"Full Generation tasks input assay.csv does not exist: {assay_csv_path}")
    if not assay_csv_path.is_file():
        raise IsADirectoryError(f"Task 1 input must be an assay.csv file: {assay_csv_path}")

    tasks: list[MutationTask] = []
    with assay_csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "assay_id" not in reader.fieldnames:
            raise ValueError(f"{assay_csv_path} missing required assay_id column")
        for row in reader:
            context = context_from_assay_row(row, assay_csv_path)
            output_path = mutation_generation_output_path(assay_csv_path, row, context, args)
            ground_truth_path = assay_csv_path.parent / mutation_generation_relative_path(row, context)

            if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
                continue
            tasks.append(
                MutationTask(
                    input_path=assay_csv_path,
                    output_path=output_path,
                    context=context,
                    mutants=(),
                    ground_truth_path=ground_truth_path,
                )
            )
            if args.max_files is not None and len(tasks) >= args.max_files:
                break
    return tasks

def discover_rank_tasks(args: argparse.Namespace) -> list[MutationTask]:
    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Ranking tasks expects norm_data or an assay directory under norm_data: {input_dir}")

    assay_csv_path = rank_assay_csv_path(input_dir)
    contexts = read_assay_contexts(assay_csv_path)
    candidate_metadata = read_candidate_assay_index(assay_csv_path)

    tasks: list[MutationTask] = []
    for candidate_path in sorted(input_dir.rglob("*.csv")):
        candidate_path = candidate_path.resolve()
        metadata = candidate_metadata.get(candidate_path)
        if metadata is None:
            continue  # Ignore auxiliary CSVs such as single_mutant_context.csv.

        assay_id = metadata["assay_id"].strip()
        context = contexts[assay_id]
        output_path = relative_output_path(candidate_path, input_dir, args)

        if output_path.exists() and output_path.stat().st_size > 0 and not args.force:
            continue

        anchor_mutant, anchor_dms_score, single_mutant_dms_scores = None, None, None

        if args.task_id == "3":
            anchor_mutant = metadata["anchor_mutant"].strip()
            anchor_dms_score = metadata["anchor_DMS_score"].strip()
        elif args.task_id == "4":
            context_path = candidate_path.parent / "single_mutant_context.csv"
            single_mutant_dms_scores = read_single_mutant_dms_scores(context_path)

        tasks.append(
            MutationTask(
                input_path=candidate_path,
                output_path=output_path,
                context=context,
                mutants=read_mutants(candidate_path),
                anchor_mutant=anchor_mutant,
                anchor_dms_score=anchor_dms_score,
                single_mutant_dms_scores=single_mutant_dms_scores,
            )
        )
        if args.max_files is not None and len(tasks) >= args.max_files:
            break
    return tasks


def discover_tasks(args: argparse.Namespace) -> list[MutationTask]:
    if args.task_id == "1":
        return discover_generation_tasks(args)
    elif args.task_id == "2" or args.task_id == "3" or args.task_id == "4":
        return discover_rank_tasks(args)


async def infer_one(client: AsyncOpenAI, task: MutationTask, args: argparse.Namespace) -> bool:
    try:
        response = await create_chat_completion(client, task, args)
        message = response.choices[0].message
        response_text = message.content or ""
        ranked_mutants = parse_prediction(response_text, args)
        
        if ranked_mutants is not None:
            usage = response.usage.model_dump() if response.usage is not None else {}
            write_prediction(task, ranked_mutants, usage, args)
            return True
        print(f"[Response finish reason] {response.choices[0].finish_reason if response.choices else None}")
        return False
    
    except Exception as exc:
        print(f"Error processing {task.input_path}: {exc}")
        return False


def write_prediction(
    task: MutationTask,
    ranked_mutants: tuple[str, ...],
    usage: dict[str, object],
    args: argparse.Namespace,
    output_path: Path | None = None,
) -> None:
    destination = output_path or task.output_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".tmp")
    fieldnames = [
        "assay",
        "mutant",
        "rank",
        "model",
        "response",
        "input_path",
        "usage_json",
    ]
    with temp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        response_for_output = json.dumps({"ranking": list(ranked_mutants)}, ensure_ascii=False)
        
        for rank, mutant in enumerate(ranked_mutants, start=1):
            row = {
                "assay": task.context.assay_id,
                "mutant": mutant,
                "rank": rank,
                "model": args.model if rank == 1 else "",
                "response": response_for_output if rank == 1 else "",
                "input_path": str(task.input_path) if rank == 1 else "",
                "usage_json": json.dumps(usage, ensure_ascii=False) if rank == 1 else "",
            }
            writer.writerow(row)
    temp_path.replace(destination)


async def infer_all(
    client: AsyncOpenAI,
    tasks: list[MutationTask],
    args: argparse.Namespace,
) -> list[bool]:
    semaphore = asyncio.Semaphore(args.num_workers)
    results: list[bool] = []

    async def run_with_semaphore(task: MutationTask) -> bool:
        async with semaphore:
            return await infer_one(client, task, args)

    futures = [run_with_semaphore(task) for task in tasks]
    for future in tqdm(asyncio.as_completed(futures), total=len(futures), desc="Tasks"):
        results.append(await future)
    return results


def validate_args(args: argparse.Namespace) -> None:
    if not API_BASE_URL or not API_KEY:
        raise RuntimeError(
            "Set API_BASE_URL and API_KEY environment variables "
            "in llm/settings.py before running inference."
        )

    supported_task_ids = tuple(DEFAULT_INPUT_PATHS)
    if args.task_id not in supported_task_ids:
        raise ValueError(
            f"Unsupported task id {args.task_id!r}; "
            f"expected one of {', '.join(supported_task_ids)}."
        )

    if not args.model.strip():
        raise ValueError("--model must not be empty")

    if args.num_workers < 1:
        raise ValueError("--num-workers must be positive")

    if args.max_completion_tokens < 1:
        raise ValueError("--max-completion-tokens must be positive")

    if args.max_files is not None and args.max_files < 1:
        raise ValueError("--max-files must be positive")

    if args.retry_runs < 1:
        raise ValueError("--retry-runs must be positive")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API LLM inference for mutation tasks.")
    parser.add_argument("--model", type=str, default="gpt-5.6-sol", help="OpenAI-compatible endpoint model name.")
    parser.add_argument("--task-id", type=str, default="1", help='Task id in ("1", "2", "3", "4") for default input directory selection.')
    parser.add_argument("--input-dir", type=Path, default=None, help="Task 1 Generation: assay.csv path. Task 2-4 Ranking: norm_data directory. Defaults to benchmark path.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Inference results output directory.")
    parser.add_argument("--max-completion-tokens", type=int, default=65536, help="API max completion tokens, needs to be large enough for LLM thinking.")
    parser.add_argument("--num-workers", type=int, default=32, help="API client concurrency.")
    parser.add_argument("--max-files", type=int, default=None, help="Limit files for smoke tests.")
    parser.add_argument("--retry-runs", type=int, default=10, help="Rerun failed responses, default 10 times until no new successes produced.")
    parser.add_argument("--force", action="store_true", help="Re-run files even when output CSV already exists.")
    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = DEFAULT_INPUT_PATHS[args.task_id]

    validate_args(args)
    print(args)
    
    return args


def main() -> int:
    args = parse_args()
    
    client = AsyncOpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
        timeout=API_TIMEOUT,
        max_retries=API_MAX_RETRIES,
    )

    for i in range(args.retry_runs):
        print("======================")
        print(f"Run {i + 1}:")
        tasks = discover_tasks(args)
        print(f"Discovered {len(tasks)} tasks/files needing inference under {args.input_dir}, using model {args.model}.")
        if not tasks:
            return 0
        print("======================")

        results = asyncio.run(infer_all(client, tasks, args))
        
        successes = sum(1 for result in results if result)
        failures = len(results) - successes
        print(f"Completed {successes} files; {failures} failed and will be retried...")
        if failures == 0:
            return 0
        if successes == 0:
            print(f"No successes, early stopping (Run {i + 1}).")
            return 1
    
    print(f"Retries exhausted; unresolved failures remain after {args.retry_runs} runs.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
