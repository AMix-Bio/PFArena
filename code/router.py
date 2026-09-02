"""Prompt-template registry for mutation benchmark tasks."""

from pathlib import Path

from prompts.multi_mutant_ranking import multi_mutant_ranking_template
from prompts.multi_mutant_ranking_anchor import multi_mutant_ranking_anchor_template
from prompts.multi_mutant_ranking_mutation import multi_mutant_ranking_mutation_template
from prompts.single_mutant_generation import single_mutant_generation_template


PROMPT_TEMPLATE_FACTORIES = {
    "1": single_mutant_generation_template,
    "2": multi_mutant_ranking_template,
    "3": multi_mutant_ranking_anchor_template,
    "4": multi_mutant_ranking_mutation_template,
}

DATASET_INPUTS = {
    "1": ("T1_single_mutant_generation", "assay.csv"),
    "2": ("T2_measurement_free_multi_mutant_ranking", "norm_data"),
    "3": ("T3_anchor_informed_multi_mutant_ranking", "norm_data"),
    "4": ("T4_mutation_informed_multi_mutant_ranking", "norm_data"),
}


def get_default_input_paths(dataset_root: str | Path) -> dict[str, Path]:
    """Returns the default input path for each supported mutation task."""
    root = Path(dataset_root)
    return {
        task_id: root / task_dir / input_name
        for task_id, (task_dir, input_name) in DATASET_INPUTS.items()
    }


def get_prompt_template(task_id: str) -> str:
    """Returns the prompt template for each supported mutation task."""
    try:
        return PROMPT_TEMPLATE_FACTORIES[task_id]()
    except KeyError as exc:
        supported = ", ".join(sorted(PROMPT_TEMPLATE_FACTORIES))
        raise ValueError(
            f"Unsupported task id {task_id!r}; supported task ids: {supported}"
        ) from exc
