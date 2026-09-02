"""
Directed-evolution fitness scoring through the local S3F inference script.

The underlying S3F model needs a wildtype PDB file in addition to the
wildtype sequence and mutation string.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_S3F_REPO_DIR = Path(os.environ.get("S3F_REPO_DIR", "/path/to/S3F"))
DEFAULT_S3F_SCRIPT_PATH = DEFAULT_S3F_REPO_DIR / "score_s3f_fitness.py"
DEFAULT_MODEL_ROOT = Path(os.environ.get("S3F_MODEL_ROOT", "/path/to/S3F/model-resources"))
DEFAULT_CHECKPOINT_PATH = DEFAULT_MODEL_ROOT / "checkpoints" / "s3f.pth"
DEFAULT_ESM_MODEL_DIR = DEFAULT_MODEL_ROOT / "esm2"
DEFAULT_SURFACE_CACHE_DIR = DEFAULT_MODEL_ROOT / "generated_surface"


@dataclass(frozen=True)
class ParsedVariant:
    original: str
    normalized: str
    valid: bool
    error: str = ""


class S3FFitnessScorer:
    """Reusable S3F scorer for the s3f_de_fitness service."""

    def __init__(
        self,
        *,
        script_path: str | Path = DEFAULT_S3F_SCRIPT_PATH,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        esm_model_dir: str | Path = DEFAULT_ESM_MODEL_DIR,
        structure_pdb_dir: str | Path | None = None,
        surface_pkl_path: str | Path | None = None,
        surface_pkl_dir: str | Path | None = None,
        surface_cache_dir: str | Path | None = DEFAULT_SURFACE_CACHE_DIR,
        config_path: str | Path | None = None,
        device: str | None = "cuda",
        structure_start: int = 1,
        plddt_threshold: float | None = None,
        mock: bool = False,
    ):
        self.script_path = Path(script_path).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.esm_model_dir = Path(esm_model_dir).expanduser()
        self.structure_pdb_dir = (
            Path(structure_pdb_dir).expanduser() if structure_pdb_dir else None
        )
        self.surface_pkl_path = (
            Path(surface_pkl_path).expanduser() if surface_pkl_path else None
        )
        self.surface_pkl_dir = (
            Path(surface_pkl_dir).expanduser() if surface_pkl_dir else None
        )
        self.surface_cache_dir = (
            Path(surface_cache_dir).expanduser() if surface_cache_dir else None
        )
        self.config_path = Path(config_path).expanduser() if config_path else None
        self.device = device
        self.structure_start = int(structure_start)
        self.plddt_threshold = plddt_threshold
        self.mock = bool(mock)

        self._module: ModuleType | None = None
        self._predictors: dict[tuple[Any, ...], Any] = {}
        self._forward_lock = threading.Lock()

        if not self.mock:
            self._load_module().preload_model(
                method="s3f",
                checkpoint_path=str(self.checkpoint_path),
                esm_model_dir=str(self.esm_model_dir),
                config_path=str(self.config_path) if self.config_path else None,
                device=self.device,
                plddt_threshold=self.plddt_threshold,
            )

    def _load_module(self) -> ModuleType:
        if self._module is not None:
            return self._module
        if not self.script_path.exists():
            raise FileNotFoundError(f"S3F scoring script not found: {self.script_path}")

        script_dir = str(self.script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)

        module_name = "_proenv_s3f_score_s3f_fitness"
        cached = sys.modules.get(module_name)
        if cached is not None:
            self._module = cached
            return cached

        spec = importlib.util.spec_from_file_location(module_name, self.script_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load S3F scoring script: {self.script_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        self._module = module
        return module

    @staticmethod
    def _error_output(error: str) -> dict[str, float | bool | str | None]:
        return {"fitness_score": None, "valid": False, "error": error}

    def _parse_variant(
        self, module: ModuleType, mutant: str, wt_sequence: str
    ) -> ParsedVariant:
        original = str(mutant).strip().upper()
        try:
            parsed = module.parse_mutations(original, wt_sequence)
            normalized = module.format_mutations(parsed)
        except Exception as exc:
            return ParsedVariant(original, original, False, str(exc))
        return ParsedVariant(original, normalized, True, "")

    def _resolve_surface_pkl_path(self, pdb_file: str) -> Path | None:
        if self.surface_pkl_path is not None:
            return self.surface_pkl_path
        if self.surface_pkl_dir is None:
            return None

        candidate = self.surface_pkl_dir / f"{Path(pdb_file).stem}.pkl"
        return candidate if candidate.is_file() else None

    def _resolve_pdb_path(self, pdb_file: str) -> Path:
        supplied = Path(pdb_file).expanduser()
        if self.structure_pdb_dir is None:
            return supplied

        # Pair a precomputed surface graph with the coordinates used to create it.
        candidate = self.structure_pdb_dir / f"{supplied.stem}.pdb"
        return candidate if candidate.is_file() else supplied

    def _predictor_key(self, pdb_file: str) -> tuple[Any, ...]:
        resolved_pdb = self._resolve_pdb_path(pdb_file)
        surface_pkl_path = self._resolve_surface_pkl_path(pdb_file)
        return (
            str(resolved_pdb),
            str(self.checkpoint_path),
            str(self.esm_model_dir),
            str(self.structure_pdb_dir) if self.structure_pdb_dir else "",
            str(surface_pkl_path) if surface_pkl_path else "",
            str(self.surface_pkl_dir) if self.surface_pkl_dir else "",
            str(self.surface_cache_dir) if self.surface_cache_dir else "",
            str(self.config_path) if self.config_path else "",
            str(self.device),
            self.structure_start,
            self.plddt_threshold,
            self.mock,
        )

    def _get_predictor(self, pdb_file: str):
        module = self._load_module()
        key = self._predictor_key(pdb_file)
        resolved_pdb = self._resolve_pdb_path(pdb_file)
        surface_pkl_path = self._resolve_surface_pkl_path(pdb_file)
        predictor = self._predictors.get(key)
        if predictor is None:
            predictor = module.DirectedEvolutionFitnessPredictor(
                method="s3f",
                checkpoint_path=str(self.checkpoint_path),
                pdb_path=str(resolved_pdb),
                surface_pkl_path=(
                    str(surface_pkl_path) if surface_pkl_path else None
                ),
                surface_cache_dir=(
                    str(self.surface_cache_dir) if self.surface_cache_dir else None
                ),
                esm_model_dir=str(self.esm_model_dir),
                config_path=str(self.config_path) if self.config_path else None,
                structure_start=self.structure_start,
                device=self.device,
                plddt_threshold=self.plddt_threshold,
                mock=self.mock,
            )
            self._predictors[key] = predictor
        return predictor

    def _predict_valid_batch(
        self,
        predictor: Any,
        wt_sequence: str,
        variants: list[ParsedVariant],
    ) -> list[dict[str, float | bool | str | None]]:
        mutations = [variant.normalized for variant in variants]
        try:
            with self._forward_lock:
                predictions = predictor.predict_many(wt_sequence, mutations)
        except Exception as batch_exc:
            outputs: list[dict[str, float | bool | str | None]] = []
            for variant in variants:
                try:
                    with self._forward_lock:
                        one_prediction = predictor.predict_many(
                            wt_sequence, [variant.normalized]
                        )[0]
                    outputs.append(
                        {
                            "seq": one_prediction.normalized_wt_sequence,
                            "mutant": one_prediction.normalized_mutation,
                            "fitness_score": float(one_prediction.fitness_score),
                            "valid": True,
                            "error": "",
                        }
                    )
                except Exception as exc:
                    outputs.append(
                        self._error_output(
                            f"{type(exc).__name__}: {exc}"
                            if str(exc)
                            else f"{type(batch_exc).__name__}: {batch_exc}"
                        )
                    )
            return outputs

        return [
            {
                "seq": prediction.normalized_wt_sequence,
                "mutant": prediction.normalized_mutation,
                "fitness_score": float(prediction.fitness_score),
                "valid": True,
                "error": "",
            }
            for prediction in predictions
        ]

    def score_batch(
        self,
        wt_sequence: str,
        mutants: list[str],
        *,
        pdb_file: str,
    ) -> list[dict[str, float | bool | str | None]]:
        module = self._load_module()
        try:
            normalized_wt = module.normalize_sequence(str(wt_sequence))
        except Exception as exc:
            return [self._error_output(str(exc)) for _ in mutants]

        parsed_variants = [
            self._parse_variant(module, mutant, normalized_wt) for mutant in mutants
        ]
        outputs: list[dict[str, float | bool | str | None]] = [
            {} for _ in parsed_variants
        ]
        valid_entries: list[tuple[int, ParsedVariant]] = []
        for idx, variant in enumerate(parsed_variants):
            if not variant.valid:
                outputs[idx] = self._error_output(variant.error)
            else:
                valid_entries.append((idx, variant))

        if not valid_entries:
            return outputs

        try:
            predictor = self._get_predictor(pdb_file)
            scored = self._predict_valid_batch(
                predictor, normalized_wt, [variant for _, variant in valid_entries]
            )
        except Exception as exc:
            scored = [
                self._error_output(f"{type(exc).__name__}: {exc}")
                for _ in valid_entries
            ]

        for (idx, variant), one in zip(valid_entries, scored):
            if one.get("valid"):
                outputs[idx] = one
            else:
                outputs[idx] = {
                    "seq": normalized_wt,
                    "mutant": variant.normalized,
                    "fitness_score": None,
                    "valid": False,
                    "error": one.get("error", "S3F scoring failed"),
                }

        return outputs

    def teardown(self) -> None:
        self._predictors.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
