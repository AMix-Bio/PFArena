import argparse
import csv
import inspect
import math
import os
import pickle
import re
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


REPO_DIR = os.path.dirname(os.path.abspath(__file__))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

DEFAULT_MODEL_ROOT = os.environ.get(
    "S3F_MODEL_ROOT",
    os.path.join(REPO_DIR, "model-resources"),
)
DEFAULT_ESM_DIR = os.path.join(DEFAULT_MODEL_ROOT, "esm2")
DEFAULT_CHECKPOINTS = {
    "s2f": os.path.join(DEFAULT_MODEL_ROOT, "checkpoints", "s2f.pth"),
    "s3f": os.path.join(DEFAULT_MODEL_ROOT, "checkpoints", "s3f.pth"),
}

AA_ALPHABET = set("ACDEFGHIKLMNPQRSTVWY")
MUTATION_RE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")
MODEL_ALIASES = {
    "s2f_msa": "s2f_msa",
    "s2f-msa": "s2f_msa",
    "s3f": "s3f",
    "s3f_msa": "s3f_msa",
    "s3f-msa": "s3f_msa",
}


@dataclass(frozen=True)
class Mutation:
    wild_type: str
    position: int
    mutant: str

    @property
    def index(self):
        return self.position - 1

    def __str__(self):
        return "%s%d%s" % (self.wild_type, self.position, self.mutant)


@dataclass(frozen=True)
class EnsembleStats:
    model_mean: float
    model_std: float
    eve_mean: float
    eve_std: float


@dataclass(frozen=True)
class FitnessPrediction:
    normalized_wt_sequence: str
    normalized_mutation: str
    fitness_score: float
    model_score: Optional[float] = None
    eve_score: Optional[float] = None

    def to_dict(self):
        return {
            "normalized_wt_sequence": self.normalized_wt_sequence,
            "normalized_mutation": self.normalized_mutation,
            "fitness_score": self.fitness_score,
        }


def normalize_sequence(sequence: str) -> str:
    """Normalize a wild-type amino-acid sequence or FASTA text."""
    lines = []
    for line in sequence.splitlines():
        line = line.strip()
        if line.startswith(">"):
            continue
        lines.append(line)
    sequence = re.sub(r"\s+", "", "".join(lines)).replace("-", "").upper()
    if not sequence:
        raise ValueError("empty wild-type sequence")
    invalid = sorted(set(sequence) - AA_ALPHABET)
    if invalid:
        raise ValueError("unsupported residue symbols in sequence: %s" % ", ".join(invalid))
    return sequence


def parse_mutations(mutation_string: str, wt_sequence: str) -> Tuple[Mutation, ...]:
    """Parse substitutions like A42G or A42G:D80N and validate against WT."""
    mutation_string = mutation_string.strip().upper()
    if mutation_string in {"", "WT", "WILDTYPE", "WILD_TYPE"}:
        return tuple()

    tokens = [token for token in re.split(r"[:;,\s]+", mutation_string) if token]
    mutations = []
    seen_positions = {}
    for token in tokens:
        if token.startswith("P."):
            token = token[2:]
        match = MUTATION_RE.match(token)
        if not match:
            raise ValueError("invalid substitution `%s`; expected format like A42G" % token)
        wild_type, position, mutant = match.groups()
        position = int(position)
        if position < 1 or position > len(wt_sequence):
            raise ValueError("mutation `%s` is outside sequence length %d" % (token, len(wt_sequence)))
        observed = wt_sequence[position - 1]
        if observed != wild_type:
            raise ValueError(
                "mutation `%s` expects wild-type %s at position %d, but sequence has %s"
                % (token, wild_type, position, observed)
            )
        if position in seen_positions:
            if seen_positions[position] == mutant:
                continue
            raise ValueError("multiple mutants specified for position %d" % position)
        seen_positions[position] = mutant
        mutations.append(Mutation(wild_type, position, mutant))
    return tuple(sorted(mutations, key=lambda item: item.position))


def format_mutations(mutations: Sequence[Mutation]) -> str:
    return ":".join(str(mutation) for mutation in mutations) if mutations else "WT"


def normalize_inputs(wt_sequence: str, mutation_strings: Iterable[str]):
    wt_sequence = normalize_sequence(wt_sequence)
    parsed = [parse_mutations(item, wt_sequence) for item in mutation_strings]
    return wt_sequence, parsed, [format_mutations(item) for item in parsed]


def _normalize_model_name(model_name: str) -> str:
    name = MODEL_ALIASES.get(model_name.lower())
    if name is None:
        raise ValueError("unknown method `%s`; choose s3f, s2f_msa or s3f_msa" % model_name)
    return name


def _base_model_name(model_name: str) -> str:
    return model_name.replace("_msa", "")


def _repo_path(*parts):
    return os.path.join(REPO_DIR, *parts)


def _default_config_path(base_model_name: str) -> str:
    return _repo_path("config", "evaluate", "%s.yaml" % base_model_name)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> float:
    if len(values) < 2:
        raise ValueError("z-score ensemble requires at least two variants or explicit EnsembleStats")
    mean = _mean(values)
    # ProteinGym builds the S2F-MSA/S3F-MSA ensemble with pandas'
    # Series.std(), whose default is the sample standard deviation (ddof=1).
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    std = math.sqrt(variance)
    if std == 0:
        raise ValueError("cannot z-score scores with zero standard deviation")
    return std


def combine_with_eve(model_scores: Sequence[float], eve_scores: Sequence[float], stats: Optional[EnsembleStats] = None):
    if len(model_scores) != len(eve_scores):
        raise ValueError("number of EVE scores must match number of variants")
    model_scores = [float(value) for value in model_scores]
    eve_scores = [float(value) for value in eve_scores]
    if stats is None:
        stats = EnsembleStats(
            model_mean=_mean(model_scores),
            model_std=_std(model_scores),
            eve_mean=_mean(eve_scores),
            eve_std=_std(eve_scores),
        )
    if stats.model_std == 0 or stats.eve_std == 0:
        raise ValueError("ensemble standard deviations must be non-zero")
    return [
        (
            ((model_score - stats.model_mean) / stats.model_std)
            + ((eve_score - stats.eve_mean) / stats.eve_std)
        )
        / 2.0
        for model_score, eve_score in zip(model_scores, eve_scores)
    ]


def _mock_base_score(mutations: Sequence[Mutation], surface_bonus: float = 0.0) -> float:
    """Deterministic mock score for API tests; not a biological model."""
    score = 0.0
    for mutation in mutations:
        score += ((ord(mutation.mutant) - ord(mutation.wild_type)) / 25.0) - 0.03 * mutation.position
    return score + surface_bonus


def _mock_eve_score(mutations: Sequence[Mutation]) -> float:
    score = 0.0
    for mutation in mutations:
        score += ((mutation.position % 7) - 3) * 0.2 + (ord(mutation.mutant) % 5) * 0.05
    return score


class DirectedEvolutionFitnessPredictor:
    """Call released S2F/S3F checkpoints for directed-evolution fitness.

    Real S3F / S3F-MSA need a WT PDB; the surface pkl is generated and cached
    automatically if it is not supplied. S2F-MSA / S3F-MSA additionally need
    one EVE score per variant. Set mock=True to test the interface without
    checkpoint, PDB or EVE inputs.
    """

    def __init__(
        self,
        method: str,
        checkpoint_path: Optional[str] = None,
        pdb_path: Optional[str] = None,
        surface_pkl_path: Optional[str] = None,
        surface_cache_dir: Optional[str] = None,
        esm_model_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        structure_start: int = 1,
        device: Optional[str] = None,
        plddt_threshold: Optional[float] = None,
        mock: bool = False,
    ):
        self.method = _normalize_model_name(method)
        self.base_method = _base_model_name(self.method)
        self.uses_msa = self.method.endswith("_msa")
        self.mock = mock

        self._real_predictor = None
        if not mock:
            checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINTS[self.base_method]
            esm_model_dir = esm_model_dir or DEFAULT_ESM_DIR
            if not checkpoint_path:
                raise ValueError("%s real inference requires checkpoint_path" % self.method)
            if not os.path.exists(os.path.expanduser(checkpoint_path)):
                raise FileNotFoundError("checkpoint not found: %s" % checkpoint_path)
            if not pdb_path:
                raise ValueError("%s real inference requires pdb_path" % self.method)
            self._real_predictor = _RealS3FPredictor(
                model_name=self.method,
                checkpoint_path=checkpoint_path,
                pdb_path=pdb_path,
                surface_pkl_path=surface_pkl_path,
                surface_cache_dir=surface_cache_dir,
                esm_model_dir=esm_model_dir,
                config_path=config_path,
                structure_start=structure_start,
                device=device,
                plddt_threshold=plddt_threshold,
            )

    def predict_many(
        self,
        wt_sequence: str,
        mutation_strings: Iterable[str],
        eve_scores: Optional[Sequence[float]] = None,
        stats: Optional[EnsembleStats] = None,
    ) -> List[FitnessPrediction]:
        wt_sequence, parsed_mutations, normalized_mutations = normalize_inputs(wt_sequence, mutation_strings)

        if self.mock:
            model_scores = [
                _mock_base_score(mutations, surface_bonus=0.25 if self.base_method == "s3f" else 0.0)
                for mutations in parsed_mutations
            ]
        else:
            model_scores = self._real_predictor.score_base_model(wt_sequence, parsed_mutations)

        if self.uses_msa:
            if eve_scores is None:
                if not self.mock:
                    raise ValueError("%s requires one EVE score per variant" % self.method)
                eve_scores = [_mock_eve_score(mutations) for mutations in parsed_mutations]
            fitness_scores = combine_with_eve(model_scores, eve_scores, stats=stats)
            eve_values = [float(value) for value in eve_scores]
        else:
            fitness_scores = model_scores
            eve_values = [None] * len(model_scores)

        return [
            FitnessPrediction(
                normalized_wt_sequence=wt_sequence,
                normalized_mutation=normalized_mutation,
                fitness_score=float(fitness_score),
                model_score=float(model_score),
                eve_score=eve_score,
            )
            for normalized_mutation, fitness_score, model_score, eve_score in zip(
                normalized_mutations, fitness_scores, model_scores, eve_values
            )
        ]


def predict_directed_evolution_fitness(
    wt_sequence: str,
    mutation_strings: Iterable[str],
    method: str,
    checkpoint_path: Optional[str] = None,
    pdb_path: Optional[str] = None,
    surface_pkl_path: Optional[str] = None,
    surface_cache_dir: Optional[str] = None,
    esm_model_dir: Optional[str] = None,
    eve_scores: Optional[Sequence[float]] = None,
    stats: Optional[EnsembleStats] = None,
    mock: bool = False,
    **kwargs,
) -> List[dict]:
    predictor = DirectedEvolutionFitnessPredictor(
        method=method,
        checkpoint_path=checkpoint_path,
        pdb_path=pdb_path,
        surface_pkl_path=surface_pkl_path,
        surface_cache_dir=surface_cache_dir,
        esm_model_dir=esm_model_dir,
        mock=mock,
        **kwargs,
    )
    return [item.to_dict() for item in predictor.predict_many(wt_sequence, mutation_strings, eve_scores=eve_scores, stats=stats)]


def predict_fitness(
    wt_sequence: str,
    mutation_strings: Iterable[str],
    method: str,
    pdb_path: Optional[str] = None,
    eve_scores: Optional[Sequence[float]] = None,
    checkpoint_path: Optional[str] = None,
    surface_pkl_path: Optional[str] = None,
    surface_cache_dir: Optional[str] = None,
    esm_model_dir: Optional[str] = None,
    device: Optional[str] = None,
    structure_start: int = 1,
    plddt_threshold: Optional[float] = None,
    mock: bool = False,
    stats: Optional[EnsembleStats] = None,
) -> List[dict]:
    """Compact public API: sequence + mutation list -> normalized records."""
    return predict_directed_evolution_fitness(
        wt_sequence=wt_sequence,
        mutation_strings=mutation_strings,
        method=method,
        checkpoint_path=checkpoint_path,
        pdb_path=pdb_path,
        surface_pkl_path=surface_pkl_path,
        surface_cache_dir=surface_cache_dir,
        esm_model_dir=esm_model_dir,
        eve_scores=eve_scores,
        stats=stats,
        device=device,
        structure_start=structure_start,
        plddt_threshold=plddt_threshold,
        mock=mock,
    )


def _canonical_path(path: Optional[str]) -> str:
    if path is None:
        return ""
    return os.path.abspath(os.path.expanduser(str(path)))


def preload_model(
    method: str,
    checkpoint_path: Optional[str] = None,
    esm_model_dir: Optional[str] = None,
    config_path: Optional[str] = None,
    device: Optional[str] = None,
    plddt_threshold: Optional[float] = None,
):
    """Load and cache the S2F/S3F task without binding it to a PDB file."""
    method = _normalize_model_name(method)
    base_model_name = _base_model_name(method)
    checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINTS[base_model_name]
    esm_model_dir = esm_model_dir or DEFAULT_ESM_DIR
    if not checkpoint_path:
        raise ValueError("%s real inference requires checkpoint_path" % method)
    if not os.path.exists(os.path.expanduser(checkpoint_path)):
        raise FileNotFoundError("checkpoint not found: %s" % checkpoint_path)

    deps = _load_real_deps()
    torch = deps["torch"]
    torch_device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    return _RealS3FPredictor.load_task_cached(
        deps=deps,
        base_model_name=base_model_name,
        config_path=config_path or _default_config_path(base_model_name),
        checkpoint_path=checkpoint_path,
        esm_model_dir=esm_model_dir,
        device=torch_device,
        plddt_threshold=plddt_threshold,
    )


class _RealS3FPredictor:
    _task_cache = {}
    _task_cache_lock = threading.Lock()

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str,
        pdb_path: str,
        surface_pkl_path: Optional[str] = None,
        surface_cache_dir: Optional[str] = None,
        esm_model_dir: Optional[str] = None,
        config_path: Optional[str] = None,
        structure_start: int = 1,
        device: Optional[str] = None,
        plddt_threshold: Optional[float] = None,
    ):
        deps = _load_real_deps()
        self.torch = deps["torch"]
        self.F = deps["F"]
        self.core = deps["core"]
        self.data = deps["data"]
        self.utils = deps["utils"]
        self.util = deps["util"]
        self.s3f_dataset = deps["s3f_dataset"]

        self.model_name = model_name
        self.base_model_name = _base_model_name(model_name)
        self.pdb_path = os.path.expanduser(pdb_path)
        self.surface_pkl_path = os.path.expanduser(surface_pkl_path) if surface_pkl_path else None
        self.structure_start = structure_start
        self.device = self.torch.device(device or ("cuda:0" if self.torch.cuda.is_available() else "cpu"))
        if self.base_model_name == "s3f" and self.surface_pkl_path is None:
            self.surface_pkl_path = self._generate_surface_from_pdb(surface_cache_dir, checkpoint_path)

        self.task = self.load_task_cached(
            deps=deps,
            base_model_name=self.base_model_name,
            config_path=config_path or _default_config_path(self.base_model_name),
            checkpoint_path=checkpoint_path,
            esm_model_dir=esm_model_dir,
            device=self.device,
            plddt_threshold=plddt_threshold,
        )
        self.wild_type_graph, self.surface_graph = self._load_graphs()

    def _generate_surface_from_pdb(self, surface_cache_dir, checkpoint_path):
        cache_dir = surface_cache_dir or os.environ.get("S3F_SURFACE_CACHE_DIR")
        if cache_dir is None:
            checkpoint_dir = os.path.dirname(os.path.expanduser(checkpoint_path))
            cache_dir = os.path.join(os.path.dirname(checkpoint_dir), "generated_surface")
        cache_dir = os.path.expanduser(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        output_path = os.path.join(cache_dir, "%s.pkl" % os.path.splitext(os.path.basename(self.pdb_path))[0])
        if not os.path.exists(output_path):
            _write_surface_from_pdb(self.pdb_path, output_path, self.torch, self.F, self.device)
        return output_path

    @classmethod
    def _task_cache_key(
        cls,
        base_model_name,
        config_path,
        checkpoint_path,
        esm_model_dir,
        device,
        plddt_threshold,
    ):
        return (
            base_model_name,
            _canonical_path(config_path),
            _canonical_path(checkpoint_path),
            _canonical_path(esm_model_dir),
            str(device),
            None if plddt_threshold is None else float(plddt_threshold),
        )

    @classmethod
    def load_task_cached(
        cls,
        *,
        deps,
        base_model_name,
        config_path,
        checkpoint_path,
        esm_model_dir=None,
        device=None,
        plddt_threshold=None,
    ):
        key = cls._task_cache_key(
            base_model_name,
            config_path,
            checkpoint_path,
            esm_model_dir,
            device,
            plddt_threshold,
        )
        with cls._task_cache_lock:
            task = cls._task_cache.get(key)
            if task is None:
                task = cls._load_task_from_deps(
                    deps=deps,
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                    esm_model_dir=esm_model_dir,
                    device=device,
                    plddt_threshold=plddt_threshold,
                )
                cls._task_cache[key] = task
            return task

    @staticmethod
    def _load_task_from_deps(
        *,
        deps,
        config_path,
        checkpoint_path,
        esm_model_dir=None,
        device=None,
        plddt_threshold=None,
    ):
        torch = deps["torch"]
        core = deps["core"]
        util = deps["util"]
        context = {"datadir": "null", "structdir": "null", "surfdir": "null", "ckpt": checkpoint_path}
        cfg = util.load_config(config_path, context=context)
        if esm_model_dir is not None:
            cfg.task.model.sequence_model.path = os.path.expanduser(esm_model_dir)
        if plddt_threshold is not None:
            cfg.task.plddt_threshold = plddt_threshold

        loaded_task = core.Configurable.load_config_dict(cfg.task)
        loaded_task.preprocess(None, None, None)
        checkpoint = torch.load(os.path.expanduser(checkpoint_path), map_location=torch.device("cpu"))
        state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
        loaded_task.load_state_dict(state_dict)
        if device is not None:
            loaded_task.to(device)
        loaded_task.eval()
        return loaded_task

    def _load_graphs(self):
        wild_type, _ = self.s3f_dataset.bio_load_pdb(self.pdb_path)
        ca_index = wild_type.atom_name == wild_type.atom_name2id["CA"]
        wild_type = wild_type.subgraph(ca_index)
        wild_type.view = "residue"
        start = self.structure_start - 1
        end = start + int(wild_type.num_residue)
        with wild_type.graph():
            wild_type.start = self.torch.as_tensor(start)
            wild_type.end = self.torch.as_tensor(end)

        if self.surface_pkl_path:
            with open(self.surface_pkl_path, "rb") as fin:
                surface_dict = pickle.load(fin)
            surface_graph = self.s3f_dataset.load_surface(surface_dict)
            res2surf = self.torch.as_tensor(surface_dict["res2surf"])
            if res2surf.shape[0] != wild_type.num_residue:
                raise ValueError(
                    "surface res2surf length %d does not match CA residues %d"
                    % (res2surf.shape[0], wild_type.num_residue)
                )
            with wild_type.residue():
                wild_type.res2surf = res2surf
        else:
            surface_graph = None
        return wild_type, surface_graph

    def _masked_sequence_graph(self, wt_sequence: str, mutations: Sequence[Mutation]):
        sites = [mutation.index for mutation in mutations]
        sequence_graph = self.data.Protein.from_sequence(wt_sequence, atom_feature=None, bond_feature=None)
        sequence_graph.view = "residue"

        structure_start = int(self.wild_type_graph.start)
        structure_end = int(self.wild_type_graph.end)
        if structure_end > len(wt_sequence):
            raise ValueError(
                "structure range [%d, %d) exceeds sequence length %d"
                % (structure_start, structure_end, len(wt_sequence))
            )

        if sites:
            start, end = _get_optimal_window(sites[0], len(wt_sequence), self.task.model.sequence_model.max_input_length)
            if start < structure_start or end > structure_end:
                if min(sites) >= structure_start and max(sites) < structure_end:
                    start, end = structure_start, structure_end
                else:
                    raise ValueError(
                        "mutation sites %s are not covered by the supplied structure range [%d, %d]"
                        % (sites, structure_start, structure_end - 1)
                    )
        else:
            start, end = structure_start, min(structure_end, len(wt_sequence))

        if end - start > self.task.model.sequence_model.max_input_length:
            raise ValueError("selected structure/sequence window is longer than the ESM input window")

        residue_mask = self.torch.zeros((sequence_graph.num_residue,), dtype=self.torch.bool)
        residue_mask[start:end] = 1
        sequence_graph = sequence_graph.subresidue(residue_mask)
        with sequence_graph.graph():
            sequence_graph.start = self.torch.as_tensor(start)
            sequence_graph.end = self.torch.as_tensor(end)

        if sites:
            node_index = self.torch.as_tensor([site - start for site in sites], dtype=self.torch.long)
            mask_id = self.task.model.sequence_model.alphabet.get_idx("<mask>")
            with sequence_graph.residue():
                sequence_graph.residue_feature[node_index] = 0
                sequence_graph.residue_type[node_index] = mask_id
        return sequence_graph, start

    def _log_probs_for_sites(self, wt_sequence: str, mutations: Sequence[Mutation]):
        sequence_graph, offset = self._masked_sequence_graph(wt_sequence, mutations)
        mutant_dataset = self.s3f_dataset.MutantDataset([sequence_graph], self.wild_type_graph, surf_graph=self.surface_graph)
        dataloader = self.data.DataLoader(mutant_dataset, batch_size=1, shuffle=False, num_workers=0)
        batch = next(iter(dataloader))
        if self.device.type == "cuda":
            batch = self.utils.cuda(batch, device=self.device)
        with self.torch.no_grad():
            logits, sizes = self.task.inference(batch)
        if len(sizes) != 1:
            raise RuntimeError("expected one scored sequence, got %d" % len(sizes))
        return self.F.log_softmax(logits[: int(sizes[0])], dim=-1), offset

    def score_base_model(self, wt_sequence: str, parsed_mutations: Sequence[Sequence[Mutation]]) -> List[float]:
        log_prob_cache = {}
        scores = []
        for mutations in parsed_mutations:
            site_key = tuple(mutation.index for mutation in mutations)
            if site_key not in log_prob_cache:
                log_prob_cache[site_key] = self._log_probs_for_sites(wt_sequence, mutations)
            log_probs, offset = log_prob_cache[site_key]
            scores.append(self._score_from_log_probs(log_probs, offset, mutations))
        return scores

    def _score_from_log_probs(self, log_probs, offset: int, mutations: Sequence[Mutation]) -> float:
        if not mutations:
            return 0.0
        device = log_probs.device
        node_index = self.torch.as_tensor([mutation.index - offset for mutation in mutations], dtype=self.torch.long, device=device)
        mutant_target = self.torch.as_tensor(
            [self.data.Protein.residue_symbol2id[mutation.mutant] for mutation in mutations],
            dtype=self.torch.long,
            device=device,
        )
        wild_type_target = self.torch.as_tensor(
            [self.data.Protein.residue_symbol2id[mutation.wild_type] for mutation in mutations],
            dtype=self.torch.long,
            device=device,
        )
        score = log_probs[node_index, mutant_target] - log_probs[node_index, wild_type_target]
        return float(score.sum().detach().cpu())


def _get_optimal_window(mutation_position: int, sequence_length: int, model_window: int):
    half_model_window = model_window // 2
    if sequence_length <= model_window:
        return 0, sequence_length
    if mutation_position < half_model_window:
        return 0, model_window
    if mutation_position >= sequence_length - half_model_window:
        return sequence_length - model_window, sequence_length
    return max(0, mutation_position - half_model_window), min(sequence_length, mutation_position + half_model_window)


def _write_surface_from_pdb(pdb_path, output_path, torch, F, device):
    import numpy as np

    from s3f import protein, surface

    if surface.robust_laplacian is None:
        raise ImportError("PDB-to-surface generation requires robust_laplacian in the s3f conda env")

    with open(pdb_path, "r") as fin:
        data_dict = protein.from_pdb_string(fin.read()).to_dict()

    num_residue = len(data_dict["aatype"])
    atom_position = torch.as_tensor(data_dict["atom_positions"][:, :3], dtype=torch.float32, device=device)
    atom_type = torch.tensor([3, 0, 0], dtype=torch.long, device=device)[None, :].repeat(num_residue, 1)
    atom_position = atom_position.flatten(0, 1)
    atom_type = F.one_hot(atom_type.flatten(0, 1), num_classes=6).float()
    batch = torch.zeros((atom_position.shape[0],), dtype=torch.long, device=device)

    surf_points, surf_normals, _ = surface.atoms_to_points_normals(atom_position, batch, atomtypes=atom_type)
    res2surf, _ = surface.knn_atoms(atom_position, surf_points, k=20)
    res2surf = res2surf.view(num_residue, 3, -1)

    batch_surf = torch.zeros((surf_points.shape[0],), dtype=torch.long, device=surf_points.device)
    surf_curvatures = surface.compute_curvatures(
        surf_points,
        surf_normals,
        batch=batch_surf,
        curvature_scales=[1.0, 2.0, 3.0, 5.0, 10.0],
    )

    surf_points_np = surf_points.detach().cpu().numpy()
    eigs_ratio = 0.01 if surf_points_np.shape[0] > 20000 else 0.06
    surf_eig_vals, surf_eig_vecs, _ = surface.compute_eigens(
        surf_points_np.shape[0], surf_points_np, min_n_eigs=50, eigs_ratio=eigs_ratio
    )
    surf_hks = surface.compute_HKS(surf_eig_vecs, surf_eig_vals, num_t=32, t_min=0.1, t_max=1000, scale=1000)

    surface_dict = {
        "surf_points": surf_points_np.astype(np.float32),
        "surf_normals": surf_normals.detach().cpu().numpy().astype(np.float32),
        "surf_hks": surf_hks.astype(np.float32),
        "surf_curvatures": surf_curvatures.detach().cpu().numpy().astype(np.float32),
        "res2surf": res2surf.detach().cpu().numpy(),
    }
    with open(output_path, "wb") as fout:
        pickle.dump(surface_dict, fout, protocol=pickle.HIGHEST_PROTOCOL)


def _load_real_deps():
    _ensure_runtime_cache_dirs()
    try:
        import torch
        from torch.nn import functional as F
        # TorchDrug patches torch.utils.data.Dataset at import time. Importing
        # this first avoids a torchvision metaclass conflict later via ESM.
        from torchvision.datasets.utils import download_url as _download_url  # noqa: F401
        from torchdrug import core, data, utils

        import util
        from s3f import dataset as s3f_dataset
        from s3f import gvp, model, task  # noqa: F401 - register configurable classes
        _patch_esm_loader_compat()
    except ImportError as exc:
        raise ImportError(
            "real S2F/S3F inference needs torch, torchdrug, torch_geometric/PyG, "
            "biopython, easydict and surface dependencies installed in the conda env"
        ) from exc
    return {
        "torch": torch,
        "F": F,
        "core": core,
        "data": data,
        "utils": utils,
        "util": util,
        "s3f_dataset": s3f_dataset,
    }


def _ensure_runtime_cache_dirs():
    cache_root = Path(os.environ.get("S3F_CACHE_DIR", Path(tempfile.gettempdir()) / "s3f_runtime_cache"))
    torch_ext_dir = cache_root / "torch_extensions"
    mpl_dir = cache_root / "mplconfig"
    torch_ext_dir.mkdir(parents=True, exist_ok=True)
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TORCH_EXTENSIONS_DIR", str(torch_ext_dir))
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def _patch_esm_loader_compat():
    import esm.pretrained

    loader = esm.pretrained.load_model_and_alphabet_core
    if getattr(loader, "_s3f_compat", False):
        return
    if len(inspect.signature(loader).parameters) != 2:
        return

    def load_model_and_alphabet_core_compat(model_name_or_data, model_data=None, regression_data=None):
        if isinstance(model_name_or_data, str) and isinstance(model_data, dict):
            return loader(model_data, regression_data)
        if model_data is not None:
            return loader(model_name_or_data, model_data)
        return loader(model_name_or_data)

    load_model_and_alphabet_core_compat._s3f_compat = True
    esm.pretrained.load_model_and_alphabet_core = load_model_and_alphabet_core_compat


def _read_sequence(args):
    if args.sequence_file:
        with open(os.path.expanduser(args.sequence_file), "r") as fin:
            return fin.read()
    return args.sequence


def _read_variants(args):
    if args.variants_csv:
        with open(os.path.expanduser(args.variants_csv), "r") as fin:
            reader = csv.DictReader(fin)
            rows = list(reader)
            fieldnames = reader.fieldnames or []
        if "mutation" not in fieldnames:
            raise ValueError("variants CSV must contain a `mutation` column")
        mutations = [row["mutation"] for row in rows]
        eve_scores = None
        if "eve_score" in fieldnames and any(row["eve_score"] for row in rows):
            eve_scores = [float(row["eve_score"]) for row in rows]
        return mutations, eve_scores

    if not args.mutation:
        raise ValueError("provide --mutation or --variants-csv")
    if args.eve_score is not None and len(args.eve_score) != len(args.mutation):
        raise ValueError("number of --eve-score values must match number of --mutation values")
    return args.mutation, args.eve_score


def _read_stats(args):
    values = [args.model_mean, args.model_std, args.eve_mean, args.eve_std]
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError("explicit ensemble stats require all of --model-mean/--model-std/--eve-mean/--eve-std")
    if args.model_std == 0 or args.eve_std == 0:
        raise ValueError("explicit ensemble standard deviations must be non-zero")
    return EnsembleStats(
        model_mean=args.model_mean,
        model_std=args.model_std,
        eve_mean=args.eve_mean,
        eve_std=args.eve_std,
    )


def _write_predictions(predictions, output_path=None, debug_scores=False):
    fieldnames = ["normalized_wt_sequence", "normalized_mutation", "fitness_score"]
    if debug_scores:
        fieldnames += ["model_score", "eve_score"]
    fout = open(os.path.expanduser(output_path), "w", newline="") if output_path else sys.stdout
    try:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for prediction in predictions:
            row = {
                "normalized_wt_sequence": prediction.normalized_wt_sequence,
                "normalized_mutation": prediction.normalized_mutation,
                "fitness_score": "%.8g" % prediction.fitness_score,
            }
            if debug_scores:
                row["model_score"] = "" if prediction.model_score is None else "%.8g" % prediction.model_score
                row["eve_score"] = "" if prediction.eve_score is None else "%.8g" % prediction.eve_score
            writer.writerow(row)
    finally:
        if output_path:
            fout.close()


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Single-file S2F-MSA / S3F / S3F-MSA fitness scorer for released S3F checkpoints."
    )
    parser.add_argument(
        "--method",
        "--model",
        dest="method",
        required=True,
        choices=["s3f", "s2f_msa", "s2f-msa", "s3f_msa", "s3f-msa"],
    )
    parser.add_argument("--sequence", help="Wild-type amino-acid sequence or FASTA text")
    parser.add_argument("--sequence-file", help="File containing the wild-type sequence or FASTA")
    parser.add_argument("--mutation", action="append", help="Mutation string such as A42G or A42G:D80N")
    parser.add_argument("--variants-csv", help="CSV with a `mutation` column and optional `eve_score` column")
    parser.add_argument("--eve-score", type=float, action="append", help="EVE score for one MSA-ensemble variant")
    parser.add_argument("--pdb", help="Wild-type PDB path. Required for real inference.")
    parser.add_argument("--surface", help="Precomputed surface pkl. If omitted for S3F/S3F-MSA, it is generated from --pdb.")
    parser.add_argument("--surface-cache-dir", help="Directory for generated surface pkl files")
    parser.add_argument("--ckpt", help="Released checkpoint path. Defaults to models/S3F/checkpoints/{s2f,s3f}.pth")
    parser.add_argument("--esm-dir", help="Directory containing esm2_t33_650M_UR50D.pt")
    parser.add_argument("--device", help="Torch device, e.g. cuda:0")
    parser.add_argument("--structure-start", type=int, default=1, help="1-based sequence position of the first PDB residue")
    parser.add_argument("--plddt-threshold", type=float, help="Override config pLDDT fallback threshold")
    parser.add_argument("--model-mean", type=float, help="Explicit mean for base model z-score")
    parser.add_argument("--model-std", type=float, help="Explicit std for base model z-score")
    parser.add_argument("--eve-mean", type=float, help="Explicit mean for EVE z-score")
    parser.add_argument("--eve-std", type=float, help="Explicit std for EVE z-score")
    parser.add_argument("--mock", action="store_true", help="Run deterministic mock scoring without real model files")
    parser.add_argument("--debug-scores", action="store_true", help="Also output model_score and eve_score columns")
    parser.add_argument("-o", "--output", help="Output CSV path. Defaults to stdout.")
    args = parser.parse_args()
    if not args.sequence and not args.sequence_file:
        parser.error("provide --sequence or --sequence-file")
    if not args.mock and not args.pdb:
        parser.error("real inference requires --pdb; use --mock for interface tests")
    return args


def main():
    args = _parse_args()
    sequence = _read_sequence(args)
    mutations, eve_scores = _read_variants(args)
    if _normalize_model_name(args.method).endswith("_msa") and eve_scores is None and not args.mock:
        raise ValueError("%s requires EVE scores via --eve-score or a CSV eve_score column" % args.method)
    predictor = DirectedEvolutionFitnessPredictor(
        method=args.method,
        checkpoint_path=args.ckpt,
        pdb_path=args.pdb,
        surface_pkl_path=args.surface,
        surface_cache_dir=args.surface_cache_dir,
        esm_model_dir=args.esm_dir,
        structure_start=args.structure_start,
        device=args.device,
        plddt_threshold=args.plddt_threshold,
        mock=args.mock,
    )
    predictions = predictor.predict_many(
        sequence,
        mutations,
        eve_scores=eve_scores,
        stats=_read_stats(args),
    )
    _write_predictions(predictions, args.output, debug_scores=args.debug_scores)


if __name__ == "__main__":
    main()
