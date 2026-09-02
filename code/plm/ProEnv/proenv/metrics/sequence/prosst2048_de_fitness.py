"""
Directed-evolution fitness scoring with the local ProSST-2048 checkpoint.

Score definition:
  fitness_score = sum(logP(mut_aa | WT sequence, WT structure)
                      - logP(wt_aa | WT sequence, WT structure))

This module adapts the standalone ProSST ``directed_evolution_fitness.py``
workflow for ProEnv services.  ProSST needs a wildtype PDB file in addition to
the wildtype sequence and mutation string.
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import types
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
import torch.nn.functional as F


DEFAULT_PROSST_REPO_DIR = Path(os.environ.get("PROSST_REPO_DIR", "/path/to/ProSST"))
DEFAULT_MODEL_DIR = Path(os.environ.get("PROSST_MODEL_DIR", "/path/to/ProSST-2048"))

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
STANDARD_AAS = set(AA_VOCAB)
MUTATION_RE = re.compile(r"([A-Za-z])\s*(\d+)\s*([A-Za-z])")
RES3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


@dataclass(frozen=True)
class Mutation:
    wt: str
    position: int
    mt: str

    def normalized(self) -> str:
        return f"{self.wt}{self.position}{self.mt}"


@dataclass(frozen=True)
class ParsedMutation:
    original: str
    normalized: str
    mutations: list[Mutation]
    valid: bool
    error: str = ""


class _Data:
    """Small subset of torch_geometric.data.Data used by ProSST's GVP encoder."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to(self, device: str | torch.device) -> "_Data":
        for key, value in list(self.__dict__.items()):
            if torch.is_tensor(value):
                setattr(self, key, value.to(device))
        return self


class _Batch(_Data):
    @classmethod
    def from_data_list(cls, data_list: Sequence[_Data]) -> "_Batch":
        node_s, node_v, edge_s, edge_v, edge_index, batch = [], [], [], [], [], []
        offset = 0
        for graph_id, graph in enumerate(data_list):
            n_nodes = graph.node_s.shape[0]
            node_s.append(graph.node_s)
            node_v.append(graph.node_v)
            edge_s.append(graph.edge_s)
            edge_v.append(graph.edge_v)
            edge_index.append(graph.edge_index + offset)
            batch.append(torch.full((n_nodes,), graph_id, dtype=torch.long))
            offset += n_nodes
        return cls(
            node_s=torch.cat(node_s, dim=0),
            node_v=torch.cat(node_v, dim=0),
            edge_index=torch.cat(edge_index, dim=1),
            edge_s=torch.cat(edge_s, dim=0),
            edge_v=torch.cat(edge_v, dim=0),
            batch=torch.cat(batch, dim=0),
        )


def _scatter_add(
    src: torch.Tensor, index: torch.Tensor, dim_size: int | None = None
) -> torch.Tensor:
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    out = torch.zeros((dim_size, *src.shape[1:]), dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


def _scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("local scatter_mean only supports dim=0")
    out = _scatter_add(src, index, dim_size=dim_size)
    counts = torch.bincount(index, minlength=out.shape[0]).clamp(min=1).to(src.device)
    return out / counts.reshape((-1,) + (1,) * (src.dim() - 1))


def _scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
) -> torch.Tensor:
    if dim != 0:
        raise NotImplementedError("local scatter_sum only supports dim=0")
    return _scatter_add(src, index, dim_size=dim_size)


def _scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = 0,
    dim_size: int | None = None,
):
    if dim != 0:
        raise NotImplementedError("local scatter_max only supports dim=0")
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() else 0
    outs = []
    for i in range(dim_size):
        values = src[index == i]
        fallback = torch.full_like(src[0], -torch.inf)
        outs.append(values.max(dim=0).values if values.numel() else fallback)
    return torch.stack(outs, dim=0), None


class _MessagePassing(torch.nn.Module):
    def __init__(self, aggr: str = "mean"):
        super().__init__()
        self.aggr = aggr

    def propagate(self, edge_index: torch.Tensor, s: torch.Tensor, v: torch.Tensor, edge_attr):
        src, dst = edge_index[0], edge_index[1]
        message = self.message(
            s_i=s[dst],
            v_i=v[dst],
            s_j=s[src],
            v_j=v[src],
            edge_attr=(edge_attr[0], edge_attr[1]),
        )
        if self.aggr == "mean":
            return _scatter_mean(message, dst, dim=0, dim_size=s.shape[0])
        if self.aggr == "add":
            return _scatter_sum(message, dst, dim=0, dim_size=s.shape[0])
        raise NotImplementedError(f"unsupported aggregation: {self.aggr}")


def _install_minimal_pyg_modules() -> None:
    """Install tiny torch_geometric/torch_scatter shims for ProSST inference."""
    if "torch_geometric.data" not in sys.modules:
        tg = types.ModuleType("torch_geometric")
        tg_data = types.ModuleType("torch_geometric.data")
        tg_nn = types.ModuleType("torch_geometric.nn")
        tg_data.Data = _Data
        tg_data.Batch = _Batch
        tg_nn.MessagePassing = _MessagePassing
        tg.data = tg_data
        tg.nn = tg_nn
        sys.modules["torch_geometric"] = tg
        sys.modules["torch_geometric.data"] = tg_data
        sys.modules["torch_geometric.nn"] = tg_nn

    if "torch_scatter" not in sys.modules:
        torch_scatter = types.ModuleType("torch_scatter")
        torch_scatter.scatter_add = _scatter_add
        torch_scatter.scatter_mean = _scatter_mean
        torch_scatter.scatter_sum = _scatter_sum
        torch_scatter.scatter_max = _scatter_max
        sys.modules["torch_scatter"] = torch_scatter


def normalize_sequence(sequence: str) -> str:
    """Normalize raw sequence or FASTA text into a plain uppercase AA string."""
    lines = [line.strip() for line in str(sequence).splitlines()]
    sequence = "".join(line for line in lines if line and not line.startswith(">"))
    sequence = re.sub(r"\s+", "", sequence).upper()
    if not sequence:
        raise ValueError("wildtype sequence is empty")

    invalid = sorted(set(sequence) - STANDARD_AAS)
    if invalid:
        raise ValueError(f"wildtype sequence contains non-standard amino acids: {invalid}")
    return sequence


def normalize_mutation_string(
    mutation_string: str,
    wildtype_sequence: str,
) -> tuple[str, list[Mutation]]:
    """Parse, validate, sort, and normalize mutations like A42V:G128D."""
    raw = str(mutation_string).strip()
    if not raw:
        raise ValueError("empty mutant")

    tokens = [token for token in re.split(r"[:;,|\s]+", raw) if token]
    mutations: list[Mutation] = []
    seen_positions: set[int] = set()

    for token in tokens:
        match = MUTATION_RE.fullmatch(token)
        if not match:
            raise ValueError(f"invalid mutation token: {token!r}; expected e.g. A42V")

        wt, position_text, mt = match.groups()
        wt, mt = wt.upper(), mt.upper()
        position = int(position_text)
        if wt not in STANDARD_AAS or mt not in STANDARD_AAS:
            raise ValueError(f"mutation {token!r} contains non-standard amino acids")
        if position < 1 or position > len(wildtype_sequence):
            raise ValueError(
                f"mutation {token!r} position is outside sequence length {len(wildtype_sequence)}"
            )
        if position in seen_positions:
            raise ValueError(f"duplicate mutation position: {position}")

        observed_wt = wildtype_sequence[position - 1]
        if observed_wt != wt:
            raise ValueError(
                f"mutation {token!r} expects wildtype {wt} at position {position}, "
                f"but sequence has {observed_wt}"
            )

        seen_positions.add(position)
        mutations.append(Mutation(wt=wt, position=position, mt=mt))

    mutations.sort(key=lambda item: item.position)
    return ":".join(item.normalized() for item in mutations), mutations


def parse_mutant(mutant: str, wt_sequence: str) -> ParsedMutation:
    original = str(mutant).strip().upper()
    try:
        normalized, mutations = normalize_mutation_string(original, wt_sequence)
    except Exception as exc:
        return ParsedMutation(original, original, [], False, str(exc))
    return ParsedMutation(original, normalized, mutations, True, "")


def _normalize_vector(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(
    distance: torch.Tensor,
    d_min: float = 0.0,
    d_max: float = 20.0,
    count: int = 16,
) -> torch.Tensor:
    centers = torch.linspace(d_min, d_max, count, device=distance.device).view(1, -1)
    sigma = (d_max - d_min) / count
    return torch.exp(-((distance.unsqueeze(-1) - centers) / sigma) ** 2)


def _orientations(ca_coords: torch.Tensor) -> torch.Tensor:
    forward = _normalize_vector(ca_coords[1:] - ca_coords[:-1])
    backward = _normalize_vector(ca_coords[:-1] - ca_coords[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)


def _sidechains(coords: torch.Tensor) -> torch.Tensor:
    n_atom, ca_atom, c_atom = coords[:, 0], coords[:, 1], coords[:, 2]
    c_vec = _normalize_vector(c_atom - ca_atom)
    n_vec = _normalize_vector(n_atom - ca_atom)
    bisector = _normalize_vector(c_vec + n_vec)
    perp = _normalize_vector(torch.cross(c_vec, n_vec, dim=-1))
    return -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)


def _positional_embeddings(
    edge_index: torch.Tensor,
    num_embeddings: int = 16,
) -> torch.Tensor:
    distance = edge_index[0] - edge_index[1]
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = distance.unsqueeze(-1) * frequency
    return torch.cat((torch.cos(angles), torch.sin(angles)), -1)


def _read_pdb_backbone(pdb_file: str | Path) -> tuple[str, torch.Tensor]:
    pdb_path = Path(pdb_file)
    if not pdb_path.exists():
        raise FileNotFoundError(f"PDB file does not exist: {pdb_path}")

    residues: dict[tuple[str, int, str], dict[str, object]] = {}
    residue_order: list[tuple[str, int, str]] = []
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in {"N", "CA", "C", "O"}:
                continue
            altloc = line[16].strip()
            if altloc and altloc != "A":
                continue
            resname = line[17:20].strip().upper()
            if resname not in RES3_TO_1:
                continue
            chain = line[21].strip()
            resid = int(line[22:26])
            icode = line[26].strip()
            key = (chain, resid, icode)
            if key not in residues:
                residues[key] = {"resname": resname, "atoms": {}}
                residue_order.append(key)
            atoms = residues[key]["atoms"]
            assert isinstance(atoms, dict)
            atoms[atom_name] = [
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ]

    sequence, coords = [], []
    for key in residue_order:
        residue = residues[key]
        atoms = residue["atoms"]
        assert isinstance(atoms, dict)
        if all(atom in atoms for atom in ("N", "CA", "C", "O")):
            sequence.append(RES3_TO_1[str(residue["resname"])])
            coords.append([atoms["N"], atoms["CA"], atoms["C"], atoms["O"]])
    if not sequence:
        raise ValueError(f"no complete standard backbone residues found in {pdb_path}")
    return "".join(sequence), torch.tensor(coords, dtype=torch.float32)


def _generate_graph_from_pdb(pdb_file: str | Path, max_distance: float = 10.0) -> _Data:
    aa_seq, coords = _read_pdb_backbone(pdb_file)
    ca_coords = coords[:, 1]
    distances = torch.cdist(ca_coords, ca_coords).cpu().numpy()
    edge_index = torch.tensor(np.array(np.where(distances < max_distance)), dtype=torch.long)
    edge_index = edge_index[:, edge_index[0] != edge_index[1]]

    node_s = torch.zeros(len(ca_coords), 20, dtype=torch.float32)
    node_v = torch.cat([_orientations(ca_coords), _sidechains(coords).unsqueeze(-2)], dim=-2)
    edge_vectors = ca_coords[edge_index[0]] - ca_coords[edge_index[1]]
    edge_s = torch.cat([_rbf(edge_vectors.norm(dim=-1)), _positional_embeddings(edge_index)], dim=-1)
    edge_v = _normalize_vector(edge_vectors).unsqueeze(-2)

    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
    return _Data(
        node_s=node_s,
        node_v=node_v,
        edge_index=edge_index,
        edge_s=edge_s,
        edge_v=edge_v,
        distances=distances,
        aa_seq=aa_seq,
        ca_coords=ca_coords,
    )


def make_structure_input_ids(
    structure_sequence: Sequence[int],
    sequence_length: int,
) -> torch.Tensor:
    """Build ss_input_ids expected by ProSST: [CLS], token+3..., [EOS]."""
    if len(structure_sequence) != sequence_length:
        raise ValueError(
            f"structure sequence length {len(structure_sequence)} does not match "
            f"wildtype sequence length {sequence_length}"
        )
    if any(token < 0 for token in structure_sequence):
        raise ValueError("structure sequence tokens must be non-negative integers")

    shifted = [token + 3 for token in structure_sequence]
    return torch.tensor([[1, *shifted, 2]], dtype=torch.long)


def _resolve_device(device: str | None) -> torch.device:
    if device is None or str(device).lower() in {"", "auto", "none"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = str(device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


class ProSST2048FitnessScorer:
    """Reusable ProSST-2048 scorer for ProEnv directed-evolution service."""

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        *,
        prosst_repo_dir: str | Path = DEFAULT_PROSST_REPO_DIR,
        device: str | None = None,
        batch_size: int = 8,
        structure_vocab_size: int = 2048,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.prosst_repo_dir = Path(prosst_repo_dir)
        self.batch_size = max(1, int(batch_size))
        self.structure_vocab_size = int(structure_vocab_size)
        self.device = _resolve_device(device)
        self._forward_lock = threading.Lock()

        if self.structure_vocab_size != 2048:
            raise ValueError("this scorer supports the downloaded ProSST-2048 model only")
        if not self.model_dir.exists():
            raise FileNotFoundError(f"model directory does not exist: {self.model_dir}")
        if not self.prosst_repo_dir.exists():
            raise FileNotFoundError(f"ProSST repository does not exist: {self.prosst_repo_dir}")

        hf_cache_dir = self.model_dir.parent / "hf_cache"
        os.environ.setdefault("HF_HOME", str(hf_cache_dir))
        os.environ.setdefault("HF_MODULES_CACHE", str(hf_cache_dir / "modules"))

        if str(self.prosst_repo_dir) not in sys.path:
            sys.path.insert(0, str(self.prosst_repo_dir))
        _install_minimal_pyg_modules()

        from prosst.structure.build_subgraph import generate_pos_subgraph
        from prosst.structure.encoder.gvp import AutoGraphEncoder

        self._generate_pos_subgraph = generate_pos_subgraph
        self.structure_model_path = (
            self.prosst_repo_dir / "prosst" / "structure" / "static" / "AE.pt"
        )
        self.structure_cluster_path = (
            self.prosst_repo_dir
            / "prosst"
            / "structure"
            / "static"
            / f"{self.structure_vocab_size}.joblib"
        )
        if not self.structure_model_path.exists():
            raise FileNotFoundError(f"structure encoder weights missing: {self.structure_model_path}")
        if not self.structure_cluster_path.exists():
            raise FileNotFoundError(f"structure cluster file missing: {self.structure_cluster_path}")

        self.structure_encoder = AutoGraphEncoder(
            node_in_dim=(20, 3),
            node_h_dim=(256, 32),
            edge_in_dim=(32, 1),
            edge_h_dim=(64, 2),
            num_layers=6,
        )
        state_dict = torch.load(self.structure_model_path, map_location=self.device)
        self.structure_encoder.load_state_dict(state_dict)
        self.structure_encoder.to(self.device)
        self.structure_encoder.eval()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Trying to unpickle estimator KMeans.*")
            self.cluster_model = joblib.load(self.structure_cluster_path)

        from transformers import AutoModelForMaskedLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.vocab = self.tokenizer.get_vocab()

    @torch.no_grad()
    def _structure_tokens_from_pdb(self, pdb_file: str | Path) -> tuple[str, list[int]]:
        graph = _generate_graph_from_pdb(pdb_file)
        subgraphs: list[_Data] = []
        for anchor_node in range(len(graph.aa_seq)):
            subgraph = self._generate_pos_subgraph(
                graph,
                subgraph_depth=None,
                max_distance=10,
                anchor_nodes=anchor_node,
                verbose=False,
                pure_subgraph=True,
            )[anchor_node]
            subgraphs.append(
                _Data(
                    node_s=subgraph.node_s.to(torch.float32),
                    node_v=subgraph.node_v.to(torch.float32),
                    edge_index=subgraph.edge_index.to(torch.long),
                    edge_s=subgraph.edge_s.to(torch.float32),
                    edge_v=subgraph.edge_v.to(torch.float32),
                )
            )

        embeddings: list[torch.Tensor] = []
        with self._forward_lock:
            for start in range(0, len(subgraphs), self.batch_size):
                batch = _Batch.from_data_list(subgraphs[start : start + self.batch_size]).to(
                    self.device
                )
                batch.node_s = torch.zeros_like(batch.node_s)
                node_embeddings = self.structure_encoder.get_embedding(
                    (batch.node_s, batch.node_v),
                    batch.edge_index,
                    (batch.edge_s, batch.edge_v),
                )
                graph_embeddings = _scatter_mean(
                    node_embeddings,
                    batch.batch,
                    dim=0,
                ).cpu()
                embeddings.append(graph_embeddings)

        normalized = F.normalize(torch.cat(embeddings, dim=0), p=2, dim=1)
        tokens = self.cluster_model.predict(normalized).tolist()
        return graph.aa_seq, [int(token) for token in tokens]

    @torch.no_grad()
    def _log_probs(self, wt_sequence: str, structure_tokens: Sequence[int]) -> torch.Tensor:
        ss_input_ids = make_structure_input_ids(structure_tokens, len(wt_sequence)).to(self.device)
        tokenized = self.tokenizer([wt_sequence], return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        with self._forward_lock:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                ss_input_ids=ss_input_ids,
                labels=input_ids,
            )
            return torch.log_softmax(outputs.logits[:, 1:-1, :], dim=-1).cpu()

    def score_batch(
        self,
        wt_sequence: str,
        mutants: list[str],
        *,
        pdb_file: str | Path,
    ) -> list[dict[str, float | bool | str | None]]:
        raw_wt_sequence = str(wt_sequence).strip()
        try:
            normalized_wt = normalize_sequence(raw_wt_sequence)
        except Exception as exc:
            return [
                {
                    "seq": raw_wt_sequence.upper(),
                    "mutant": str(mutant).strip().upper(),
                    "fitness_score": None,
                    "valid": False,
                    "error": str(exc),
                }
                for mutant in mutants
            ]

        parsed = [parse_mutant(mutant, normalized_wt) for mutant in mutants]
        outputs: list[dict[str, float | bool | str | None]] = [
            {
                "seq": normalized_wt,
                "mutant": pm.normalized,
                "fitness_score": None,
                "valid": False,
                "error": pm.error,
            }
            for pm in parsed
        ]
        valid_indices = [idx for idx, pm in enumerate(parsed) if pm.valid]
        if not valid_indices:
            return outputs

        try:
            pdb_sequence, structure_tokens = self._structure_tokens_from_pdb(pdb_file)
            if pdb_sequence != normalized_wt:
                raise ValueError(
                    "PDB-derived sequence does not match wildtype sequence. "
                    f"PDB has {pdb_sequence!r}; input has {normalized_wt!r}."
                )
            log_probs = self._log_probs(normalized_wt, structure_tokens)
        except Exception as exc:
            error = str(exc)
            for idx in valid_indices:
                outputs[idx] = {
                    "seq": normalized_wt,
                    "mutant": parsed[idx].normalized,
                    "fitness_score": None,
                    "valid": False,
                    "error": error,
                }
            return outputs

        for idx in valid_indices:
            pm = parsed[idx]
            fitness_score = 0.0
            for mutation in pm.mutations:
                pos_idx = mutation.position - 1
                fitness_score += (
                    log_probs[0, pos_idx, self.vocab[mutation.mt]]
                    - log_probs[0, pos_idx, self.vocab[mutation.wt]]
                ).item()

            outputs[idx] = {
                "seq": normalized_wt,
                "mutant": pm.normalized,
                "fitness_score": float(fitness_score),
                "valid": True,
                "error": "",
            }

        return outputs

    def teardown(self) -> None:
        for attr in ("model", "tokenizer", "structure_encoder", "cluster_model"):
            if hasattr(self, attr):
                delattr(self, attr)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
