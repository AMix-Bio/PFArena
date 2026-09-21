"""
VenusREM scoring adapter using ProSST logits.

The implementation follows VenusREM's upstream ``compute_fitness.py`` scoring
rule through a reusable scoring interface:

    fitness_score = sum(logP(mut_aa) - logP(wt_aa))

VenusREM additionally needs a structure-token sequence, and optionally residue
or structure alignment files. Those resources are resolved from a configured
VenusREM-style data directory or explicit item fields.
"""

from __future__ import annotations

import os
import random
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer


AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")
ALLOWED_LOGIT_MODES = {
    "aa_seq_aln",
    "struc_seq_aln",
    "aa_seq_aln+struc_seq_aln",
    "struc_seq_aln+aa_seq_aln",
}


@dataclass
class ParsedMutation:
    mutant: str
    wt_list: list[str]
    pos_list: list[int]
    mt_list: list[str]
    valid: bool
    error: str = ""


@dataclass(frozen=True)
class VenusREMProteinContext:
    protein_name: str
    residue_fasta: str
    structure_fasta: str
    aa_seq_aln_file: str | None = None
    struc_seq_aln_file: str | None = None


def parse_mutant(mutant: str, wt_sequence: str) -> ParsedMutation:
    mutant = str(mutant).strip().upper()
    wt_sequence = str(wt_sequence).strip().upper()

    if not mutant:
        return ParsedMutation(mutant, [], [], [], False, "empty mutant")
    if not wt_sequence:
        return ParsedMutation(mutant, [], [], [], False, "empty wt sequence")

    wt_list: list[str] = []
    pos_list: list[int] = []
    mt_list: list[str] = []
    for sub in mutant.split(":"):
        match = MUT_RE.fullmatch(sub)
        if not match:
            return ParsedMutation(mutant, [], [], [], False, f"cannot parse sub-mutation: {sub}")
        wt_aa, pos_str, mt_aa = match.groups()
        pos = int(pos_str)
        if wt_aa not in AA_VOCAB or mt_aa not in AA_VOCAB:
            return ParsedMutation(mutant, [], [], [], False, f"non-standard amino acid in {sub}")
        if pos < 1 or pos > len(wt_sequence):
            return ParsedMutation(
                mutant,
                [],
                [],
                [],
                False,
                f"position {pos} out of range (1..{len(wt_sequence)})",
            )
        if wt_sequence[pos - 1] != wt_aa:
            return ParsedMutation(
                mutant,
                [],
                [],
                [],
                False,
                f"wt mismatch at pos {pos}: expected {wt_sequence[pos - 1]}, got {wt_aa}",
            )
        wt_list.append(wt_aa)
        pos_list.append(pos)
        mt_list.append(mt_aa)

    if len(set(pos_list)) != len(pos_list):
        return ParsedMutation(mutant, [], [], [], False, "duplicate positions")

    return ParsedMutation(mutant, wt_list, pos_list, mt_list, True, "")


def _read_fasta_sequence(path: str) -> str:
    seq_parts: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            seq_parts.append(line)
    return "".join(seq_parts).strip()


def _read_multi_fasta(path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    header = ""
    seq_parts: list[str] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    sequences[header] = (
                        "".join(seq_parts)
                        .upper()
                        .replace("-", "<pad>")
                        .replace(".", "<pad>")
                    )
                header = line
                seq_parts = []
            else:
                seq_parts.append(line)
    if header:
        sequences[header] = (
            "".join(seq_parts)
            .upper()
            .replace("-", "<pad>")
            .replace(".", "<pad>")
        )
    return sequences


def _tokenize_structure_sequence(structure_sequence: list[int]) -> torch.Tensor:
    shifted = [idx + 3 for idx in structure_sequence]
    return torch.tensor([[1, *shifted, 2]], dtype=torch.long)


def _model_suffix(model_name_or_path: str, structure_vocab_size: int | str) -> str:
    if structure_vocab_size:
        return str(structure_vocab_size)
    return str(model_name_or_path).rstrip("/").split("-")[-1]


class VenusREMFitnessScorer:
    """VenusREM scorer reused by the ``venusrem`` service."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cuda",
        base_dir: str | None = None,
        aa_seq_dir: str | None = None,
        struc_seq_dir: str | None = None,
        aa_seq_aln_dir: str | None = None,
        struc_seq_aln_dir: str | None = None,
        logit_mode: str | None = "aa_seq_aln",
        alpha: float = 0.8,
        structure_vocab_size: int | str = 2048,
        sample_ratio: float = 1.0,
        sample_times: int = 1,
        cache_dir: str | None = None,
        local_files_only: bool = False,
        trust_remote_code: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.base_dir = base_dir
        self.aa_seq_dir = aa_seq_dir or (
            os.path.join(base_dir, "aa_seq") if base_dir else None
        )
        self.struc_seq_dir = struc_seq_dir or (
            os.path.join(base_dir, "struc_seq") if base_dir else None
        )
        self.aa_seq_aln_dir = aa_seq_aln_dir or (
            os.path.join(base_dir, "aa_seq_aln_a2m") if base_dir else None
        )
        self.struc_seq_aln_dir = struc_seq_aln_dir or (
            os.path.join(base_dir, "struc_seq_aln_foldseek") if base_dir else None
        )
        self.logit_mode = (
            None if logit_mode in (None, "", "none", "None") else str(logit_mode)
        )
        if self.logit_mode is not None and self.logit_mode not in ALLOWED_LOGIT_MODES:
            allowed = ", ".join(sorted(ALLOWED_LOGIT_MODES))
            raise ValueError(
                f"unsupported VenusREM logit_mode {self.logit_mode!r}; allowed: {allowed}"
            )
        self.alpha = float(alpha)
        self.structure_vocab_size = str(structure_vocab_size)
        self.sample_ratio = float(sample_ratio)
        if self.sample_ratio <= 0:
            raise ValueError("venusrem sample_ratio must be positive")
        self.sample_times = max(1, int(sample_times))
        self._sequence_index: dict[str, VenusREMProteinContext] | None = None
        self._forward_lock = threading.Lock()

        if str(device).startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(str(device))
        else:
            self.device = torch.device("cpu")

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": bool(trust_remote_code),
            "local_files_only": bool(local_files_only),
        }
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name_or_path, **load_kwargs
        ).to(self.device)
        self.model.eval()
        self.vocab = self.tokenizer.get_vocab()
        missing = [aa for aa in AA_VOCAB if aa not in self.vocab]
        if missing:
            raise ValueError(f"Tokenizer vocab is missing amino-acid tokens: {missing}")

    def _build_sequence_index(self) -> dict[str, VenusREMProteinContext]:
        if not self.aa_seq_dir:
            raise ValueError(
                "venusrem aa_seq_dir/base_dir is required when protein_name is not provided"
            )
        aa_dir = Path(self.aa_seq_dir)
        if not aa_dir.is_dir():
            raise FileNotFoundError(f"VenusREM aa_seq_dir does not exist: {aa_dir}")

        index: dict[str, VenusREMProteinContext] = {}
        for fasta in sorted(aa_dir.glob("*.fasta")):
            seq = _read_fasta_sequence(str(fasta)).upper()
            if not seq or seq in index:
                continue
            context = self._resolve_context_for_name(fasta.stem, residue_fasta=str(fasta))
            index[seq] = context
        return index

    def _resolve_context_for_name(
        self,
        protein_name: str,
        *,
        residue_fasta: str | None = None,
        structure_fasta: str | None = None,
        aa_seq_aln_file: str | None = None,
        struc_seq_aln_file: str | None = None,
    ) -> VenusREMProteinContext:
        if residue_fasta is None:
            if not self.aa_seq_dir:
                raise ValueError("venusrem aa_seq_dir/base_dir is required")
            residue_fasta = os.path.join(self.aa_seq_dir, f"{protein_name}.fasta")

        if structure_fasta is None:
            if not self.struc_seq_dir:
                raise ValueError("venusrem struc_seq_dir/base_dir is required")
            suffix = _model_suffix(self.model_name_or_path, self.structure_vocab_size)
            candidates = [
                os.path.join(self.struc_seq_dir, suffix, f"{protein_name}.fasta"),
                os.path.join(self.struc_seq_dir, f"{protein_name}.fasta"),
            ]
            structure_fasta = next(
                (path for path in candidates if os.path.exists(path)),
                candidates[0],
            )

        if (
            aa_seq_aln_file is None
            and self.alpha != 0.0
            and self.logit_mode
            and "aa_seq_aln" in self.logit_mode
        ):
            if not self.aa_seq_aln_dir:
                raise ValueError("venusrem aa_seq_aln_dir/base_dir is required for aa_seq_aln mode")
            candidates = [
                os.path.join(self.aa_seq_aln_dir, f"{protein_name}.a2m"),
                os.path.join(self.aa_seq_aln_dir, f"{protein_name}.a3m"),
                os.path.join(self.aa_seq_aln_dir, f"{protein_name}.fasta"),
            ]
            aa_seq_aln_file = next(
                (path for path in candidates if os.path.exists(path)),
                candidates[0],
            )

        if (
            struc_seq_aln_file is None
            and self.alpha != 0.0
            and self.logit_mode
            and "struc_seq_aln" in self.logit_mode
        ):
            if not self.struc_seq_aln_dir:
                raise ValueError(
                    "venusrem struc_seq_aln_dir/base_dir is required for struc_seq_aln mode"
                )
            struc_seq_aln_file = os.path.join(self.struc_seq_aln_dir, f"{protein_name}.fasta")

        return VenusREMProteinContext(
            protein_name=protein_name,
            residue_fasta=residue_fasta,
            structure_fasta=structure_fasta,
            aa_seq_aln_file=aa_seq_aln_file,
            struc_seq_aln_file=struc_seq_aln_file,
        )

    def _resolve_context(
        self, wt_sequence: str, context: dict[str, Any] | None
    ) -> VenusREMProteinContext:
        context = context or {}
        protein_name = str(
            context.get("protein_name")
            or context.get("name")
            or context.get("protein")
            or ""
        ).strip()
        residue_fasta = context.get("residue_fasta") or context.get("aa_seq_fasta")
        structure_fasta = context.get("structure_fasta") or context.get("struc_seq_fasta")
        aa_seq_aln_file = context.get("aa_seq_aln_file")
        struc_seq_aln_file = context.get("struc_seq_aln_file")

        if protein_name:
            return self._resolve_context_for_name(
                protein_name,
                residue_fasta=str(residue_fasta) if residue_fasta else None,
                structure_fasta=str(structure_fasta) if structure_fasta else None,
                aa_seq_aln_file=str(aa_seq_aln_file) if aa_seq_aln_file else None,
                struc_seq_aln_file=str(struc_seq_aln_file) if struc_seq_aln_file else None,
            )

        if residue_fasta and structure_fasta:
            return VenusREMProteinContext(
                protein_name=Path(str(residue_fasta)).stem,
                residue_fasta=str(residue_fasta),
                structure_fasta=str(structure_fasta),
                aa_seq_aln_file=str(aa_seq_aln_file) if aa_seq_aln_file else None,
                struc_seq_aln_file=str(struc_seq_aln_file) if struc_seq_aln_file else None,
            )

        if self._sequence_index is None:
            self._sequence_index = self._build_sequence_index()
        found = self._sequence_index.get(wt_sequence)
        if found is None:
            raise ValueError(
                "cannot resolve VenusREM protein context from WT sequence; "
                "provide item['protein_name'] or configure aa_seq_dir/base_dir "
                "with a matching FASTA"
            )
        return found

    @staticmethod
    def _check_file(path: str, label: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} does not exist: {path}")

    def _count_matrix_from_residue_alignment(
        self, alignment_dict: dict[str, str]
    ) -> tuple[torch.Tensor, int, int]:
        alignment_seqs = list(alignment_dict.values())
        if not alignment_seqs:
            raise ValueError("empty residue alignment")
        try:
            aln_start, aln_end = list(alignment_dict.keys())[0].split("/")[-1].split("-")
            start_idx = int(aln_start) - 1
            end_idx = int(aln_end)
        except Exception:
            start_idx = 0
            end_idx = len(alignment_seqs[0])
        tokenized = self.tokenizer(alignment_seqs, return_tensors="pt", padding=True)
        alignment_ids = tokenized["input_ids"][:, 1:-1]
        return alignment_ids, start_idx, end_idx

    def _count_matrix_from_structure_alignment(
        self, alignment_dict: dict[str, str]
    ) -> torch.Tensor | None:
        alignment_seqs = list(alignment_dict.values())
        if not alignment_seqs:
            return None
        tokenized = self.tokenizer(alignment_seqs, return_tensors="pt", padding=True)
        return tokenized["input_ids"][:, 1:-1]

    def _alignment_log_probs(self, alignment_ids: torch.Tensor) -> torch.Tensor:
        if self.sample_ratio < 1.0:
            sample_size = max(1, int(len(alignment_ids) * self.sample_ratio))
            sample_indices = random.sample(range(len(alignment_ids)), sample_size)
            alignment_ids = alignment_ids[sample_indices]
        count_matrix = torch.zeros(alignment_ids.size(1), self.tokenizer.vocab_size)
        for idx in range(alignment_ids.size(1)):
            count_matrix[idx] = torch.bincount(
                alignment_ids[:, idx], minlength=self.tokenizer.vocab_size
            )
        count_matrix = (count_matrix / count_matrix.sum(dim=1, keepdim=True)).to(self.device)
        return torch.log_softmax(count_matrix, dim=-1)

    @torch.no_grad()
    def _precompute_logits(
        self,
        wt_sequence: str,
        context: VenusREMProteinContext,
    ) -> torch.Tensor:
        self._check_file(context.residue_fasta, "VenusREM residue FASTA")
        self._check_file(context.structure_fasta, "VenusREM structure FASTA")
        if context.aa_seq_aln_file is not None:
            self._check_file(context.aa_seq_aln_file, "VenusREM residue alignment")
        if context.struc_seq_aln_file is not None:
            self._check_file(context.struc_seq_aln_file, "VenusREM structure alignment")

        fasta_seq = _read_fasta_sequence(context.residue_fasta).upper()
        if fasta_seq != wt_sequence:
            raise ValueError(
                f"WT sequence does not match residue FASTA for {context.protein_name}: "
                f"{len(wt_sequence)} input aa vs {len(fasta_seq)} FASTA aa"
            )

        structure_raw = _read_fasta_sequence(context.structure_fasta)
        structure_sequence = [int(item) for item in structure_raw.split(",") if item.strip()]
        if len(structure_sequence) != len(wt_sequence):
            raise ValueError(
                f"structure sequence length mismatch for {context.protein_name}: "
                f"{len(structure_sequence)} structure tokens vs {len(wt_sequence)} aa"
            )

        ss_input_ids = _tokenize_structure_sequence(structure_sequence).to(self.device)
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

        logits = torch.log_softmax(outputs.logits[0][1:-1, :], dim=-1)
        if self.alpha == 0.0 or not self.logit_mode:
            return logits

        use_aa = "aa_seq_aln" in self.logit_mode and context.aa_seq_aln_file is not None
        use_struc = "struc_seq_aln" in self.logit_mode and context.struc_seq_aln_file is not None

        if use_aa and not use_struc:
            alignment_dict = _read_multi_fasta(context.aa_seq_aln_file or "")
            alignment_ids, aln_start, aln_end = self._count_matrix_from_residue_alignment(
                alignment_dict
            )
            if aln_end > logits.size(0) or alignment_ids.size(1) != (aln_end - aln_start):
                raise ValueError("residue alignment length is incompatible with WT sequence")
            modified = logits[aln_start:aln_end, :]
            for _ in range(self.sample_times):
                count_log_probs = self._alignment_log_probs(alignment_ids)
                modified = (1.0 - self.alpha) * modified + self.alpha * count_log_probs
            return torch.cat([logits[:aln_start], modified, logits[aln_end:]], dim=0)

        if use_struc and not use_aa:
            alignment_dict = _read_multi_fasta(context.struc_seq_aln_file or "")
            alignment_ids = self._count_matrix_from_structure_alignment(alignment_dict)
            if alignment_ids is None:
                return logits
            count_log_probs = self._alignment_log_probs(alignment_ids)
            if count_log_probs.size(0) != logits.size(0):
                raise ValueError("structure alignment length is incompatible with WT sequence")
            return (1.0 - self.alpha) * logits + self.alpha * count_log_probs

        if use_aa and use_struc:
            plm_logits = logits.clone()
            struc_dict = _read_multi_fasta(context.struc_seq_aln_file or "")
            struc_ids = self._count_matrix_from_structure_alignment(struc_dict)
            if struc_ids is not None:
                struc_log_probs = self._alignment_log_probs(struc_ids)
                if struc_log_probs.size(0) != logits.size(0):
                    raise ValueError("structure alignment length is incompatible with WT sequence")
                logits = (1.0 - self.alpha) * plm_logits + self.alpha * struc_log_probs

            aa_dict = _read_multi_fasta(context.aa_seq_aln_file or "")
            aa_ids, aln_start, aln_end = self._count_matrix_from_residue_alignment(aa_dict)
            aa_log_probs = self._alignment_log_probs(aa_ids)
            if aln_end > logits.size(0) or aa_log_probs.size(0) != (aln_end - aln_start):
                raise ValueError("residue alignment length is incompatible with WT sequence")
            modified = (1.0 - self.alpha) * logits[aln_start:aln_end, :] + self.alpha * aa_log_probs
            return torch.cat([plm_logits[:aln_start], modified, plm_logits[aln_end:]], dim=0)

        return logits

    def score_batch(
        self,
        wt_sequence: str,
        mutants: list[str],
        *,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, float | bool | str | None]]:
        wt_sequence = str(wt_sequence).strip().upper()
        parsed = [parse_mutant(mutant, wt_sequence) for mutant in mutants]
        if not any(pm.valid for pm in parsed):
            return [{"fitness_score": None, "valid": False, "error": pm.error} for pm in parsed]

        try:
            protein_context = self._resolve_context(wt_sequence, context)
            logits = self._precompute_logits(wt_sequence, protein_context)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            return [
                {"fitness_score": None, "valid": False, "error": pm.error or error}
                for pm in parsed
            ]

        outputs: list[dict[str, float | bool | str | None]] = []
        for pm in parsed:
            if not pm.valid:
                outputs.append({"fitness_score": None, "valid": False, "error": pm.error})
                continue

            score = 0.0
            try:
                for wt_aa, pos, mt_aa in zip(pm.wt_list, pm.pos_list, pm.mt_list):
                    delta = logits[pos - 1, self.vocab[mt_aa]] - logits[pos - 1, self.vocab[wt_aa]]
                    score += float(
                        delta.item()
                    )
            except Exception as exc:
                outputs.append(
                    {
                        "fitness_score": None,
                        "valid": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            outputs.append({"fitness_score": float(score), "valid": True, "error": ""})
        return outputs

    def teardown(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
