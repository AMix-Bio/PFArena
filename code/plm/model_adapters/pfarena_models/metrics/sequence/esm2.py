"""
ESM-2 masked-marginal scoring adapter.

Score definition (per mutant string):
  fitness_score = sum(logP(mut_aa) - logP(wt_aa))
where mutant supports:
  - single substitution: A42G
  - multi substitution:  A42G:F88Y
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
MUT_RE = re.compile(r"([A-Z])(\d+)([A-Z])")


@dataclass
class ParsedMutation:
    mutant: str
    wt_list: list[str]
    pos_list: list[int]  # 1-indexed
    mt_list: list[str]
    valid: bool
    error: str = ""


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
        m = MUT_RE.fullmatch(sub)
        if not m:
            return ParsedMutation(
                mutant, [], [], [], False, f"cannot parse sub-mutation: {sub}"
            )
        wt_aa, pos_str, mt_aa = m.groups()
        pos = int(pos_str)

        if wt_aa not in AA_VOCAB or mt_aa not in AA_VOCAB:
            return ParsedMutation(
                mutant, [], [], [], False, f"non-standard amino acid in {sub}"
            )
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


def _get_window(
    wt_sequence: str, position_1idx: int, max_window: int
) -> tuple[str, int]:
    """Return (window_sequence, relative_position_1idx)."""
    seq_len = len(wt_sequence)
    if seq_len <= max_window:
        return wt_sequence, position_1idx

    half = max_window // 2
    start = max(1, position_1idx - half)
    end = min(seq_len, start + max_window - 1)
    if (end - start + 1) < max_window:
        start = max(1, end - max_window + 1)

    window_seq = wt_sequence[start - 1 : end]
    rel_pos = position_1idx - start + 1
    return window_seq, rel_pos


class ESM2FitnessScorer:
    """ESM2 masked-marginal scorer reused by esm2 service."""

    def __init__(
        self,
        model_dir: str,
        *,
        device: str = "cuda",
        batch_size: int = 8,
        max_window: int = 1022,
    ):
        self.model_dir = model_dir
        self.batch_size = max(1, int(batch_size))
        self.max_window = int(max_window)

        if device == "cuda" and torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        if "esm2" not in model_dir.lower():
            raise ValueError(
                "esm2 is ESM2-only; please provide an ESM2 checkpoint path"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self._forward_lock = threading.Lock()

        self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is None:
            raise ValueError("Tokenizer has no mask token; ESM2 masked scoring needs it")

        self.aa_token_ids: dict[str, int] = {}
        for aa in AA_VOCAB:
            tok = self.tokenizer.encode(aa, add_special_tokens=False)
            if len(tok) != 1:
                raise ValueError(f"Tokenizer returned {len(tok)} tokens for amino acid {aa}")
            self.aa_token_ids[aa] = tok[0]

    @torch.no_grad()
    def _precompute_position_scores(
        self, wt_sequence: str, positions: list[int]
    ) -> dict[int, dict[str, float]]:
        if not positions:
            return {}

        prepared_inputs: list[tuple[int, torch.Tensor, torch.Tensor, int]] = []
        for pos in positions:
            window_seq, rel_pos = _get_window(
                wt_sequence=wt_sequence,
                position_1idx=pos,
                max_window=self.max_window,
            )
            enc = self.tokenizer(window_seq, return_tensors="pt", add_special_tokens=True)
            input_ids = enc["input_ids"][0]
            attention_mask = enc["attention_mask"][0]

            token_index = rel_pos  # residue token starts after BOS
            if token_index >= input_ids.shape[0] - 1:
                raise ValueError(f"invalid token index at position {pos}")

            masked_ids = input_ids.clone()
            masked_ids[token_index] = self.mask_token_id
            prepared_inputs.append((pos, masked_ids, attention_mask, token_index))

        position_scores: dict[int, dict[str, float]] = {}
        for i in range(0, len(prepared_inputs), self.batch_size):
            chunk = prepared_inputs[i : i + self.batch_size]
            ids = torch.stack([x[1] for x in chunk], dim=0).to(self.device)
            mask = torch.stack([x[2] for x in chunk], dim=0).to(self.device)
            # Hugging Face ESM rotary embeddings keep mutable cos/sin caches
            # on the module. Guard forward passes because the service can
            # process multiple requests concurrently in different threads.
            with self._forward_lock:
                logits = self.model(input_ids=ids, attention_mask=mask).logits

            for row_idx, (pos, _, _, token_idx) in enumerate(chunk):
                row_logits = logits[row_idx, token_idx, :]
                row_log_probs = torch.log_softmax(row_logits, dim=-1)
                position_scores[pos] = {
                    aa: float(row_log_probs[self.aa_token_ids[aa]].item())
                    for aa in AA_VOCAB
                }
        return position_scores

    def score_batch(
        self, wt_sequence: str, mutants: list[str]
    ) -> list[dict[str, float | bool | str | None]]:
        wt_sequence = str(wt_sequence).strip().upper()
        parsed = [parse_mutant(mutant, wt_sequence) for mutant in mutants]
        valid_positions = sorted({p for pm in parsed if pm.valid for p in pm.pos_list})
        position_scores = self._precompute_position_scores(wt_sequence, valid_positions)

        outputs: list[dict[str, float | bool | str | None]] = []
        for pm in parsed:
            if not pm.valid:
                outputs.append(
                    {"fitness_score": None, "valid": False, "error": pm.error}
                )
                continue

            score = 0.0
            missing = False
            for wt_aa, pos, mt_aa in zip(pm.wt_list, pm.pos_list, pm.mt_list):
                aa_scores = position_scores.get(pos)
                if aa_scores is None:
                    missing = True
                    break
                score += aa_scores[mt_aa] - aa_scores[wt_aa]

            if missing:
                outputs.append(
                    {
                        "fitness_score": None,
                        "valid": False,
                        "error": "missing precomputed position scores",
                    }
                )
            else:
                outputs.append({"fitness_score": float(score), "valid": True, "error": ""})

        return outputs

    def teardown(self) -> None:
        del self.model
        if self.device == "cuda":
            torch.cuda.empty_cache()
