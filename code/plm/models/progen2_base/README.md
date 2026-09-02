# ProGen2-base

The runner evaluates complete mutant and WT sequences in both causal directions. The primary score is the mutant-minus-WT bidirectional mean log-likelihood; terminal tokens and non-overlapping chunk handling are fixed in `model.json`. This is direct joint-sequence scoring for multi-substitution candidates.
