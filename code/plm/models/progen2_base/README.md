# ProGen2-base

The runner evaluates complete mutant and WT sequences in both causal directions. For each direction, it sums the mean token NLL of every non-overlapping context chunk; the negative average of the two sums is divided by the terminalized sequence length. The primary score is the mutant score minus its WT-chain score. Terminal tokens and chunk handling are fixed in `model.json`. Multi-substitution candidates are scored jointly within each chain, and mutated-chain contributions are summed.
