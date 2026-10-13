# ESM-2 650M

The runner uses the official `facebook/esm2_t33_650M_UR50D` model. It computes WT masked marginals and sums mutant-minus-WT log-probabilities over substituted positions and mutated chains. Inputs longer than 1,022 residues use the registered centered-window protocol. Model resources must be obtained from the official repository.
