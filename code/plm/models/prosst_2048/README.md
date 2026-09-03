# ProSST-2048

The runner uses the official `AI4Protein/ProSST-2048` model. AF3 PDBs are converted to the 2,048-state ProSST structure alphabet and paired with the WT sequence. The score sums WT-marginal mutant log-odds over substituted positions and mutated chains. Model resources must be obtained from the official repository.

Structure-token caches are generated from the supplied AF3 structures during
inference and are not distributed. `run_candidates.py` contains the complete
conversion and scoring path.
