# S3F

S3F combines ESM-2 sequence representations with structural surface context. `prepare_inputs.py` validates and copies the required chain PDBs, and `prepare_candidate_surfaces.py` precomputes surface graphs with memory-bounded curvature evaluation. Inference sums structure-aware masked-marginal scores over substituted positions and chains.

Surface graphs are rebuildable runtime artifacts and are not distributed. The
PDBs and manifest from the accompanying PFArena PLM Structures dataset are
their authoritative inputs.
