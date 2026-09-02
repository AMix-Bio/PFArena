# S3F

S3F combines ESM-2 sequence representations with AF3-derived structural surface context. `prepare_inputs.py` freezes the required chain PDBs, and `prepare_candidate_surfaces.py` precomputes validated surface graphs with chunked curvature evaluation. Inference sums structure-aware masked-marginal scores over substituted positions and chains.

Prepared PDB copies and surface graphs are rebuildable runtime artifacts and
are not distributed. The supplied chain-structure manifest is their
authoritative input.
