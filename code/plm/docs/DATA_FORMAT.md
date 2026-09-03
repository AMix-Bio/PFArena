# Model input contract

All six baselines consume one canonical, label-free dataset directory supplied
by the parent PFArena project. The directory contains `dataset.json` and the
following tables:

| File | Purpose |
|---|---|
| `proteins.csv` | wild-type proteins or complexes and ordered chain sequences |
| `samples.csv` | evaluation candidates and stable sample/query identities |
| `substitutions.csv` | chain, position, WT residue, and mutant residue for each candidate |
| `queries.csv` | task/query membership and expected candidate counts |
| `context_samples.csv` | conditioning candidates required by T3/T4 |
| `context_substitutions.csv` | substitutions for conditioning candidates |
| `provided_context.csv` | task-provided contextual values |
| `msa_contexts.csv` | chain-level MSA paths and checksums |

`dataset.json` records the schema version, dataset identity, table counts, and
one combined content hash. `dataset_io.py` validates all tables before inference.
The released predictions use dataset ID `pfarena_t1_t4` and dataset hash
`183dc180cf74b3ea950aec6b51c5c55e8a25c444dc262625393b70ef750bb093`;
the parent benchmark release must expose the same identity and content hash.

## Stable identifiers

- `dataset_hash`: complete canonical model-input release.
- `sequence_sha256`: full wild-type protein or complex sequence.
- `context_sha256`: one chain sequence shared by MSA, structure, and cache data.
- `sample_id`: stable lowercase SHA256 identifier for one candidate within one query.
- `query_id`: candidate set over which ranking metrics are computed.

For structure-aware models, a separately supplied structure manifest must contain
`dataset_hash`, `context_sha256`, `sequence_length`, `structure_status`,
`structure_qc`, `structure_source`, `pdb_path`, and `pdb_sha256`. Relative structure paths are resolved first
against the manifest directory and then against the project root. Relative MSA
paths are resolved against the canonical dataset directory. Predicted
structures are not distributed in this module.

WT structures and MSAs are shared by candidates with the same chain context;
mutant-specific structures or MSAs are not required. Output columns and
multi-substitution score semantics are defined by each model's
`output_schema_candidates.json`.

`provided_context.csv` is validated as part of the benchmark release but is not
used by these task-agnostic protein-model baselines.
