# Model adapters

This directory contains only the scoring adapters and ProGen2 model-definition
files required by the six released protein-model runners. Service components,
runtime configuration, cached datasets, and model parameters are excluded.

Machine-specific defaults in the ProSST and S3F adapters are replaced by
`PROSST_*` and `S3F_*` environment variables. Their scoring logic is unchanged.

Upstream repositories and model parameters remain external dependencies and
must be obtained from their official sources.
