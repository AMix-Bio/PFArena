# Model adapters

This directory contains the scoring adapters, ProGen2 model definitions, and
S3F inference source required by the protein-model runners. Model weights,
datasets, and service components are excluded.

The runners accept explicit model-resource paths; adapters also support
`PROSST_*` and `S3F_*` environment variables.

`s3f_runtime/` includes the evaluated S3F bridge, supporting modules, and task
configuration. Internal default paths and unused configuration sections were
removed without changing scoring or model parameters. Training scripts are not
included. Required external resources are listed in `../docs/DEPENDENCIES.md`.
