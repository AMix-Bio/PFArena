# External dependencies

The repository bundles the minimal scoring adapters required by the runners,
but does not redistribute upstream model weights or full third-party
repositories. Supply these resources through the command-line interfaces of the
model entry points.

| Component | Required official resource | Local configuration |
|---|---|---|
| ESM-2 | [`facebook/esm2_t33_650M_UR50D`](https://huggingface.co/facebook/esm2_t33_650M_UR50D) and [ESM](https://github.com/facebookresearch/esm) | `models/esm2/model.json` |
| ProGen2-base | [ProGen2 model and tokenizer](https://github.com/salesforce/progen) | `models/progen2_base/model.json` |
| ProSST / VenusREM | [`AI4Protein/ProSST-2048`](https://huggingface.co/AI4Protein/ProSST-2048), [ProSST](https://github.com/ai4protein/ProSST), and [VenusREM](https://github.com/ai4protein/VenusREM) | corresponding model JSON files |
| S3F | [S3F](https://github.com/DeepGraphLearning/S3F) model and source plus ESM-2 resources | `models/s3f/model.json` |
| EVE | [ProteinGym EVE implementation](https://github.com/OATML-Markslab/ProteinGym/tree/144fe22b07dfaeec2b366f2346203a9838a55b4c/proteingym/baselines/EVE) | source-file hashes in `models/s3f_msa/run_stage.py` |
| AlphaFold 3 | [official runtime](https://github.com/google-deepmind/alphafold3) and model parameters | required only to regenerate the companion structures |
| Shared scoring adapters | bundled minimal adapter subset | `model_adapters` (default) |

No upstream model parameters or model-specific caches are included. Obtain the
required resources from their official sources. S3F-MSA EVE checkpoints are
trained from the benchmark MSAs with the provided workflow. The required WT
chain PDBs are distributed in the accompanying structure dataset; AlphaFold 3
parameters and intermediate outputs are not included.

For EVE, check out ProteinGym commit
`144fe22b07dfaeec2b366f2346203a9838a55b4c` and pass its
`proteingym/baselines/EVE` directory as `--eve-root`. A different EVE repository
or revision may not provide the same entry points and will fail source validation.

The verified runtime versions are recorded in
`environments/runtime_versions.csv`, and each protein model has a corresponding
specification under `environments/`. Separate environments are retained because
the upstream implementations require different PyTorch, Transformers, and
compiled-extension versions.
