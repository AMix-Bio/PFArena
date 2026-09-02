# S3F-MSA

S3F-MSA combines the S3F score with an EVE evolutionary score. Each chain uses the supplied full-chain UniRef100/MMseqs2 MSA; EVE sequence weights use theta 0.01 for viral assays and 0.2 otherwise. Five seeds are trained for 400,000 steps, and 20,000 latent samples per seed are used for evolutionary indices. S3F and the five-seed EVE ensemble are standardized within each evaluation query and averaged. Multi-substitution candidates receive a direct joint EVE score within each chain.

`prepare_v7_inputs.py` creates the EVE-formatted inputs, and `run_stage.py`
implements sequence weighting, five-seed training, and scoring. Generated EVE
inputs, checkpoints, logs, weights, and intermediate scores are runtime
artifacts and are not distributed; the complete final S3F-MSA predictions and
evaluation results are retained under `results/s3f_msa`.
