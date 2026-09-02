# VenusREM

VenusREM uses synchronized amino-acid sequence, ProSST-2048 structure tokens, and UniRef100/MMseqs2 alignment inputs. The deployed `aa_seq_aln` mode uses alpha 0.8, sampling ratio 1.0, and one sampling pass. Candidate scores sum sitewise mutant log-odds over substitutions and mutated chains.

`prepare_v7_inputs.py` builds synchronized inputs from the supplied canonical
dataset, MSA, and AF3 resources. These model-specific inputs and inference
caches are not distributed.
