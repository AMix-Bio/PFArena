"""
    T1: Single-mutant generation
"""
def single_mutant_generation_template():
    prompt = \
"""
You are an expert protein engineer and computational biologist specializing in deep mutational scanning (DMS) and mutation-effect prediction.

### TASK GOAL
Given a wild-type protein sequence and its experimental assay context, predict the **top 40 single point mutations** (WTposMUT format) that optimize the target fitness metric, ordered from highest expected fitness to lowest expected fitness.

### REASONING & EVIDENCE BOUNDARIES
1. **Biochemical Deductions**: Analyze residue chemistry, conservation, secondary structure propensities, steric packing, hydrophobic cores, electrostatic interactions, and sequence motifs within the supplied wild-type sequence.
2. **Assay Alignment**: Align every ranked mutation strictly with the supplied assay readout. For example, if evaluating stability/abundance, prioritize mutations that improve hydrophobic packing or thermostability without disrupting necessary structural dynamics.
3. **No Hallucination**: Do NOT invent or claim specific numerical model scores, experimental PDB coordinates, literature measurements, or alignments that are not logically derivable from sequence biochemistry.
4. **Single-Response Constraint**: You have no external tools, browsing, or follow-up turns. Complete the analysis in this single response.

### STRICT MUTATION & FORMAT CONSTRAINTS
1. **Ranking Size**: The output list MUST contain **exactly 40 mutations**, ordered from best to worst.
2. **Format**: Every mutant MUST be represented in standard 1-indexed `WTposMUT` format (e.g., `H24R`, `A15V`).
3. **Alphabet**: Both `WT` and `MUT` must be standard 20 amino acid single-letter codes: `A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y`.
4. **Uniqueness**: Every mutation string MUST appear exactly once (no duplicates, no omissions within your top-40 list).
5. **Strict Validation**:
   - Every mutation position follows `1 <= position <= Length`.
   - `WT` MUST strictly match the character at `wildtype_sequence[position - 1]`.
   - `MUT` MUST be strictly different from `WT` (no synonymous/no-op mutations).
6. **Search Space**: Consider all valid single amino acid substitutions across the full wild-type sequence, then return only the top 40.
7. **Prohibited Formats**: Multi-site mutations, insertions (`ins`), deletions (`del`), stop codons (`*`), HGVS notations, or numerical confidence scores.

---

### OUTPUT FORMAT
Return a **valid JSON object ONLY** with NO markdown code block wrappers, prefix, or conversational text. Use the following exact JSON schema:

{{
  "ranking": [
    "<best mutant>",
    "<second-best mutant>",
    "..."
  ]
}}

---

### INSTANCE DATA
1. ASSAY CONTEXT & TARGET METRICS
- **UniProt ID**: {uniprot_id}
- **Primary Task Class**: {primary_task_class}
- **Fitness Metric Type**: {fitness_type}
- **Assay Readout Subclass**: {readout_subclass}

2. INPUT WILD-TYPE SEQUENCE (Length: {sequence_length})
`{wildtype_sequence}`
"""
    return prompt