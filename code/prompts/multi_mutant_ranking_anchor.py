"""
    T3: Anchor-informed multi-mutant ranking
"""
def multi_mutant_ranking_anchor_template():
    prompt = \
"""
You are an expert protein engineer and computational biologist specializing in deep mutational scanning (DMS) and mutation-effect prediction.

### TASK GOAL
Given a wild-type protein sequence, its experimental assay context, and a specific list of candidate multi-mutations, **rank all candidate multi-mutations from best to worst** according to their expected target fitness metric.
The ground-truth DMS score of an anchor mutant appearing in every candidate, which may be single-site or multi-site, will also be provided.

### REASONING & EVIDENCE BOUNDARIES
1. **Biochemical Deductions**: Compare the candidates based on residue chemistry, conservation, secondary structure propensities, steric packing, hydrophobic cores, electrostatic interactions, and sequence motifs within the supplied wild-type sequence.
2. **Assay Alignment**: Evaluate relative effects of these specific substitutions on the supplied assay readout. Rank mutations that better preserve or enhance structural/functional requirements above those that introduce severe clashes, charge mismatch, or instability.
3. **No Hallucination**: Do NOT invent or claim specific numerical model scores, experimental PDB coordinates, literature measurements, or alignments that are not logically derivable from sequence biochemistry.
4. **Single-Response Constraint**: You have no external tools, browsing, or follow-up turns. Complete the ranking in this single response.

### STRICT MUTATION & FORMAT CONSTRAINTS
1. **Closed Set Principle**: You MUST ONLY rank the mutations provided in the list above. Do NOT introduce new mutations, insertions, deletions, or wild-type strings.
2. **Multi-Mutation Candidates**: A candidate contains multiple mutations joined by + (e.g., K23A+A40P+T52S), with the anchor mutant included. When ranking multi-mutation candidates, account for their combined effects.
3. **Exact Copy & Completeness**:
   - The output list MUST contain **exactly `total_candidates` items**.
   - Every candidate from the input list MUST appear **exactly once** (no duplicates, no omissions).
   - Each mutation string MUST be copied **exactly as provided**.
4. **Strict Validation**:
   - Every mutation position follows `1 <= position <= Length`.
   - `WT` MUST strictly match the character at `wildtype_sequence[position - 1]`.
   - `MUT` MUST be strictly different from `WT` (no synonymous/no-op mutations).

---

### OUTPUT FORMAT
Return a **valid JSON object ONLY** with NO markdown code block wrappers, prefix, or conversational text. Use the following exact JSON schema:

{{
  "ranking": [
    "<best mutant copied exactly from the provided list>",
    "<second-best mutant copied exactly from the provided list>",
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

3. ANCHOR MUTANT CONTEXT
- **Anchor Mutant**: {anchor_mutant}
- **Anchor DMS Score**: {anchor_DMS_score} 

4. CANDIDATE MUTATIONS TO RANK (Total Candidates: {num_candidates})
{candidate_mutants}
"""
    return prompt