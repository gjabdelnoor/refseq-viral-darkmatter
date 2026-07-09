# Preliminary viral dark-matter test set

**File:** `preliminary_test_set.tsv` (498 genomes)
**Purpose:** honest holdout for a frozen-EVO2 / frozen-ESM-2 taxonomic classifier.

## How it was built
1. Candidate pool: NCBI GenBank nucleotide, `txid10239` viruses, **"complete genome"**,
   released **after 2025-09-22** (DeepVirus preprint date, bioRxiv 2025.09.22.677955),
   phages excluded, 8 mega-surveillance taxa removed → 7,866 unique genomes.
2. Novelty scored two ways vs the union of all 19,433 RefSeq viral genomes (release 235):
   - `nuc_novelty` = 1 − fraction of the genome's 21-mers seen in any known virus (EVO2 / DNA arm).
   - `prot_novelty` = same in 6-frame-translated **protein 7-mer** space (ESM-2 / homology arm;
     the field-standard "no known-protein homolog" definition of dark matter).
3. **Sorted OUT** every genome I recognize as a characterized species/genus (or a viroid),
   and everything with a clear protein homolog (`prot_novelty < 0.45`). 7,368 removed
   (see `sorted_out_recognized.tsv`). The 498 that remain are unfamiliar + divergent.

## CRITICAL caveats (why this is *preliminary*)
- **Labels are provisional.** `provisional_identity` is the submitter's GenBank description,
  NOT wet-lab- or phylogeny-confirmed. Many are MAG (metagenome-assembled) with no isolate.
- This is the **OOD / abstention** arm of the test, not an accuracy set. For most of these the
  correct classifier behaviour is to **decline / flag novel**, not to emit a confident family.
- k=7 protein / scaled=100 saturates for the most divergent cases; treat rank as ordinal.
- Viroids and other non-coding elements were removed (protein-space novelty is undefined for them).
- Known limitation: furtivovirus/Manesviridae (J.Virol 2026) is NOT captured — too recent / not
  public with a matching title in the query window.

## Companion files
- `novelty_final.tsv` — all 7,866 scored genomes (nuc+prot novelty, test_tier).
- `sorted_out_recognized.tsv` — the 7,368 removed, with reason.
- `queries_dedup.fna` — the sequences. `compute_novelty*.py` / `run_novelty.sh` — reproduce.
