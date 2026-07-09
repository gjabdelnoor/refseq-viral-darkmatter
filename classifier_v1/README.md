# Windowed Frozen-EVO2 + SAE Extraction

`windowed_evo2_sae.py` is the resumable data-production stage for the frozen
EVO2 classifier experiment. It does not fine-tune or download model weights.
It calls the already-running local Gradio backend at `127.0.0.1:7860`.

Each labelled genome is cleaned by converting every non-`ACGT` IUPAC character
to `N`, then contributes `min(8, ceil(length / 8192))` evenly-spaced windows.
Every complete window is atomically cached under `evo2_sae_windows/cache/`.
Interrupted work can therefore be restarted with the same command.

The runner rejects a server that silently truncates a requested window: it
checks the SAE CSV's reported token count against the submitted sequence. Do
not use `--allow-server-truncation` for this experiment; it exists only for
diagnosing an older backend with a lower `MAX_SEQ_LEN` cap.

Run a one-genome endpoint canary first:

```bash
cd /home/gabriel/data/refseq-viral
python3 classifier_v1/windowed_evo2_sae.py --canary
```

Launch the complete eukaryotic RefSeq job after the canary succeeds:

```bash
cd /home/gabriel/data/refseq-viral
python3 classifier_v1/windowed_evo2_sae.py \
  --fasta sequences/viral.1.1.genomic.fna.gz \
  --labels refseq_viral_eukaryote.tsv \
  --out classifier_v1/evo2_sae_windows
```

Useful operational flags: `--limit N`, `--canary`, `--retries N`,
`--stop-on-error`, `--matrix-dtype float16`, and `--endpoint URL`.

The source of truth is the per-window `.npz` cache. `matrices/` is replaced
atomically after each run and contains fixed `[genome, 8, ...]` arrays plus
`genome_meta.tsv`, `window_meta.tsv`, and `manifest.json`. Failed windows are
left uncached so a later invocation retries them.
