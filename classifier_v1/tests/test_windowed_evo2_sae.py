from __future__ import annotations

import csv
import gzip
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from windowed_evo2_sae import (  # noqa: E402
    EMBEDDING_DIM,
    SAE_TOP_K,
    GenomeRecord,
    SaeTopK,
    _assert_full_window_was_processed,
    deterministic_windows,
    load_labeled_genomes,
    normalize_iupac_to_n,
    run_bulk,
)


class FakeClient:
    def __init__(self) -> None:
        self.embed_calls = 0
        self.sae_calls = 0

    def embed(self, sequence: str) -> np.ndarray:
        self.embed_calls += 1
        return np.full(EMBEDDING_DIM, len(sequence), dtype=np.float32)

    def sae_top50(self, sequence: str) -> SaeTopK:
        self.sae_calls += 1
        count = 3
        return SaeTopK(
            feature_ids=np.arange(100, 100 + count, dtype=np.int32),
            max_act=np.arange(count, dtype=np.float32) + 1,
            total_act=np.arange(count, dtype=np.float32) + 11,
            active_in=np.arange(count, dtype=np.float32) + 21,
            n_tokens=np.full(count, len(sequence), dtype=np.float32),
        )


def test_iupac_and_evenly_spaced_windows() -> None:
    assert normalize_iupac_to_n("aCGTryswkmbdhvun") == "ACGTNNNNNNNNNNNN"
    windows = deterministic_windows("A" * 10_000, window_bp=8192, max_windows=8)
    assert [(start, end) for start, end, _ in windows] == [(0, 8192), (1808, 10_000)]
    long_windows = deterministic_windows("A" * 65_536, window_bp=8192, max_windows=8)
    assert len(long_windows) == 8
    assert long_windows[0][0] == 0
    assert long_windows[-1][1] == 65_536
    assert all(len(window) == 8192 for _, _, window in long_windows)


def test_truncation_guard_rejects_a_shorter_server_response() -> None:
    sae = SaeTopK(
        feature_ids=np.array([1], dtype=np.int32),
        max_act=np.array([1.0], dtype=np.float32),
        total_act=np.array([1.0], dtype=np.float32),
        active_in=np.array([1.0], dtype=np.float32),
        n_tokens=np.array([2000.0], dtype=np.float32),
    )
    try:
        _assert_full_window_was_processed(sae, 8192)
    except RuntimeError as error:
        assert "silently truncated" in str(error)
    else:
        raise AssertionError("expected a truncation error")


def test_gz_fasta_label_join_and_atomic_resume(tmp_path: Path) -> None:
    fasta = tmp_path / "viral.fna.gz"
    labels = tmp_path / "labels.tsv"
    with gzip.open(fasta, "wt", encoding="utf-8") as handle:
        handle.write(">A.1 an ambiguous genome\nACGTRYSW\n>B.1 unlabeled\nACGT\n>C.1\n" + "A" * 9000 + "\n")
    with open(labels, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["accession", "family", "genus"], delimiter="\t")
        writer.writeheader()
        writer.writerow({"accession": "A.1", "family": "F1", "genus": "G1"})
        writer.writerow({"accession": "C.1", "family": "F2", "genus": "G2"})
        writer.writerow({"accession": "MISSING.1", "family": "F3", "genus": "G3"})

    records, missing = load_labeled_genomes(fasta, labels)
    assert [record.accession for record in records] == ["A.1", "C.1"]
    assert records[0].sequence == "ACGTNNNN"
    assert missing == ["MISSING.1"]

    out = tmp_path / "out"
    first = FakeClient()
    summary = run_bulk(
        records,
        output_dir=out,
        client=first,
        window_bp=8192,
        max_windows=8,
        missing_labels=missing,
    )
    # A.1 produces one short window; C.1 produces two endpoint-aligned windows.
    assert first.embed_calls == 3
    assert first.sae_calls == 3
    assert summary["completed_windows"] == 3
    assert summary["failed_windows"] == 0

    embeddings = np.load(out / "matrices/evo2_window_embeddings.npy")
    means = np.load(out / "matrices/evo2_genome_mean.npy")
    feature_ids = np.load(out / "matrices/sae_top50_feature_ids.npy")
    mask = np.load(out / "matrices/window_mask.npy")
    assert embeddings.shape == (2, 8, EMBEDDING_DIM)
    assert means.shape == (2, EMBEDDING_DIM)
    assert feature_ids.shape == (2, 8, SAE_TOP_K)
    assert mask.tolist() == [[True, False, False, False, False, False, False, False], [True, True, False, False, False, False, False, False]]
    assert feature_ids[0, 0, :3].tolist() == [100, 101, 102]
    assert feature_ids[0, 0, 3] == -1
    assert np.isnan(embeddings[0, 1]).all()
    assert np.allclose(means[1], 8192.0)

    second = FakeClient()
    resumed = run_bulk(records, output_dir=out, client=second, window_bp=8192, max_windows=8)
    assert resumed["completed_windows"] == 3
    assert second.embed_calls == 0
    assert second.sae_calls == 0
