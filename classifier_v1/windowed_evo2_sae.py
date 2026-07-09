#!/usr/bin/env python3
"""Resumable windowed EVO2 + SAE extraction through the local Gradio API.

The runner deliberately keeps EVO2 frozen.  It turns each labelled viral genome
into at most ``max_windows`` deterministic, evenly-spaced windows, obtains the
mean-pooled block-26 embedding and Goodfire SAE top-50 features for each window,
and stores each completed window in an independent atomic cache file.  A later
run only calls the API for missing or invalid cache entries.

The cache is the source of truth.  ``matrices/`` is a replace-in-place,
fixed-shape materialisation for classifier training:

* ``evo2_window_embeddings.npy``: [genome, window, 4096]
* ``evo2_genome_mean.npy``:       [genome, 4096]
* ``sae_top50_*.npy``:             [genome, window, 50]
* ``window_mask.npy``:             completed windows

It uses Gradio's documented queue API rather than browser automation.  The
endpoint names and returned file positions match the existing EVO2 UI:
``/_embeddings_callback`` returns the mean embedding at result[2], and
``/_features_callback`` returns the SAE CSV at result[4].
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence
from urllib.parse import urljoin

import numpy as np
import requests


SCHEMA_VERSION = "evo2-sae-window-cache-v1"
EMBEDDING_DIM = 4096
SAE_TOP_K = 50
DEFAULT_FASTA = Path("/home/gabriel/data/refseq-viral/sequences/viral.1.1.genomic.fna.gz")
DEFAULT_LABELS = Path("/home/gabriel/data/refseq-viral/refseq_viral_eukaryote.tsv")
DEFAULT_OUT = Path("/home/gabriel/data/refseq-viral/classifier_v1/evo2_sae_windows")
DNA = frozenset("ACGT")


@dataclass(frozen=True)
class GenomeRecord:
    """A FASTA record joined to its immutable labelled-table row."""

    index: int
    accession: str
    sequence: str
    original_length: int
    labels: Mapping[str, str]

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class WindowSpec:
    genome_index: int
    accession: str
    window_index: int
    start_bp: int
    end_bp: int
    sequence: str
    genome_sha256: str

    @property
    def sequence_sha256(self) -> str:
        return hashlib.sha256(self.sequence.encode("ascii")).hexdigest()


@dataclass(frozen=True)
class SaeTopK:
    feature_ids: np.ndarray
    max_act: np.ndarray
    total_act: np.ndarray
    active_in: np.ndarray
    n_tokens: np.ndarray


class Evo2SaeClient(Protocol):
    """Small protocol that makes the bulk loop testable without network calls."""

    def embed(self, sequence: str) -> np.ndarray: ...

    def sae_top50(self, sequence: str) -> SaeTopK: ...


def normalize_iupac_to_n(sequence: str) -> str:
    """Upper-case a nucleotide sequence and convert every non-ACGT code to N.

    This intentionally maps all ambiguous IUPAC symbols (and RNA U) to N.  It
    avoids inventing a biologically arbitrary base while preserving length and
    coordinates, which is essential for reproducible window locations.
    """

    cleaned = "".join(sequence.split()).upper()
    return "".join(base if base in DNA else "N" for base in cleaned)


def deterministic_windows(
    sequence: str,
    *,
    window_bp: int = 8192,
    max_windows: int = 8,
) -> list[tuple[int, int, str]]:
    """Return <= max_windows full-span, evenly-spaced windows in source order.

    For a genome shorter than a window, the sole window is the complete genome.
    For longer genomes, use ``ceil(length / window_bp)`` windows capped at eight;
    starts are integer-spaced from zero through the final valid start.  This gives
    exact endpoint coverage without floating-point rounding differences.
    """

    if window_bp <= 0:
        raise ValueError("window_bp must be positive")
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if not sequence:
        return []

    length = len(sequence)
    if length <= window_bp:
        return [(0, length, sequence)]

    count = min(max_windows, math.ceil(length / window_bp))
    last_start = length - window_bp
    starts = [index * last_start // (count - 1) for index in range(count)]
    return [(start, start + window_bp, sequence[start : start + window_bp]) for start in starts]


def _open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else open(
        path, "rt", encoding="utf-8", newline=""
    )


def iter_fasta(path: Path) -> Iterator[tuple[str, str]]:
    """Stream a plain or gzipped FASTA; accession is the first header token."""

    header: str | None = None
    chunks: list[str] = []
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header = line[1:].split()[0]
                chunks = []
            elif header is None:
                raise ValueError(f"FASTA sequence before first header in {path}")
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def load_label_rows(path: Path) -> tuple[dict[str, dict[str, str]], list[str]]:
    """Load the TSV and require unique accessions for deterministic joining."""

    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames or "accession" not in reader.fieldnames:
            raise ValueError(f"{path} must be a TSV containing an 'accession' column")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            accession = (row.get("accession") or "").strip()
            if not accession:
                continue
            if accession in rows:
                raise ValueError(f"Duplicate accession in labels TSV: {accession}")
            rows[accession] = {key: (value or "") for key, value in row.items()}
        return rows, list(reader.fieldnames)


def load_labeled_genomes(fasta_path: Path, labels_path: Path) -> tuple[list[GenomeRecord], list[str]]:
    """Join streamed FASTA records against labels, preserving deterministic FASTA order."""

    label_rows, fieldnames = load_label_rows(labels_path)
    records: list[GenomeRecord] = []
    seen: set[str] = set()
    for accession, raw_sequence in iter_fasta(fasta_path):
        row = label_rows.get(accession)
        if row is None:
            continue
        cleaned = normalize_iupac_to_n(raw_sequence)
        if not cleaned:
            continue
        records.append(
            GenomeRecord(
                index=len(records),
                accession=accession,
                sequence=cleaned,
                original_length=len("".join(raw_sequence.split())),
                labels=row,
            )
        )
        seen.add(accession)
    if not records:
        raise RuntimeError(f"No labels from {labels_path} matched records in {fasta_path}")
    missing = sorted(set(label_rows) - seen)
    return records, missing


def windows_for_record(record: GenomeRecord, *, window_bp: int, max_windows: int) -> list[WindowSpec]:
    return [
        WindowSpec(
            genome_index=record.index,
            accession=record.accession,
            window_index=window_index,
            start_bp=start,
            end_bp=end,
            sequence=window_sequence,
            genome_sha256=record.sequence_sha256,
        )
        for window_index, (start, end, window_sequence) in enumerate(
            deterministic_windows(record.sequence, window_bp=window_bp, max_windows=max_windows)
        )
    ]


def _safe_accession(accession: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", accession)


def cache_path(output_dir: Path, spec: WindowSpec) -> Path:
    identity = f"{spec.accession}|{spec.window_index}|{spec.start_bp}|{spec.end_bp}|{spec.sequence_sha256}"
    digest = hashlib.sha256(identity.encode("ascii")).hexdigest()[:16]
    return output_dir / "cache" / f"{_safe_accession(spec.accession)}__w{spec.window_index:02d}__{digest}.npz"


def _atomic_replace(path: Path, write: Any) -> None:
    """Write a file beside its destination, fsync it, then atomically replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    def write(handle: Any) -> None:
        handle.write((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))

    _atomic_replace(path, write)


def _atomic_write_tsv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise


def _validate_embedding(embedding: np.ndarray) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if array.shape != (EMBEDDING_DIM,):
        raise ValueError(f"Expected mean EVO2 embedding shape ({EMBEDDING_DIM},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("EVO2 embedding contains non-finite values")
    return array


def _padded_sae(sae: SaeTopK) -> SaeTopK:
    """Validate and pad/truncate an SAE response to exactly the endpoint's top-50."""

    fields = (sae.feature_ids, sae.max_act, sae.total_act, sae.active_in, sae.n_tokens)
    lengths = {np.asarray(field).reshape(-1).size for field in fields}
    if len(lengths) != 1:
        raise ValueError("SAE response fields have inconsistent lengths")
    count = lengths.pop()
    if count == 0:
        raise ValueError("SAE endpoint returned no features")
    keep = min(count, SAE_TOP_K)
    ids = np.full(SAE_TOP_K, -1, dtype=np.int32)
    numeric = [np.full(SAE_TOP_K, np.nan, dtype=np.float32) for _ in range(4)]
    ids[:keep] = np.asarray(sae.feature_ids).reshape(-1)[:keep].astype(np.int32)
    for target, source in zip(numeric, fields[1:]):
        target[:keep] = np.asarray(source).reshape(-1)[:keep].astype(np.float32)
    return SaeTopK(ids, numeric[0], numeric[1], numeric[2], numeric[3])


def _assert_full_window_was_processed(sae: SaeTopK, expected_tokens: int) -> None:
    """Fail closed when a UI/backend silently truncates a requested window.

    The existing SAE CSV reports ``n_tokens`` for each selected feature.  Evo2's
    nucleotide tokenizer is character-level for the normalised A/C/G/T/N input,
    so it must equal the submitted window length.  A mismatch most often means a
    server-side MAX_SEQ_LEN cap, which would make an advertised 8192bp window a
    misleading 2000bp prefix embedding.
    """

    reported = sae.n_tokens[np.isfinite(sae.n_tokens) & (sae.feature_ids >= 0)]
    if reported.size == 0:
        raise RuntimeError("SAE response has no usable n_tokens values")
    unique = np.unique(reported.astype(np.int64))
    if unique.size != 1 or int(unique[0]) != expected_tokens:
        raise RuntimeError(
            f"Server processed {unique.tolist()} tokens for a {expected_tokens}bp window; "
            "refusing a silently truncated embedding"
        )


def write_window_cache(path: Path, spec: WindowSpec, embedding: np.ndarray, sae: SaeTopK) -> None:
    """Atomically commit one fully-complete window; partial API work is never cached."""

    embedding = _validate_embedding(embedding)
    sae = _padded_sae(sae)

    def write(handle: Any) -> None:
        np.savez_compressed(
            handle,
            schema_version=np.asarray(SCHEMA_VERSION),
            accession=np.asarray(spec.accession),
            genome_index=np.asarray(spec.genome_index, dtype=np.int64),
            window_index=np.asarray(spec.window_index, dtype=np.int32),
            start_bp=np.asarray(spec.start_bp, dtype=np.int64),
            end_bp=np.asarray(spec.end_bp, dtype=np.int64),
            genome_sha256=np.asarray(spec.genome_sha256),
            window_sha256=np.asarray(spec.sequence_sha256),
            embedding=embedding,
            sae_feature_ids=sae.feature_ids,
            sae_max_act=sae.max_act,
            sae_total_act=sae.total_act,
            sae_active_in=sae.active_in,
            sae_n_tokens=sae.n_tokens,
        )

    _atomic_replace(path, write)


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    return np.asarray(data[key]).item()


def load_window_cache(path: Path, spec: WindowSpec) -> tuple[np.ndarray, SaeTopK] | None:
    """Return a cache entry only if its full identity and array schema are valid."""

    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            expected = {
                "schema_version": SCHEMA_VERSION,
                "accession": spec.accession,
                "genome_index": spec.genome_index,
                "window_index": spec.window_index,
                "start_bp": spec.start_bp,
                "end_bp": spec.end_bp,
                "genome_sha256": spec.genome_sha256,
                "window_sha256": spec.sequence_sha256,
            }
            if any(str(_scalar(data, key)) != str(value) for key, value in expected.items()):
                return None
            embedding = _validate_embedding(data["embedding"])
            sae = _padded_sae(
                SaeTopK(
                    data["sae_feature_ids"],
                    data["sae_max_act"],
                    data["sae_total_act"],
                    data["sae_active_in"],
                    data["sae_n_tokens"],
                )
            )
            return embedding, sae
    except (OSError, KeyError, ValueError, EOFError, zipfile.BadZipFile):
        return None


class GradioEndpointClient:
    """Minimal, retry-friendly client for a local Gradio queue endpoint."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7860",
        *,
        request_timeout_s: float = 90.0,
        job_timeout_s: float = 1800.0,
    ) -> None:
        root = base_url.rstrip("/")
        self.api_root = root if root.endswith("/gradio_api") else f"{root}/gradio_api"
        self.request_timeout_s = request_timeout_s
        self.job_timeout_s = job_timeout_s
        self.session = requests.Session()

    def _call(self, api_name: str, data: list[Any]) -> list[Any]:
        endpoint = api_name.lstrip("/")
        response = self.session.post(
            f"{self.api_root}/call/{endpoint}", json={"data": data}, timeout=self.request_timeout_s
        )
        response.raise_for_status()
        event_id = response.json()["event_id"]
        deadline = time.monotonic() + self.job_timeout_s
        while time.monotonic() < deadline:
            pending_event: str | None = None
            stream = self.session.get(
                f"{self.api_root}/call/{endpoint}/{event_id}", stream=True, timeout=self.request_timeout_s
            )
            stream.raise_for_status()
            for raw_line in stream.iter_lines(decode_unicode=True):
                line = raw_line.strip() if raw_line else ""
                if not line:
                    continue
                if line.startswith("event:"):
                    pending_event = line.partition(":")[2].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                payload = json.loads(line.partition(":")[2].strip())
                if pending_event == "error":
                    message = payload.get("error", payload) if isinstance(payload, dict) else payload
                    raise RuntimeError(f"Gradio {api_name} failed: {message}")
                if pending_event == "complete":
                    if not isinstance(payload, list):
                        raise RuntimeError(f"Gradio {api_name} returned non-list completion payload")
                    return payload
            # A proxy can cut an SSE response before completion; resume the same event.
            time.sleep(0.25)
        raise TimeoutError(f"Timed out after {self.job_timeout_s:.0f}s waiting for {api_name}")

    def _download(self, file_descriptor: Any) -> bytes:
        if not isinstance(file_descriptor, Mapping) or not file_descriptor.get("url"):
            raise RuntimeError(f"Gradio returned no downloadable file descriptor: {file_descriptor!r}")
        url = str(file_descriptor["url"])
        if not url.startswith(("http://", "https://")):
            url = urljoin(f"{self.api_root}/", url)
        response = self.session.get(url, timeout=self.request_timeout_s)
        response.raise_for_status()
        return response.content

    def embed(self, sequence: str) -> np.ndarray:
        result = self._call("/_embeddings_callback", [sequence])
        if len(result) < 3:
            raise RuntimeError(f"Unexpected embedding response length: {len(result)}")
        return _validate_embedding(np.load(io.BytesIO(self._download(result[2])), allow_pickle=False))

    @staticmethod
    def _csv_column(fieldnames: Sequence[str] | None, candidates: Sequence[str]) -> str:
        names = {name.lower().strip(): name for name in fieldnames or []}
        for candidate in candidates:
            if candidate.lower() in names:
                return names[candidate.lower()]
        raise RuntimeError(f"SAE CSV missing expected column; wanted one of {list(candidates)}, got {fieldnames}")

    def sae_top50(self, sequence: str) -> SaeTopK:
        result = self._call("/_features_callback", [sequence, SAE_TOP_K])
        if len(result) < 5:
            raise RuntimeError(f"Unexpected SAE response length: {len(result)}")
        markdown = result[0] if result else ""
        if isinstance(markdown, str) and "failed" in markdown.lower():
            raise RuntimeError(f"SAE endpoint reported failure: {markdown[:500]}")
        payload = self._download(result[4]).decode("utf-8")
        reader = csv.DictReader(io.StringIO(payload))
        id_col = self._csv_column(reader.fieldnames, ("feat_id", "feature_id", "id"))
        max_col = self._csv_column(reader.fieldnames, ("max_act", "max_activation"))
        total_col = self._csv_column(reader.fieldnames, ("total_act", "total_activation"))
        active_col = self._csv_column(reader.fieldnames, ("active_in", "n_active", "active_count"))
        token_col = self._csv_column(reader.fieldnames, ("n_tokens", "tokens", "sequence_length"))
        rank_col = next((name for name in (reader.fieldnames or []) if name.lower().strip() == "rank"), None)
        parsed: list[tuple[int, int, float, float, float, float]] = []
        for fallback_rank, row in enumerate(reader, start=1):
            try:
                rank = int(float(row[rank_col])) if rank_col else fallback_rank
                parsed.append(
                    (
                        rank,
                        int(float(row[id_col])),
                        float(row[max_col]),
                        float(row[total_col]),
                        float(row[active_col]),
                        float(row[token_col]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(f"Malformed SAE CSV row: {row!r}") from exc
        if not parsed:
            raise RuntimeError("SAE endpoint returned an empty CSV")
        parsed.sort(key=lambda row: row[0])
        parsed = parsed[:SAE_TOP_K]
        return _padded_sae(
            SaeTopK(
                np.asarray([row[1] for row in parsed], dtype=np.int32),
                np.asarray([row[2] for row in parsed], dtype=np.float32),
                np.asarray([row[3] for row in parsed], dtype=np.float32),
                np.asarray([row[4] for row in parsed], dtype=np.float32),
                np.asarray([row[5] for row in parsed], dtype=np.float32),
            )
        )


def _window_result(
    output_dir: Path,
    spec: WindowSpec,
    client: Evo2SaeClient,
    *,
    retries: int,
    require_full_window: bool,
) -> tuple[np.ndarray, SaeTopK, str]:
    path = cache_path(output_dir, spec)
    cached = load_window_cache(path, spec)
    if cached is not None:
        if require_full_window:
            try:
                _assert_full_window_was_processed(cached[1], len(spec.sequence))
            except RuntimeError:
                # A cache generated against a truncating legacy backend is not
                # valid evidence for this study. Remove it before retrying.
                path.unlink(missing_ok=True)
            else:
                return cached[0], cached[1], "cached"
        else:
            return cached[0], cached[1], "cached"

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            embedding = _validate_embedding(client.embed(spec.sequence))
            sae = _padded_sae(client.sae_top50(spec.sequence))
            if require_full_window:
                _assert_full_window_was_processed(sae, len(spec.sequence))
            write_window_cache(path, spec, embedding, sae)
            return embedding, sae, "api"
        except Exception as exc:  # Endpoint failures must remain retriable, not cached.
            last_error = exc
            if attempt < retries:
                delay_s = min(60.0, 2.0**attempt)
                print(
                    f"  retry {attempt + 1}/{retries} for {spec.accession} window {spec.window_index} "
                    f"after {delay_s:.0f}s: {exc}",
                    flush=True,
                )
                time.sleep(delay_s)
    assert last_error is not None
    raise last_error


def _new_atomic_memmap(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any], fill: Any) -> tuple[Path, np.memmap]:
    """Create a valid temporary .npy memmap so a partial consolidation never replaces output."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=shape)
    matrix[...] = fill
    return temporary, matrix


def _consolidate(
    output_dir: Path,
    records: Sequence[GenomeRecord],
    specs_by_genome: Sequence[Sequence[WindowSpec]],
    states: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    max_windows: int,
    matrix_dtype: str,
    missing_labels: Sequence[str],
    run_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialise cache entries into fixed arrays and descriptive TSV metadata."""

    matrix_dir = output_dir / "matrices"
    n_genomes = len(records)
    dtype = np.dtype(matrix_dtype)
    if dtype.kind != "f":
        raise ValueError("matrix_dtype must be a floating-point numpy dtype")
    targets = {
        "evo2_window_embeddings.npy": ((n_genomes, max_windows, EMBEDDING_DIM), dtype, np.nan),
        "evo2_genome_mean.npy": ((n_genomes, EMBEDDING_DIM), dtype, np.nan),
        "sae_top50_feature_ids.npy": ((n_genomes, max_windows, SAE_TOP_K), np.int32, -1),
        "sae_top50_max_act.npy": ((n_genomes, max_windows, SAE_TOP_K), np.float32, np.nan),
        "sae_top50_total_act.npy": ((n_genomes, max_windows, SAE_TOP_K), np.float32, np.nan),
        "sae_top50_active_in.npy": ((n_genomes, max_windows, SAE_TOP_K), np.float32, np.nan),
        "sae_top50_n_tokens.npy": ((n_genomes, max_windows, SAE_TOP_K), np.float32, np.nan),
        "window_mask.npy": ((n_genomes, max_windows), np.bool_, False),
    }
    temporary: dict[str, Path] = {}
    matrices: dict[str, np.memmap] = {}
    try:
        for name, (shape, target_dtype, fill) in targets.items():
            temp_path, matrix = _new_atomic_memmap(matrix_dir / name, shape, target_dtype, fill)
            temporary[name] = temp_path
            matrices[name] = matrix

        for record, specs in zip(records, specs_by_genome):
            vectors: list[np.ndarray] = []
            for spec in specs:
                cache = load_window_cache(cache_path(output_dir, spec), spec)
                if cache is None:
                    continue
                embedding, sae = cache
                slot = spec.window_index
                matrices["evo2_window_embeddings.npy"][record.index, slot] = embedding.astype(dtype, copy=False)
                matrices["sae_top50_feature_ids.npy"][record.index, slot] = sae.feature_ids
                matrices["sae_top50_max_act.npy"][record.index, slot] = sae.max_act
                matrices["sae_top50_total_act.npy"][record.index, slot] = sae.total_act
                matrices["sae_top50_active_in.npy"][record.index, slot] = sae.active_in
                matrices["sae_top50_n_tokens.npy"][record.index, slot] = sae.n_tokens
                matrices["window_mask.npy"][record.index, slot] = True
                vectors.append(embedding)
            if vectors:
                matrices["evo2_genome_mean.npy"][record.index] = np.mean(vectors, axis=0, dtype=np.float32).astype(
                    dtype, copy=False
                )

        for matrix in matrices.values():
            matrix.flush()
        matrices.clear()  # release mmap descriptors before replacement
        for name, temporary_path in temporary.items():
            os.replace(temporary_path, matrix_dir / name)
        temporary.clear()
    finally:
        matrices.clear()
        for temporary_path in temporary.values():
            temporary_path.unlink(missing_ok=True)

    label_fields = sorted({field for record in records for field in record.labels})
    genome_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    completed_windows = 0
    requested_windows = 0
    failed_windows = 0
    for record, specs in zip(records, specs_by_genome):
        completed_for_genome = 0
        for spec in specs:
            requested_windows += 1
            state = dict(
                states.get(
                    (spec.genome_index, spec.window_index),
                    {"status": "pending", "source": "none", "error": "not attempted"},
                )
            )
            if state["status"] == "complete":
                completed_windows += 1
                completed_for_genome += 1
            elif state["status"] == "failed":
                failed_windows += 1
            window_rows.append(
                {
                    "genome_index": record.index,
                    "accession": record.accession,
                    "window_index": spec.window_index,
                    "start_bp": spec.start_bp,
                    "end_bp": spec.end_bp,
                    "window_length_bp": len(spec.sequence),
                    "genome_sha256": spec.genome_sha256,
                    "window_sha256": spec.sequence_sha256,
                    **state,
                }
            )
        genome_rows.append(
            {
                "genome_index": record.index,
                "accession": record.accession,
                "original_length_bp": record.original_length,
                "cleaned_length_bp": len(record.sequence),
                "genome_sha256": record.sequence_sha256,
                "n_windows_requested": len(specs),
                "n_windows_completed": completed_for_genome,
                "status": "complete" if completed_for_genome == len(specs) else "partial",
                **{f"label_{field}": record.labels.get(field, "") for field in label_fields},
            }
        )

    _atomic_write_tsv(
        matrix_dir / "genome_meta.tsv",
        [
            "genome_index",
            "accession",
            "original_length_bp",
            "cleaned_length_bp",
            "genome_sha256",
            "n_windows_requested",
            "n_windows_completed",
            "status",
            *[f"label_{field}" for field in label_fields],
        ],
        genome_rows,
    )
    _atomic_write_tsv(
        matrix_dir / "window_meta.tsv",
        [
            "genome_index",
            "accession",
            "window_index",
            "start_bp",
            "end_bp",
            "window_length_bp",
            "genome_sha256",
            "window_sha256",
            "status",
            "source",
            "error",
        ],
        window_rows,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_unix": time.time(),
        "n_genomes": n_genomes,
        "requested_windows": requested_windows,
        "completed_windows": completed_windows,
        "failed_windows": failed_windows,
        "max_windows": max_windows,
        "embedding_dim": EMBEDDING_DIM,
        "sae_top_k": SAE_TOP_K,
        "matrix_dtype": dtype.name,
        "missing_label_accessions": len(missing_labels),
        "run_options": dict(run_options),
    }
    _atomic_write_json(matrix_dir / "manifest.json", summary)
    return summary


def run_bulk(
    records: Sequence[GenomeRecord],
    *,
    output_dir: Path,
    client: Evo2SaeClient,
    window_bp: int = 8192,
    max_windows: int = 8,
    retries: int = 2,
    matrix_dtype: str = "float32",
    require_full_window: bool = True,
    missing_labels: Sequence[str] = (),
    run_options: Mapping[str, Any] | None = None,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """Execute or resume all windows, then atomically refresh fixed matrices."""

    output_dir.mkdir(parents=True, exist_ok=True)
    specs_by_genome = [windows_for_record(record, window_bp=window_bp, max_windows=max_windows) for record in records]
    states: dict[tuple[int, int], dict[str, Any]] = {}
    total = sum(len(specs) for specs in specs_by_genome)
    completed = 0
    attempted = 0
    started = time.monotonic()
    for record, specs in zip(records, specs_by_genome):
        for spec in specs:
            key = (spec.genome_index, spec.window_index)
            t0 = time.monotonic()
            attempted += 1
            try:
                _, _, source = _window_result(
                    output_dir,
                    spec,
                    client,
                    retries=retries,
                    require_full_window=require_full_window,
                )
                state = {"status": "complete", "source": source, "error": ""}
            except Exception as exc:
                state = {"status": "failed", "source": "none", "error": f"{type(exc).__name__}: {exc}"}
                print(
                    f"[{attempted}/{total}] {record.accession} window {spec.window_index}: FAILED {state['error']}",
                    flush=True,
                )
                states[key] = state
                if stop_on_error:
                    _consolidate(
                        output_dir,
                        records,
                        specs_by_genome,
                        states,
                        max_windows=max_windows,
                        matrix_dtype=matrix_dtype,
                        missing_labels=missing_labels,
                        run_options=run_options or {},
                    )
                    raise
                continue
            completed += 1
            states[key] = state
            elapsed = time.monotonic() - started
            print(
                f"[{attempted}/{total}] {record.accession} window {spec.window_index} "
                f"{source} {time.monotonic() - t0:.1f}s elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    return _consolidate(
        output_dir,
        records,
        specs_by_genome,
        states,
        max_windows=max_windows,
        matrix_dtype=matrix_dtype,
        missing_labels=missing_labels,
        run_options=run_options or {},
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA, help="Reference viral FASTA (.fna or .fna.gz)")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="TSV containing accession and taxonomic labels")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root; cache and matrices are created below it")
    parser.add_argument("--endpoint", default="http://127.0.0.1:7860", help="Local EVO2 Gradio host or /gradio_api URL")
    parser.add_argument("--window-bp", type=int, default=8192)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N FASTA-ordered labelled genomes")
    parser.add_argument("--canary", action="store_true", help="Process exactly the first labelled genome (endpoint smoke test)")
    parser.add_argument("--retries", type=int, default=2, help="Retries after an endpoint error, per window")
    parser.add_argument("--request-timeout-s", type=float, default=90.0)
    parser.add_argument("--job-timeout-s", type=float, default=1800.0)
    parser.add_argument("--matrix-dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument(
        "--allow-server-truncation",
        action="store_true",
        help="Permit a backend to process fewer tokens than the requested window (not valid for 8192bp study)",
    )
    parser.add_argument("--stop-on-error", action="store_true", help="Write current matrices then exit on first failed window")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    records, missing_labels = load_labeled_genomes(args.fasta, args.labels)
    if args.canary:
        records = records[:1]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        raise SystemExit("No genomes selected")
    print(
        f"Loaded {len(records):,} labelled genomes from {args.fasta}; "
        f"{len(missing_labels):,} labels did not occur in FASTA. "
        f"Each genome -> <= {args.max_windows} x {args.window_bp}bp windows.",
        flush=True,
    )
    client = GradioEndpointClient(
        args.endpoint, request_timeout_s=args.request_timeout_s, job_timeout_s=args.job_timeout_s
    )
    options = {
        "fasta": str(args.fasta),
        "labels": str(args.labels),
        "endpoint": args.endpoint,
        "window_bp": args.window_bp,
        "max_windows": args.max_windows,
        "canary": args.canary,
        "limit": args.limit,
        "retries": args.retries,
        "require_full_window": not args.allow_server_truncation,
    }
    summary = run_bulk(
        records,
        output_dir=args.out,
        client=client,
        window_bp=args.window_bp,
        max_windows=args.max_windows,
        retries=args.retries,
        matrix_dtype=args.matrix_dtype,
        require_full_window=not args.allow_server_truncation,
        missing_labels=missing_labels,
        run_options=options,
        stop_on_error=args.stop_on_error,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
