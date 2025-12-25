import gzip
import json
import os
from pathlib import Path

import pytest

from rag.ingest.hf_slice import build_hf_slice, parse_data_files


def test_parse_data_files_empty_string_returns_none():
    assert parse_data_files("") is None
    assert parse_data_files("   ") is None


def test_parse_data_files_filters_empty_entries_and_splits():
    assert parse_data_files("a.parquet,, ,b.parquet") == ["a.parquet", "b.parquet"]
    assert parse_data_files(["", "foo", " ", "bar"]) == ["foo", "bar"]


def test_build_hf_slice_raises_clear_error_on_missing_cache(monkeypatch, tmp_path: Path):
    def _fake_load_dataset(*args, **kwargs):
        raise ValueError("Could not find config 'default' at cached or remote")

    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    monkeypatch.setenv("LOG_FILE", str(tmp_path / "log.txt"))
    monkeypatch.setattr("rag.ingest.hf_slice.load_dataset", _fake_load_dataset)

    with pytest.raises(RuntimeError) as err:
        build_hf_slice(
            dataset_id="irlspbru/RusLawOD",
            split="train",
            n=1,
            seed=42,
            output_dir=tmp_path,
            strategy="stream",
        )
    assert "HuggingFace Hub недоступен" in str(err.value)


@pytest.mark.network
@pytest.mark.slow
def test_build_hf_slice_creates_gzip_and_manifest(tmp_path: Path):
    import logging

    logging.getLogger("datasets").setLevel(logging.INFO)
    logging.getLogger("huggingface_hub").setLevel(logging.INFO)
    logging.getLogger("fsspec").setLevel(logging.INFO)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    os.environ.setdefault("LOG_LEVEL", "INFO")

    pytest.importorskip("datasets")

    if os.environ.get("HF_DATASETS_OFFLINE") == "1":
        pytest.skip("HF datasets offline")

    output_dir = tmp_path / "data" / "raw"
    try:
        print("START: build_hf_slice")
        slice_path, manifest_path = build_hf_slice(
            dataset_id="irlspbru/RusLawOD",
            split="train[:20]",
            n=2,
            seed=42,
            output_dir=output_dir,
            strategy="download",
            data_files=None,
            filters={"statusIPS": ["Действует"]},
            max_retries=3,
        )
        print("DONE: build_hf_slice", slice_path, manifest_path)
    except Exception as exc:  # pragma: no cover - сетевые ошибки приводят к skip
        # Скипаем только сетевые/transport ошибки, но не валидационные
        if any(word in str(exc) for word in ["Connection", "Timeout", "HTTP"]):
            pytest.skip(f"HF slice unavailable: {exc}")
        raise

    assert slice_path.exists()
    assert manifest_path.exists()

    with gzip.open(slice_path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    assert len(rows) == 2
    first = rows[0]
    assert "doc_id" in first and "text" in first
    assert first["metadata"]["source"] == "RusLawOD"

    manifest = json.loads(manifest_path.read_text())
    assert manifest["dataset_id"] == "irlspbru/RusLawOD"
    assert manifest["split"] == "train[:20]"
    assert manifest["seed"] == 42
    assert manifest["n"] == 2
    assert manifest["strategy"] == "download"
    assert manifest["filters"] == {"statusIPS": ["Действует"]}
    assert manifest["data_files"] is None
