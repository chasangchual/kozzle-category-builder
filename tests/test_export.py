"""Tests for export module."""

import json
from pathlib import Path

import numpy as np

from kozzle_word_grouper.export import WordGroupExporter


def test_exporter_initialization(temp_output_dir: Path) -> None:
    """Test exporter initialization."""
    exporter = WordGroupExporter(output_dir=temp_output_dir)

    assert exporter.output_dir == temp_output_dir
    assert temp_output_dir.exists()


def test_export_to_json(
    temp_output_dir: Path,
    sample_cluster_labels: np.ndarray,
) -> None:
    """Test exporting to JSON format."""
    exporter = WordGroupExporter(output_dir=temp_output_dir)

    cluster_info = {
        0: {
            "cluster_id": 0,
            "label": "animals",
            "word_count": 3,
            "words": ["dog", "cat", "bird"],
        },
        1: {
            "cluster_id": 1,
            "label": "vehicles",
            "word_count": 3,
            "words": ["car", "truck", "bus"],
        },
    }

    output_path = exporter.export_to_json(cluster_info, filename="test.json")

    assert output_path.exists()
    assert output_path.name == "test.json"

    # Verify content
    with open(output_path) as f:
        data = json.load(f)

    assert "cluster_0" in data
    assert "cluster_1" in data
    assert data["cluster_0"]["word_count"] == 3


def test_export_to_json_without_words(temp_output_dir: Path) -> None:
    """Test exporting to JSON without word lists."""
    exporter = WordGroupExporter(output_dir=temp_output_dir)

    cluster_info = {
        0: {
            "cluster_id": 0,
            "label": "animals",
            "word_count": 3,
        },
    }

    output_path = exporter.export_to_json(
        cluster_info,
        filename="test_no_words.json",
        include_words=False,
    )

    with open(output_path) as f:
        data = json.load(f)

    assert "words" not in data["cluster_0"]


def test_export_to_csv(temp_output_dir: Path) -> None:
    """Test exporting to CSV format."""
    exporter = WordGroupExporter(output_dir=temp_output_dir)

    cluster_info = {
        0: {
            "cluster_id": 0,
            "label": "animals",
            "words": ["dog", "cat", "bird"],
        },
        1: {
            "cluster_id": 1,
            "label": "vehicles",
            "words": ["car", "truck"],
        },
    }

    output_path = exporter.export_to_csv(cluster_info, filename="test.csv")

    assert output_path.exists()
    assert output_path.name == "test.csv"

    # Verify content
    import pandas as pd

    df = pd.read_csv(output_path)

    assert len(df) == 5
    assert "word" in df.columns
    assert "cluster_id" in df.columns
    assert "cluster_label" in df.columns


def test_export_summary(temp_output_dir: Path) -> None:
    """Test exporting cluster summary."""
    exporter = WordGroupExporter(output_dir=temp_output_dir)

    cluster_info = {
        0: {
            "cluster_id": 0,
            "label": "animals",
            "word_count": 5,
            "representative_words": ["dog", "cat"],
        },
        1: {
            "cluster_id": 1,
            "label": "vehicles",
            "word_count": 3,
            "representative_words": ["car", "bus"],
        },
    }

    quality_metrics = {
        "silhouette_score": 0.75,
        "n_clusters": 2,
        "noise_ratio": 0.05,
    }

    output_path = exporter.export_summary(
        cluster_info,
        quality_metrics,
        filename="test_summary.txt",
    )

    assert output_path.exists()

    # Verify content
    content = output_path.read_text()

    assert "Word Cluster Summary" in content
    assert "Silhouette score" in content
    assert "animals" in content
    assert "vehicles" in content


def test_export_all(temp_output_dir: Path) -> None:
    """Test exporting all formats."""
    exporter = WordGroupExporter(output_dir=temp_output_dir)

    cluster_info = {
        0: {
            "cluster_id": 0,
            "label": "test",
            "word_count": 2,
            "words": ["word1", "word2"],
        },
    }

    exported_files = exporter.export_all(
        cluster_info,
        output_format=["json", "csv", "summary"],
    )

    assert len(exported_files) == 3
    for path in exported_files:
        assert path.exists()


def test_export_all_default_format(temp_output_dir: Path) -> None:
    """Test exporting with default format."""
    exporter = WordGroupExporter(output_dir=temp_output_dir)

    cluster_info = {
        0: {
            "cluster_id": 0,
            "label": "test",
            "word_count": 1,
            "words": ["word1"],
        },
    }

    exported_files = exporter.export_all(cluster_info)

    assert len(exported_files) == 3
