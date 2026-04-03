"""Tests for core module."""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from kozzle_word_grouper.core import WordGrouperPipeline
from kozzle_word_grouper.exceptions import WordGrouperError


@pytest.fixture
def mock_supabase_client(sample_words: list[str]) -> Mock:
    """Mock Supabase client."""
    client = Mock()
    client.fetch_words.return_value = sample_words
    return client


@pytest.fixture
def mock_embedding_generator(sample_embeddings: np.ndarray) -> Mock:
    """Mock embedding generator."""
    generator = Mock()
    generator.generate_embeddings.return_value = sample_embeddings
    return generator


@pytest.fixture
def mock_clusterer() -> Mock:
    """Mock word clusterer."""
    clusterer = Mock()
    clusterer.fit_predict.return_value = np.array([0, 0, 0, 1, 1, 1])
    clusterer.get_cluster_info.return_value = {
        0: {
            "cluster_id": 0,
            "label": "cluster_0",
            "words": ["dog", "cat", "bird"],
            "word_count": 3,
        },
        1: {
            "cluster_id": 1,
            "label": "cluster_1",
            "words": ["car", "truck", "bus"],
            "word_count": 3,
        },
    }
    clusterer.calculate_cluster_quality.return_value = {
        "silhouette_score": 0.75,
        "n_clusters": 2,
        "noise_ratio": 0.0,
    }
    return clusterer


def test_pipeline_initialization() -> None:
    """Test pipeline initialization."""
    pipeline = WordGrouperPipeline(
        model_name="all-MiniLM-L6-v2",
        min_cluster_size=10,
        output_dir="./output",
    )

    assert pipeline.model_name == "all-MiniLM-L6-v2"
    assert pipeline.min_cluster_size == 10
    assert pipeline.output_dir == Path("./output")


def test_pipeline_fetch_words(
    mock_supabase_client: Mock, sample_words: list[str]
) -> None:
    """Test word fetching."""
    pipeline = WordGrouperPipeline()
    pipeline.supabase_client = mock_supabase_client

    words = pipeline.fetch_words(table_name="words", word_column="word")

    assert len(words) == len(sample_words)
    assert words == sample_words
    mock_supabase_client.fetch_words.assert_called_once_with(
        table_name="words",
        word_column="word",
    )


def test_pipeline_generate_embeddings(
    mock_embedding_generator: Mock,
    sample_words: list[str],
    sample_embeddings: np.ndarray,
) -> None:
    """Test embedding generation."""
    pipeline = WordGrouperPipeline()
    pipeline.embedding_generator = mock_embedding_generator

    embeddings = pipeline.generate_embeddings(sample_words, show_progress=False)

    assert embeddings.shape == sample_embeddings.shape
    mock_embedding_generator.generate_embeddings.assert_called_once()


def test_pipeline_cluster_words(
    mock_clusterer: Mock,
    sample_words: list[str],
    sample_embeddings: np.ndarray,
) -> None:
    """Test word clustering."""
    pipeline = WordGrouperPipeline()
    pipeline.clusterer = mock_clusterer

    cluster_info, quality_metrics = pipeline.cluster_words(
        sample_words, sample_embeddings
    )

    assert "silhouette_score" in quality_metrics
    assert "n_clusters" in quality_metrics
    mock_clusterer.fit_predict.assert_called_once()


@patch("kozzle_word_grouper.core.SupabaseClient")
@patch("kozzle_word_grouper.core.EmbeddingGenerator")
@patch("kozzle_word_grouper.core.WordClusterer")
@patch("kozzle_word_grouper.core.WordGroupExporter")
def test_pipeline_run(
    mock_exporter_class: Mock,
    mock_clusterer_class: Mock,
    mock_embedding_gen_class: Mock,
    mock_supabase_class: Mock,
    sample_words: list[str],
    sample_embeddings: np.ndarray,
    temp_output_dir: Path,
) -> None:
    """Test full pipeline execution."""
    # Setup mocks
    mock_supabase = MagicMock()
    mock_supabase.fetch_words.return_value = sample_words
    mock_supabase_class.return_value = mock_supabase

    mock_embedding_gen = MagicMock()
    mock_embedding_gen.generate_embeddings.return_value = sample_embeddings
    mock_embedding_gen_class.return_value = mock_embedding_gen

    mock_clusterer = MagicMock()
    mock_clusterer.fit_predict.return_value = np.array([0] * len(sample_words))
    mock_clusterer.get_cluster_info.return_value = {
        0: {"cluster_id": 0, "words": sample_words, "word_count": len(sample_words)}
    }
    mock_clusterer.calculate_cluster_quality.return_value = {
        "silhouette_score": 0.5,
        "n_clusters": 1,
        "noise_ratio": 0.0,
    }
    mock_clusterer_class.return_value = mock_clusterer

    mock_exporter = MagicMock()
    mock_exporter.export_all.return_value = [temp_output_dir / "test.json"]
    mock_exporter_class.return_value = mock_exporter

    # Run pipeline
    pipeline = WordGrouperPipeline(output_dir=temp_output_dir)
    result = pipeline.run(show_progress=False)

    assert isinstance(result, dict)
    mock_supabase.fetch_words.assert_called_once()
    mock_embedding_gen.generate_embeddings.assert_called_once()
    mock_clusterer.fit_predict.assert_called_once()
    mock_exporter.export_all.assert_called_once()


def test_pipeline_run_no_words(temp_output_dir: Path) -> None:
    """Test pipeline with no words."""
    with patch("kozzle_word_grouper.core.SupabaseClient") as mock_supabase_class:
        mock_supabase = MagicMock()
        mock_supabase.fetch_words.return_value = []
        mock_supabase_class.return_value = mock_supabase

        pipeline = WordGrouperPipeline(output_dir=temp_output_dir)

        with pytest.raises(WordGrouperError):
            pipeline.run()
