"""Tests for clustering module."""

import numpy as np
from numpy.typing import NDArray

from kozzle_word_grouper.clustering import WordClusterer


def test_clusterer_initialization() -> None:
    """Test clusterer initialization."""
    clusterer = WordClusterer(min_cluster_size=10)

    assert clusterer.min_cluster_size == 10
    assert clusterer.min_samples == 10
    assert clusterer.metric == "cosine"


def test_clusterer_initialization_with_min_samples() -> None:
    """Test clusterer initialization with custom min_samples."""
    clusterer = WordClusterer(min_cluster_size=10, min_samples=5)

    assert clusterer.min_cluster_size == 10
    assert clusterer.min_samples == 5


def test_group_words_by_cluster(sample_words: list[str]) -> None:
    """Test grouping words by cluster labels."""
    clusterer = WordClusterer()
    labels = np.array([0, 0, 0, 1, 1, 1])
    words = sample_words[:6]

    groups = clusterer.group_words_by_cluster(words, labels)

    assert len(groups) == 2
    assert 0 in groups
    assert 1 in groups
    assert len(groups[0]) == 3
    assert len(groups[1]) == 3


def test_group_words_by_cluster_with_noise(sample_words: list[str]) -> None:
    """Test grouping words with noise points."""
    clusterer = WordClusterer()
    labels = np.array([0, 0, 0, -1, 1, 1])
    words = sample_words[:6]

    groups = clusterer.group_words_by_cluster(words, labels)

    assert len(groups) == 3
    assert -1 in groups
    assert len(groups[-1]) == 1


def test_fit_predict_small_dataset(sample_embeddings: NDArray[np.floating]) -> None:
    """Test clustering on small dataset."""
    clusterer = WordClusterer(min_cluster_size=2)
    labels = clusterer.fit_predict(sample_embeddings)

    assert labels.shape[0] == sample_embeddings.shape[0]
    assert labels.dtype == np.integer or np.issubdtype(labels.dtype, np.integer)


def test_fit_predict_empty_embeddings() -> None:
    """Test clustering with empty embeddings."""
    clusterer = WordClusterer()
    labels = clusterer.fit_predict(np.array([]))

    assert len(labels) == 0


def test_get_cluster_info(
    sample_words: list[str],
    sample_embeddings: NDArray[np.floating],
    sample_cluster_labels: NDArray[np.integer],
) -> None:
    """Test getting cluster information."""
    clusterer = WordClusterer(min_cluster_size=5)
    cluster_info = clusterer.get_cluster_info(
        sample_words, sample_cluster_labels, sample_embeddings
    )

    assert isinstance(cluster_info, dict)
    assert len(cluster_info) > 0

    # Check structure of cluster info
    for cluster_id, info in cluster_info.items():
        assert "cluster_id" in info
        assert "words" in info
        assert info["cluster_id"] == cluster_id


def test_calculate_cluster_quality(
    sample_embeddings: NDArray[np.floating],
    sample_cluster_labels: NDArray[np.integer],
) -> None:
    """Test calculating cluster quality metrics."""
    clusterer = WordClusterer()
    metrics = clusterer.calculate_cluster_quality(
        sample_embeddings, sample_cluster_labels
    )

    assert "silhouette_score" in metrics
    assert "n_clusters" in metrics
    assert "noise_ratio" in metrics
    assert isinstance(metrics["silhouette_score"], float)
    assert isinstance(metrics["n_clusters"], int)
    assert isinstance(metrics["noise_ratio"], float)
    assert 0.0 <= metrics["noise_ratio"] <= 1.0
