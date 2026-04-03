"""Semantic clustering using HDBSCAN."""

from collections import defaultdict
from typing import Any, cast

import hdbscan
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_distances

from kozzle_word_grouper.exceptions import ClusteringError
from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class WordClusterer:
    """Cluster words by semantic similarity using HDBSCAN."""

    def __init__(
        self,
        min_cluster_size: int = 5,
        min_samples: int | None = None,
        metric: str = "cosine",
        cluster_selection_method: str = "eom",
    ) -> None:
        """Initialize word clusterer.

        Args:
            min_cluster_size: Minimum number of words in a cluster.
            min_samples: Minimum samples parameter for HDBSCAN
                (default: same as min_cluster_size).
            metric: Distance metric (default: cosine for embeddings).
            cluster_selection_method: Method for selecting clusters (eom or leaf).
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method

        logger.info(
            f"Initialized clusterer with min_cluster_size={min_cluster_size}, "
            f"min_samples={self.min_samples}"
        )

    def fit_predict(
        self,
        embeddings: NDArray[np.floating[Any]],
    ) -> NDArray[np.integer[Any]]:
        """Cluster embeddings using HDBSCAN.

        Args:
            embeddings: Array of embeddings with shape (n_samples, embedding_dim).

        Returns:
            Array of cluster labels (-1 for noise/outliers).

        Raises:
            ClusteringError: If clustering fails.
        """
        if len(embeddings) == 0:
            logger.warning("Empty embeddings array provided")
            return np.array([])

        try:
            logger.info(f"Clustering {len(embeddings)} embeddings with HDBSCAN")

            # HDBSCAN doesn't support cosine metric directly,
            # so use precomputed distances
            if self.metric == "cosine":
                distance_matrix = cosine_distances(embeddings)
                # HDBSCAN expects float64 for precomputed distances
                distance_matrix = distance_matrix.astype(np.float64)
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric="precomputed",
                    cluster_selection_method=self.cluster_selection_method,
                    prediction_data=True,
                )
                labels = clusterer.fit_predict(distance_matrix)
            else:
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=self.min_cluster_size,
                    min_samples=self.min_samples,
                    metric=self.metric,
                    cluster_selection_method=self.cluster_selection_method,
                    prediction_data=True,
                )
                labels = clusterer.fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)

            logger.info(f"Found {n_clusters} clusters, {n_noise} noise points")

            return cast(NDArray[np.integer[Any]], labels)

        except Exception as e:
            raise ClusteringError(f"Failed to cluster embeddings: {e}") from e

    def group_words_by_cluster(
        self,
        words: list[str],
        labels: NDArray[np.integer[Any]],
    ) -> dict[int, list[str]]:
        """Group words by cluster labels.

        Args:
            words: List of words corresponding to the labels.
            labels: Cluster labels from fit_predict.

        Returns:
            Dictionary mapping cluster ID to list of words.
            Cluster -1 contains noise/unclustered words.
        """
        groups: dict[int, list[str]] = defaultdict(list)

        for word, label in zip(words, labels):
            groups[int(label)].append(word)

        logger.info(f"Grouped {len(words)} words into {len(groups)} clusters")

        return dict(groups)

    def get_cluster_info_korean(
        self,
        words: list[KoreanWord],
        labels: NDArray[np.integer[Any]],
        korean_labels: dict[int, str],
        embeddings: NDArray[np.floating[Any]],
    ) -> dict[int, dict[str, Any]]:
        """Get detailed information about each cluster for Korean words.

        Args:
            words: List of KoreanWord objects.
            labels: Cluster labels.
            korean_labels: Dict mapping cluster_id to Korean label.
            embeddings: Word embeddings.

        Returns:
            Dictionary with cluster information including:
            - cluster_id
            - label: Korean category name
            - words: list of {"public_id": ..., "lemma": ...}
            - word_count
            - representative_words: representative Korean words
        """
        cluster_info: dict[int, dict[str, Any]] = {}

        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue

            # Get words in this cluster
            indices = np.where(labels == cluster_id)[0]
            cluster_words = [words[i] for i in indices]

            # Format words for output
            words_data = [
                {"public_id": w.public_id, "lemma": w.lemma} for w in cluster_words
            ]

            # Get representative words
            cluster_embeddings = embeddings[indices]
            centroid = np.mean(cluster_embeddings, axis=0)
            distances = cosine_distances(
                cluster_embeddings, centroid.reshape(1, -1)
            ).flatten()
            representative_indices = np.argsort(distances)[: min(5, len(indices))]
            representative_words = [
                cluster_words[i].lemma for i in representative_indices
            ]

            cluster_info[cluster_id] = {
                "cluster_id": int(cluster_id),
                "label": korean_labels.get(int(cluster_id), f"클러스터_{cluster_id}"),
                "words": words_data,
                "word_count": len(cluster_words),
                "representative_words": representative_words,
            }

        return cluster_info

    def get_cluster_info(
        self,
        words: list[str],
        labels: NDArray[np.integer[Any]],
        embeddings: NDArray[np.floating[Any]],
    ) -> dict[int, dict[str, Any]]:
        """Get detailed information about each cluster.

        Args:
            words: List of words.
            labels: Cluster labels.
            embeddings: Word embeddings.

        Returns:
            Dictionary with cluster information including:
            - cluster_id
            - words: list of words in cluster
            - centroid: average embedding
            - representative_words: words closest to centroid
        """
        cluster_info: dict[int, dict[str, Any]] = {}

        groups = self.group_words_by_cluster(words, labels)

        for cluster_id, cluster_words in groups.items():
            # Get indices of words in this cluster
            indices = np.where(labels == cluster_id)[0]

            if cluster_id == -1:
                # Noise cluster
                cluster_info[cluster_id] = {
                    "cluster_id": cluster_id,
                    "label": "noise",
                    "word_count": len(cluster_words),
                    "words": cluster_words,
                }
                continue

            # Calculate centroid
            cluster_embeddings = embeddings[indices]
            centroid = np.mean(cluster_embeddings, axis=0)

            # Find representative words (closest to centroid)
            distances = cosine_distances(
                cluster_embeddings, centroid.reshape(1, -1)
            ).flatten()
            representative_indices = np.argsort(distances)[: min(5, len(indices))]
            representative_words = [cluster_words[i] for i in representative_indices]

            cluster_info[cluster_id] = {
                "cluster_id": cluster_id,
                "label": f"cluster_{cluster_id}",
                "word_count": len(cluster_words),
                "words": cluster_words,
                "representative_words": representative_words,
            }

        return cluster_info

    def predict_cluster(
        self,
        embeddings: NDArray[np.floating[Any]],
        clusterer: hdbscan.HDBSCAN,
    ) -> NDArray[np.integer[Any]]:
        """Predict cluster for new embeddings using a fitted clusterer.

        Args:
            embeddings: New embeddings to predict.
            clusterer: Fitted HDBSCAN clusterer.

        Returns:
            Predicted cluster labels.
        """
        try:
            labels, _ = hdbscan.approximate_predict(clusterer, embeddings)
            return cast(NDArray[np.integer[Any]], labels)
        except Exception as e:
            raise ClusteringError(f"Failed to predict clusters: {e}") from e

    def calculate_cluster_quality(
        self,
        embeddings: NDArray[np.floating[Any]],
        labels: NDArray[np.integer[Any]],
    ) -> dict[str, float]:
        """Calculate quality metrics for clustering.

        Args:
            embeddings: Word embeddings.
            labels: Cluster labels.

        Returns:
            Dictionary with quality metrics:
            - silhouette_score: overall clustering quality
            - n_clusters: number of clusters
            - noise_ratio: proportion of unclustered points
        """
        from sklearn.metrics import silhouette_score

        # Filter out noise points for silhouette score
        non_noise_mask = labels != -1

        if len(set(labels[non_noise_mask])) < 2:
            # Need at least 2 clusters for silhouette score
            return {
                "silhouette_score": 0.0,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "noise_ratio": float(np.sum(labels == -1)) / len(labels),
            }

        try:
            # Use cosine distance for silhouette
            silhouette = silhouette_score(
                embeddings[non_noise_mask],
                labels[non_noise_mask],
                metric="cosine",
            )

            return {
                "silhouette_score": float(silhouette),
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "noise_ratio": float(np.sum(labels == -1)) / len(labels),
            }
        except Exception as e:
            logger.warning(f"Failed to calculate silhouette score: {e}")
            return {
                "silhouette_score": 0.0,
                "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
                "noise_ratio": float(np.sum(labels == -1)) / len(labels),
            }
