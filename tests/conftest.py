"""Test configuration and fixtures."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import NDArray

from kozzle_word_grouper.models import KoreanWord


@pytest.fixture
def sample_korean_words() -> list[KoreanWord]:
    """Sample Korean words for testing."""
    return [
        KoreanWord(public_id="1", lemma="개", definition="canine animal, dog"),
        KoreanWord(public_id="2", lemma="고양이", definition="feline pet, cat"),
        KoreanWord(public_id="3", lemma="새", definition="flying animal, bird"),
        KoreanWord(public_id="4", lemma="빨강", definition="red color"),
        KoreanWord(public_id="5", lemma="파랑", definition="blue color"),
        KoreanWord(public_id="6", lemma="기쁨", definition="happiness emotion"),
        KoreanWord(public_id="7", lemma="슬픔", definition="sadness emotion"),
        KoreanWord(public_id="8", lemma="책", definition="book, reading material"),
        KoreanWord(public_id="9", lemma="자동차", definition="automobile, car"),
        KoreanWord(public_id="10", lemma="바다", definition="ocean, sea"),
    ]


@pytest.fixture
def sample_words() -> list[str]:
    """Sample words for testing (legacy format)."""
    return [
        "dog",
        "cat",
        "bird",
        "fish",
        "lion",
        "tiger",
        "elephant",
        "zebra",
        "car",
        "truck",
        "bus",
        "bicycle",
        "motorcycle",
        "train",
        "airplane",
        "apple",
        "banana",
        "orange",
        "grape",
        "strawberry",
        "blueberry",
        "happy",
        "sad",
        "angry",
        "joyful",
        "excited",
        "calm",
        "anxious",
    ]


@pytest.fixture
def sample_embeddings(sample_words: list[str]) -> NDArray[np.floating]:
    """Sample embeddings for testing (random for unit tests)."""
    np.random.seed(42)
    return np.random.randn(len(sample_words), 384).astype(np.float32)


@pytest.fixture
def sample_cluster_labels(sample_words: list[str]) -> NDArray[np.integer]:
    """Sample cluster labels for testing."""
    # Assign words to clusters based on position
    labels = []
    for i, word in enumerate(sample_words):
        if i < 8:  # Animals
            labels.append(0)
        elif i < 15:  # Vehicles
            labels.append(1)
        elif i < 21:  # Fruits
            labels.append(2)
        else:  # Emotions
            labels.append(3)
    return np.array(labels)


@pytest.fixture
def temp_output_dir() -> Path:
    """Temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_supabase_client(sample_korean_words: list[KoreanWord]) -> Mock:
    """Mock Supabase client."""
    client = Mock()
    client.fetch_korean_words.return_value = sample_korean_words
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
    clusterer.get_cluster_info_korean.return_value = {
        0: {
            "cluster_id": 0,
            "label": "동물",
            "words": [
                {"public_id": "1", "lemma": "개"},
                {"public_id": "2", "lemma": "고양이"},
                {"public_id": "3", "lemma": "새"},
            ],
            "word_count": 3,
        },
        1: {
            "cluster_id": 1,
            "label": "색상",
            "words": [
                {"public_id": "4", "lemma": "빨강"},
                {"public_id": "5", "lemma": "파랑"},
            ],
            "word_count": 2,
        },
    }
    clusterer.calculate_cluster_quality.return_value = {
        "silhouette_score": 0.75,
        "n_clusters": 2,
        "noise_ratio": 0.0,
    }
    return clusterer


@pytest.fixture
def mock_labeler() -> Mock:
    """Mock cluster labeler."""
    labeler = Mock()
    labeler.label_clusters.return_value = {0: "동물", 1: "색상"}
    return labeler
