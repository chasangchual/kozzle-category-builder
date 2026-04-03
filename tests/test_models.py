"""Tests for models module."""

import pytest
from kozzle_word_grouper.models import KoreanWord, ClusteredWord, WordCluster


def test_korean_word_creation() -> None:
    """Test KoreanWord instantiation."""
    word = KoreanWord(public_id="123", lemma="개", definition="canine animal")

    assert word.public_id == "123"
    assert word.lemma == "개"
    assert word.definition == "canine animal"


def test_korean_word_embedding_text_with_definition() -> None:
    """Test get_text_for_embedding with definition."""
    word = KoreanWord(public_id="123", lemma="개", definition="canine animal")

    assert word.get_text_for_embedding() == "canine animal"


def test_korean_word_embedding_text_without_definition() -> None:
    """Test get_text_for_embedding without definition."""
    word = KoreanWord(public_id="123", lemma="개", definition=None)

    assert word.get_text_for_embedding() == "개"


def test_korean_word_embedding_text_with_empty_definition() -> None:
    """Test get_text_for_embedding with empty definition."""
    word = KoreanWord(public_id="123", lemma="개", definition="   ")

    assert word.get_text_for_embedding() == "개"


def test_clustered_word_creation() -> None:
    """Test ClusteredWord instantiation."""
    word = ClusteredWord(
        public_id="123",
        lemma="개",
        definition="canine animal",
        cluster_id=1,
        cluster_label="동물",
    )

    assert word.public_id == "123"
    assert word.lemma == "개"
    assert word.cluster_id == 1
    assert word.cluster_label == "동물"


def test_word_cluster_to_dict() -> None:
    """Test WordCluster to_dict conversion."""
    cluster = WordCluster(
        cluster_id=1,
        label="동물",
        words=[
            {"public_id": "123", "lemma": "개"},
            {"public_id": "456", "lemma": "고양이"},
        ],
        word_count=2,
        representative_words=["개", "고양이"],
    )

    result = cluster.to_dict()

    assert result["cluster_id"] == 1
    assert result["label"] == "동물"
    assert len(result["words"]) == 2
    assert result["word_count"] == 2
    assert len(result["representative_words"]) == 2
