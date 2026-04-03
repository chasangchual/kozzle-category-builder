"""Data models for Korean word grouper."""

from dataclasses import dataclass
from typing import Any


@dataclass
class KoreanWord:
    """Represents a Korean word entry from database."""

    public_id: str
    lemma: str
    definition: str | None

    def get_text_for_embedding(self) -> str:
        """Return definition or lemma for embedding.

        Returns:
            Text to use for embedding (definition if available, else lemma).
        """
        if self.definition and self.definition.strip():
            return self.definition
        return self.lemma


@dataclass
class ClusteredWord:
    """Represents a clustered word with metadata."""

    public_id: str
    lemma: str
    definition: str | None
    cluster_id: int
    cluster_label: str | None = None


@dataclass
class WordCluster:
    """Represents a word cluster with Korean label."""

    cluster_id: int
    label: str  # Korean category name
    words: list[dict[str, str]]  # [{"public_id": "...", "lemma": "..."}]
    word_count: int
    representative_words: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for export.

        Returns:
            Dictionary representation of cluster.
        """
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "words": self.words,
            "word_count": self.word_count,
            "representative_words": self.representative_words,
        }
