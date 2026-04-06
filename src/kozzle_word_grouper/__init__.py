"""Kozzle Word Grouper - Semantically group Korean words from Supabase using Ollama."""

__version__ = "0.2.0"

# Core classes
from kozzle_word_grouper.clustering import WordClusterer
from kozzle_word_grouper.core import WordGrouperPipeline
from kozzle_word_grouper.embeddings import EmbeddingGenerator

# Exceptions
from kozzle_word_grouper.exceptions import (
    ClusteringError,
    DataRetrievalError,
    EmbeddingError,
    ExportError,
    LabelGenerationError,
    OllamaConnectionError,
    OllamaModelError,
    SupabaseConnectionError,
    WordGrouperError,
)
from kozzle_word_grouper.export import WordGroupExporter
from kozzle_word_grouper.labeler import ClusterLabeler
from kozzle_word_grouper.models import ClusteredWord, KoreanWord, WordCluster
from kozzle_word_grouper.supabase_client import SupabaseClient

# Utility functions
from kozzle_word_grouper.utils import ensure_directory, get_logger, setup_logging

__all__ = [
    # Version
    "__version__",
    # Core classes
    "WordGrouperPipeline",
    "WordClusterer",
    "EmbeddingGenerator",
    "WordGroupExporter",
    "ClusterLabeler",
    "SupabaseClient",
    # Models
    "KoreanWord",
    "ClusteredWord",
    "WordCluster",
    # Utilities
    "setup_logging",
    "get_logger",
    "ensure_directory",
    # Exceptions
    "WordGrouperError",
    "SupabaseConnectionError",
    "DataRetrievalError",
    "EmbeddingError",
    "ClusteringError",
    "ExportError",
    "OllamaConnectionError",
    "OllamaModelError",
    "LabelGenerationError",
]
