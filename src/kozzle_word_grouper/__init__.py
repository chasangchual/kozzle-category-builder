"""Kozzle Word Grouper - Semantically group Korean words from Supabase using Ollama."""

__version__ = "0.2.0"

# Core classes
from kozzle_word_grouper.core import WordGrouperPipeline
from kozzle_word_grouper.clustering import WordClusterer
from kozzle_word_grouper.embeddings import EmbeddingGenerator
from kozzle_word_grouper.export import WordGroupExporter
from kozzle_word_grouper.labeler import ClusterLabeler
from kozzle_word_grouper.supabase_client import SupabaseClient
from kozzle_word_grouper.models import KoreanWord, ClusteredWord, WordCluster

# Utility functions
from kozzle_word_grouper.utils import setup_logging, get_logger, ensure_directory

# Exceptions
from kozzle_word_grouper.exceptions import (
    WordGrouperError,
    SupabaseConnectionError,
    DataRetrievalError,
    EmbeddingError,
    ClusteringError,
    ExportError,
    OllamaConnectionError,
    OllamaModelError,
    LabelGenerationError,
)

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
