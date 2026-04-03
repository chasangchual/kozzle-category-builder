"""Custom exceptions for kozzle-word-grouper."""


class WordGrouperError(Exception):
    """Base exception for word grouper errors."""

    pass


class SupabaseConnectionError(WordGrouperError):
    """Raised when connection to Supabase fails."""

    pass


class DataRetrievalError(WordGrouperError):
    """Raised when data retrieval from Supabase fails."""

    pass


class EmbeddingError(WordGrouperError):
    """Raised when embedding generation fails."""

    pass


class ClusteringError(WordGrouperError):
    """Raised when clustering fails."""

    pass


class ExportError(WordGrouperError):
    """Raised when exporting results fails."""

    pass


class OllamaConnectionError(WordGrouperError):
    """Raised when connection to Ollama fails."""

    pass


class OllamaModelError(WordGrouperError):
    """Raised when Ollama model is not found or invalid."""

    pass


class LabelGenerationError(WordGrouperError):
    """Raised when cluster label generation fails."""

    pass


class SupabaseRetryError(SupabaseConnectionError):
    """Raised when Supabase operations fail after all retries."""

    pass


class CategorizationError(WordGrouperError):
    """Raised when LLM categorization fails."""

    pass
