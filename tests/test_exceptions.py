"""Tests for exceptions module."""

import pytest

from kozzle_word_grouper.exceptions import (
    ClusteringError,
    DataRetrievalError,
    EmbeddingError,
    ExportError,
    SupabaseConnectionError,
    WordGrouperError,
)


def test_base_exception() -> None:
    """Test that base exception can be raised."""
    with pytest.raises(WordGrouperError):
        raise WordGrouperError("Test error")


def test_supabase_connection_error() -> None:
    """Test Supabase connection error."""
    error = SupabaseConnectionError("Connection failed")
    assert isinstance(error, WordGrouperError)
    assert str(error) == "Connection failed"


def test_data_retrieval_error() -> None:
    """Test data retrieval error."""
    error = DataRetrievalError("No data found")
    assert isinstance(error, WordGrouperError)
    assert str(error) == "No data found"


def test_embedding_error() -> None:
    """Test embedding error."""
    error = EmbeddingError("Model not found")
    assert isinstance(error, WordGrouperError)
    assert str(error) == "Model not found"


def test_clustering_error() -> None:
    """Test clustering error."""
    error = ClusteringError("Clustering failed")
    assert isinstance(error, WordGrouperError)
    assert str(error) == "Clustering failed"


def test_export_error() -> None:
    """Test export error."""
    error = ExportError("File not found")
    assert isinstance(error, WordGrouperError)
    assert str(error) == "File not found"


def test_exception_inheritance() -> None:
    """Test that all custom exceptions inherit from WordGrouperError."""
    exceptions = [
        SupabaseConnectionError("test"),
        DataRetrievalError("test"),
        EmbeddingError("test"),
        ClusteringError("test"),
        ExportError("test"),
    ]

    for exc in exceptions:
        assert isinstance(exc, WordGrouperError)
        assert isinstance(exc, Exception)
