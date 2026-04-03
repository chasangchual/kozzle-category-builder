"""Tests for utils module."""

from pathlib import Path

from kozzle_word_grouper.utils import batch_list, ensure_directory


def test_batch_list_empty() -> None:
    """Test batching an empty list."""
    result = batch_list([], 10)
    assert result == []


def test_batch_list_small() -> None:
    """Test batching a small list."""
    items = [1, 2, 3, 4, 5]
    result = batch_list(items, 10)
    assert result == [[1, 2, 3, 4, 5]]


def test_batch_list_exact() -> None:
    """Test batching when list size is exact multiple of batch size."""
    items = list(range(10))
    result = batch_list(items, 5)
    assert result == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_batch_list_partial() -> None:
    """Test batching when list size is not exact multiple."""
    items = list(range(7))
    result = batch_list(items, 3)
    assert result == [[0, 1, 2], [3, 4, 5], [6]]


def test_ensure_directory_creates_new(tmp_path: Path) -> None:
    """Test creating a new directory."""
    new_dir = tmp_path / "new_dir"
    result = ensure_directory(new_dir)

    assert new_dir.exists()
    assert new_dir.is_dir()
    assert result == new_dir


def test_ensure_directory_existing(tmp_path: Path) -> None:
    """Test with existing directory."""
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()

    result = ensure_directory(existing_dir)

    assert existing_dir.exists()
    assert result == existing_dir


def test_ensure_directory_nested(tmp_path: Path) -> None:
    """Test creating nested directories."""
    nested_dir = tmp_path / "level1" / "level2" / "level3"

    result = ensure_directory(nested_dir)

    assert nested_dir.exists()
    assert result == nested_dir
