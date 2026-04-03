"""Tests for package initialization."""

from kozzle_word_grouper import __version__


def test_version() -> None:
    """Test that version is defined."""
    assert __version__ == "0.1.0"
    assert isinstance(__version__, str)
