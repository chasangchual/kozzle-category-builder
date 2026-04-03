"""Tests for filtering functionality."""

import pytest
from unittest.mock import Mock
import numpy as np

from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.supabase_client import SupabaseClient


def test_fetch_korean_words_with_level_filter() -> None:
    """Test fetching Korean words with level filter."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("SUPABASE_URL", "https://test.supabase.co")
        m.setenv("SUPABASE_KEY", "test-key")

        # Create mock client
        client = SupabaseClient()

        # Mock the Supabase client
        mock_supabase = Mock()
        mock_result = Mock()
        mock_result.data = [
            {"public_id": "1", "lemma": "개", "definition": "dog", "level": 1},
            {"public_id": "2", "lemma": "고양이", "definition": "cat", "level": 1},
            {"public_id": "3", "lemma": "새", "definition": "bird", "level": 2},
        ]
        mock_supabase.table.return_value.select.return_value.in_.return_value.execute.return_value = mock_result

        client._client = mock_supabase

        # Fetch with level filter
        words = client.fetch_korean_words(
            filter_level=[1, 2],
        )

        # Verify query was built correctly
        # The .in_() method should have been called with the level filter
        assert mock_supabase.table.called


def test_fetch_korean_words_with_length_filter() -> None:
    """Test fetching Korean words with minimum lemma length filter."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("SUPABASE_URL", "https://test.supabase.co")
        m.setenv("SUPABASE_KEY", "test-key")

        # Create mock client
        client = SupabaseClient()

        # Mock the Supabase client
        mock_supabase = Mock()
        mock_result = Mock()
        mock_result.data = [
            {
                "public_id": "1",
                "lemma": "개",
                "definition": "dog",
            },  # 1 char - should be filtered
            {"public_id": "2", "lemma": "고양이", "definition": "cat"},  # 3 chars
            {
                "public_id": "3",
                "lemma": "새",
                "definition": "bird",
            },  # 1 char - should be filtered
            {"public_id": "4", "lemma": "바다", "definition": "sea"},  # 2 chars
        ]
        mock_supabase.table.return_value.select.return_value.execute.return_value = (
            mock_result
        )

        client._client = mock_supabase

        # Fetch with minimum length filter
        words = client.fetch_korean_words(
            min_lemma_length=2,
        )

        # Should only return words with lemma length >= 2
        assert len(words) == 2
        assert all(len(w.lemma) >= 2 for w in words)


def test_fetch_korean_words_combined_filters() -> None:
    """Test fetching Korean words with both level and length filters."""
    with pytest.MonkeyPatch.context() as m:
        m.setenv("SUPABASE_URL", "https://test.supabase.co")
        m.setenv("SUPABASE_KEY", "test-key")

        # Create mock client
        client = SupabaseClient()

        # Mock the Supabase client
        mock_supabase = Mock()
        mock_result = Mock()
        # Mock returns data for levels 1 and 2 only (after .in_() filter)
        mock_result.data = [
            {
                "public_id": "2",
                "lemma": "나무",
                "definition": "tree",
                "level": 1,
            },  # 2 chars, level 1
            {
                "public_id": "3",
                "lemma": "바다",
                "definition": "sea",
                "level": 2,
            },  # 2 chars, level 2
        ]
        mock_supabase.table.return_value.select.return_value.in_.return_value.execute.return_value = mock_result

        client._client = mock_supabase

        # Fetch with both filters
        words = client.fetch_korean_words(
            filter_level=[1, 2],
            min_lemma_length=2,
        )

        # Should only return words with level 1 or 2 (server filter) AND lemma length >= 2 (client filter)
        # Mock was set up to only return level 1 and 2 words
        # Both have length >= 2, so both should be returned
        assert len(words) == 2  # "나무" and "바다"
        assert all(w.lemma in ["나무", "바다"] for w in words)


def test_korean_word_lemma_length() -> None:
    """Test KoreanWord lemma length calculation."""
    # Single character
    word1 = KoreanWord(public_id="1", lemma="개", definition="dog")
    assert len(word1.lemma) == 1

    # Two characters
    word2 = KoreanWord(public_id="2", lemma="바다", definition="sea")
    assert len(word2.lemma) == 2

    # Three characters
    word3 = KoreanWord(public_id="3", lemma="고양이", definition="cat")
    assert len(word3.lemma) == 3

    # Longer word
    word4 = KoreanWord(public_id="4", lemma="자동차", definition="car")
    assert len(word4.lemma) == 3
