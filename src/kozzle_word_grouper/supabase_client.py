"""Supabase client for data retrieval with connection pooling."""

import os
from typing import Any

import requests
from supabase import Client

from kozzle_word_grouper.connection_pool import ConnectionPoolManager
from kozzle_word_grouper.exceptions import DataRetrievalError, SupabaseConnectionError
from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.retry import supabase_retry
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class SupabaseClient:
    """Client for interacting with Supabase with connection pooling and retry logic."""

    def __init__(
        self,
        url: str | None = None,
        key: str | None = None,
        use_mcp: bool = True,
    ) -> None:
        """Initialize Supabase client with connection pool.

        Args:
            url: Supabase project URL. If None, reads from SUPABASE_URL env var.
            key: Supabase anon/service key. If None, reads from SUPABASE_KEY env var.
            use_mcp: Whether to attempt MCP connection first (deprecated).

        Raises:
            SupabaseConnectionError: If connection fails.
        """
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self.use_mcp = use_mcp

        self._pool_manager = ConnectionPoolManager()
        self._client: Client | None = None

        if not self.url or not self.key:
            raise SupabaseConnectionError(
                "Supabase URL and key must be provided via parameters or "
                "SUPABASE_URL and SUPABASE_KEY environment variables"
            )

        from kozzle_word_grouper.connection_pool import get_pool_stats

        stats = get_pool_stats()
        logger.info(
            f"Supabase client configured: "
            f"max_connections={stats['max_connections']}, "
            f"idle_timeout={stats['idle_timeout_ms']}ms"
        )

    @property
    def client(self) -> Client:
        """Get or create Supabase client from connection pool.

        Returns:
            Supabase client instance from pool.

        Raises:
            SupabaseConnectionError: If client creation fails.
        """
        if self._client is None:
            try:
                assert self.url is not None
                assert self.key is not None
                self._client = self._pool_manager.initialize_supabase_client(
                    self.url, self.key
                )
                logger.info("Successfully connected to Supabase via connection pool")
            except requests.exceptions.ConnectionError as e:
                raise SupabaseConnectionError(
                    f"Failed to connect to Supabase (connection error): {e}"
                ) from e
            except requests.exceptions.Timeout as e:
                raise SupabaseConnectionError(
                    f"Failed to connect to Supabase (timeout): {e}"
                ) from e
            except Exception as e:
                raise SupabaseConnectionError(
                    f"Failed to create Supabase client: {e}"
                ) from e
        return self._client

    def close(self) -> None:
        """Close connection pool and cleanup resources."""
        self._pool_manager.close()
        self._client = None
        logger.info("Supabase client connection pool closed")

    def __enter__(self) -> "SupabaseClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()

    @supabase_retry
    def _execute_with_retry(
        self,
        query: Any,
        operation_name: str = "query",
    ) -> Any:
        """Execute Supabase query with retry logic.

        Args:
            query: Supabase query object.
            operation_name: Name of operation for logging.

        Returns:
            Query result.

        Raises:
            DataRetrievalError: If query fails after all retries.
        """
        try:
            logger.debug(f"Executing {operation_name}")
            result = query.execute()
            logger.debug(f"Successfully completed {operation_name}")
            return result
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error during {operation_name}: {e}")
            raise
        except requests.exceptions.Timeout as e:
            logger.warning(f"Timeout during {operation_name}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 503:
                    logger.warning(
                        f"Supabase service unavailable (503) during {operation_name}"
                    )
                    raise
                else:
                    logger.error(
                        f"HTTP error {e.response.status_code} during {operation_name}: {e}"
                    )
                    raise DataRetrievalError(
                        f"HTTP error {e.response.status_code}: {e}"
                    ) from e
            raise
        except Exception as e:
            logger.error(f"Unexpected error during {operation_name}: {e}")
            raise DataRetrievalError(f"Failed {operation_name}: {e}") from e

    @supabase_retry
    def fetch_korean_words(
        self,
        table_name: str = "kor_word",
        lemma_column: str = "lemma",
        definition_column: str = "definition",
        public_id_column: str = "public_id",
        level_column: str = "level",
        filter_level: list[int] | None = None,
        min_lemma_length: int | None = None,
        batch_size: int = 1000,
    ) -> list[KoreanWord]:
        """Fetch Korean words with public_id, lemma, and definition with retry logic.

        Args:
            table_name: Table name (default: kor_word).
            lemma_column: Lemma column name.
            definition_column: Definition column name.
            public_id_column: Public ID column name.
            level_column: Level column name for filtering.
            filter_level: List of levels to include (e.g., [1, 2] for level 1 or 2).
            min_lemma_length: Minimum lemma length (e.g., 2 for >= 2 characters).
            batch_size: Number of rows to fetch per batch (for pagination).

        Returns:
            List of KoreanWord objects.

        Raises:
            DataRetrievalError: If fetching fails after all retries.
        """
        try:
            logger.info(f"Fetching Korean words from {table_name}")

            all_words = []
            offset = 0
            batch_count = 0

            while True:
                # Build query with filters and pagination
                query = self.client.table(table_name).select(
                    f"{public_id_column}, {lemma_column}, {definition_column}"
                )

                # Apply level filter at database level
                if filter_level:
                    query = query.in_(level_column, filter_level)

                # Apply lemma length filter at database level (PostgreSQL)
                if min_lemma_length is not None:
                    query = query.gte(f"length({lemma_column})", min_lemma_length)

                # Apply pagination
                query = query.range(offset, offset + batch_size - 1)

                result = self._execute_with_retry(
                    query, f"fetch from {table_name} (batch {batch_count + 1})"
                )

                if not result.data:
                    # No more data
                    break

                # Process this batch
                batch_count += 1

                for row in result.data:
                    if isinstance(row, dict):
                        word = KoreanWord(
                            public_id=str(row.get(public_id_column, "")),
                            lemma=row.get(lemma_column, ""),
                            definition=row.get(definition_column),
                        )
                        all_words.append(word)

                logger.info(f"Batch {batch_count}: fetched {len(result.data)} words")

                # Check if we've fetched all data
                if len(result.data) < batch_size:
                    # Last batch
                    break

                offset += batch_size

            logger.info(
                f"Retrieved {len(all_words)} words from Supabase "
                f"({batch_count} batches)"
            )
            return all_words

        except requests.exceptions.ConnectionError:
            logger.error(f"Connection lost while fetching from {table_name}")
            raise
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching from {table_name}")
            raise
        except Exception as e:
            logger.error(f"Failed to fetch Korean words: {e}")
            raise DataRetrievalError(
                f"Failed to fetch words from {table_name}: {e}"
            ) from e

    @supabase_retry
    def fetch_words(
        self,
        table_name: str = "words",
        word_column: str = "word",
        batch_size: int = 1000,
        select_columns: str = "*",
    ) -> list[str]:
        """Fetch words from Supabase table with retry logic.

        Args:
            table_name: Name of the table containing words.
            word_column: Name of the column containing the word text.
            batch_size: Number of rows to fetch per batch.
            select_columns: Columns to select (default: all columns).

        Returns:
            List of words.

        Raises:
            DataRetrievalError: If data retrieval fails after all retries.
        """
        try:
            logger.info(
                f"Fetching words from table '{table_name}', column '{word_column}'"
            )

            query = self.client.table(table_name).select(select_columns)
            result = self._execute_with_retry(query, f"fetch words from {table_name}")

            if not result.data:
                logger.warning(f"No data found in table {table_name}")
                return []

            words = []
            for row in result.data:
                if isinstance(row, dict) and word_column in row:
                    word_value = row[word_column]
                    if isinstance(word_value, str):
                        words.append(word_value)
                    else:
                        logger.warning(
                            f"Column '{word_column}' value "
                            f"is not a string: {word_value}"
                        )
                else:
                    logger.warning(f"Column '{word_column}' not found in row: {row}")

            logger.info(f"Retrieved {len(words)} words from Supabase")
            return words

        except requests.exceptions.ConnectionError:
            logger.error(f"Connection lost while fetching from {table_name}")
            raise
        except requests.exceptions.Timeout:
            logger.error(f"Timeout while fetching from {table_name}")
            raise
        except Exception as e:
            raise DataRetrievalError(
                f"Failed to fetch words from {table_name}: {e}"
            ) from e

    def get_table_schema(self, table_name: str) -> dict[str, Any]:
        """Get schema information for a table.

        Args:
            table_name: Name of the table.

        Returns:
            Dictionary with column information.

        Raises:
            DataRetrievalError: If schema retrieval fails.
        """
        try:
            result = self.client.table(table_name).select("*").limit(1).execute()

            if result.data:
                row = result.data[0]
                if isinstance(row, dict):
                    return {key: type(value).__name__ for key, value in row.items()}
            logger.warning(f"Table {table_name} is empty")
            return {}
        except Exception as e:
            raise DataRetrievalError(
                f"Failed to get schema for table {table_name}: {e}"
            ) from e

    def close_connection(self) -> None:
        """Close the Supabase client connection."""
        if self._client is not None:
            self._client = None
            logger.info("Supabase client connection cleared")
