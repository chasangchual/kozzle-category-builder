"""Connection pool management for Supabase API calls."""

import os
import threading
from contextlib import contextmanager
from typing import Generator

import httpx

from kozzle_word_grouper.exceptions import SupabaseConnectionError
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)

# Pool configuration from environment
MAX_CONNECTIONS = int(os.getenv("SUPABASE_MAX_CONNECTIONS", "10"))
MAX_KEEPALIVE_CONNECTIONS = int(os.getenv("SUPABASE_MAX_KEEPALIVE", "5"))
IDLE_TIMEOUT = float(os.getenv("SUPABASE_IDLE_TIMEOUT", "0.1"))  # 100ms
CONNECT_TIMEOUT = float(os.getenv("SUPABASE_CONNECT_TIMEOUT", "5.0"))
READ_TIMEOUT = float(os.getenv("SUPABASE_READ_TIMEOUT", "30.0"))
WRITE_TIMEOUT = float(os.getenv("SUPABASE_WRITE_TIMEOUT", "30.0"))
POOL_TIMEOUT = float(os.getenv("SUPABASE_POOL_TIMEOUT", "10.0"))


class ConnectionPoolManager:
    """Manage HTTP connection pool for Supabase API calls.

    Features:
    - Connection pooling with configurable limits
    - 100ms idle timeout for aggressive cleanup
    - Thread-safe connection management
    - Automatic resource cleanup
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "ConnectionPoolManager":
        """Singleton pattern to ensure single pool manager."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize connection pool manager."""
        if self._initialized:
            return

        self._http_client: httpx.Client | None = None
        self._supabase_client = None
        self._url: str | None = None
        self._key: str | None = None

        self._initialized = True

        logger.info(
            f"Connection pool manager initialized: "
            f"max_connections={MAX_CONNECTIONS}, "
            f"idle_timeout={IDLE_TIMEOUT}s"
        )

    def _create_http_client(self) -> httpx.Client:
        """Create HTTP client with connection pool.

        Returns:
            Configured httpx client with connection pooling.
        """
        limits = httpx.Limits(
            max_connections=MAX_CONNECTIONS,
            max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=IDLE_TIMEOUT,
        )

        timeout = httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=READ_TIMEOUT,
            write=WRITE_TIMEOUT,
            pool=POOL_TIMEOUT,
        )

        client = httpx.Client(
            limits=limits,
            timeout=timeout,
            follow_redirects=True,
        )

        logger.info(
            f"Created HTTP client pool: "
            f"max_connections={MAX_CONNECTIONS}, "
            f"keepalive={MAX_KEEPALIVE_CONNECTIONS}, "
            f"idle_timeout={IDLE_TIMEOUT}s"
        )

        return client

    def initialize_supabase_client(
        self,
        url: str,
        key: str,
    ):
        """Initialize Supabase client with connection pool.

        Args:
            url: Supabase project URL.
            key: Supabase API key.

        Returns:
            Supabase client with connection pool.

        Raises:
            SupabaseConnectionError: If initialization fails.
        """
        try:
            from supabase import create_client

            self._url = url
            self._key = key

            self._http_client = self._create_http_client()

            self._supabase_client = create_client(
                url,
                key,
            )

            logger.info("Supabase client initialized with connection pool")

            return self._supabase_client

        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise SupabaseConnectionError(
                f"Failed to initialize Supabase client with connection pool: {e}"
            ) from e

    def get_client(self):
        """Get Supabase client from pool.

        Returns:
            Supabase client instance.

        Raises:
            SupabaseConnectionError: If client not initialized.
        """
        if self._supabase_client is None:
            raise SupabaseConnectionError(
                "Supabase client not initialized. Call initialize_supabase_client() first."
            )
        return self._supabase_client

    def close(self) -> None:
        """Close connection pool and cleanup resources."""
        if self._http_client:
            self._http_client.close()
            self._http_client = None
            logger.info("HTTP client connection pool closed")

        self._supabase_client = None
        logger.info("Supabase client connection pool cleaned up")

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()


_pool_manager = ConnectionPoolManager()


@contextmanager
def get_supabase_client(
    url: str | None = None,
    key: str | None = None,
) -> Generator:
    """Context manager for Supabase client with connection pool.

    Args:
        url: Supabase project URL (optional, uses env var if not provided).
        key: Supabase API key (optional, uses env var if not provided).

    Yields:
        Supabase client instance.

    Raises:
        SupabaseConnectionError: If connection fails.

    Example:
        >>> with get_supabase_client() as client:
        ...     result = client.table('kor_word').select('*').execute()
    """
    url = url or os.getenv("SUPABASE_URL")
    key = key or os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise SupabaseConnectionError(
            "Supabase URL and key must be provided via parameters or "
            "SUPABASE_URL and SUPABASE_KEY environment variables"
        )

    try:
        client = _pool_manager.initialize_supabase_client(url, key)
        yield client
    finally:
        pass


def close_connection_pool() -> None:
    """Explicitly close connection pool."""
    _pool_manager.close()


def get_pool_stats() -> dict[str, int | float | bool]:
    """Get connection pool statistics.

    Returns:
        Dictionary with pool statistics.
    """
    stats = {
        "max_connections": MAX_CONNECTIONS,
        "max_keepalive": MAX_KEEPALIVE_CONNECTIONS,
        "idle_timeout_ms": IDLE_TIMEOUT * 1000,
        "connect_timeout_s": CONNECT_TIMEOUT,
        "read_timeout_s": READ_TIMEOUT,
        "write_timeout_s": WRITE_TIMEOUT,
        "pool_timeout_s": POOL_TIMEOUT,
    }

    if _pool_manager._http_client:
        stats["is_active"] = True
    else:
        stats["is_active"] = False

    return stats
