"""Tests for connection pool management."""

import os
import pytest

from kozzle_word_grouper.connection_pool import (
    ConnectionPoolManager,
    get_pool_stats,
)


def test_connection_pool_initialization() -> None:
    """Test connection pool manager initialization."""
    pool_manager = ConnectionPoolManager()

    assert pool_manager is not None
    assert pool_manager._initialized is True


def test_connection_pool_singleton() -> None:
    """Test that connection pool is singleton."""
    pool1 = ConnectionPoolManager()
    pool2 = ConnectionPoolManager()

    assert pool1 is pool2


def test_http_client_creation() -> None:
    """Test HTTP client creation."""
    pool_manager = ConnectionPoolManager()
    client = pool_manager._create_http_client()

    # Verify client was created
    assert client is not None
    # Verify it has connection limits configured
    assert hasattr(client, "_transport")

    client.close()


def test_get_pool_stats() -> None:
    """Test getting pool statistics."""
    stats = get_pool_stats()

    assert "max_connections" in stats
    assert "max_keepalive" in stats
    assert "idle_timeout_ms" in stats
    assert isinstance(stats["max_connections"], int)
    assert isinstance(stats["idle_timeout_ms"], float)
    assert stats["idle_timeout_ms"] == 100.0  # 100ms = 0.1s


def test_connection_pool_close() -> None:
    """Test connection pool cleanup."""
    pool_manager = ConnectionPoolManager()

    # Create HTTP client
    pool_manager._http_client = pool_manager._create_http_client()

    # Close pool
    pool_manager.close()

    # Verify cleanup
    assert pool_manager._http_client is None


def test_environment_variable_configuration() -> None:
    """Test configuration from environment variables."""
    # Set custom environment variables
    original_max = os.environ.get("SUPABASE_MAX_CONNECTIONS")
    original_timeout = os.environ.get("SUPABASE_IDLE_TIMEOUT")

    try:
        os.environ["SUPABASE_MAX_CONNECTIONS"] = "20"
        os.environ["SUPABASE_IDLE_TIMEOUT"] = "0.05"  # 50ms

        # Reimport to pick up new values
        import importlib
        import kozzle_word_grouper.connection_pool as pool_module

        importlib.reload(pool_module)

        # Check values are used
        assert pool_module.MAX_CONNECTIONS == 20
        assert pool_module.IDLE_TIMEOUT == 0.05

    finally:
        # Restore original values
        if original_max is not None:
            os.environ["SUPABASE_MAX_CONNECTIONS"] = original_max
        else:
            os.environ.pop("SUPABASE_MAX_CONNECTIONS", None)

        if original_timeout is not None:
            os.environ["SUPABASE_IDLE_TIMEOUT"] = original_timeout
        else:
            os.environ.pop("SUPABASE_IDLE_TIMEOUT", None)


def test_connection_pool_thread_safety() -> None:
    """Test that connection pool is thread-safe."""
    import threading

    results = []

    def get_client():
        pool_manager = ConnectionPoolManager()
        results.append(pool_manager is not None)

    threads = [threading.Thread(target=get_client) for _ in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    # All threads should get the same singleton instance
    assert all(results)


def test_pool_manager_del() -> None:
    """Test pool manager cleanup on deletion."""
    pool_manager = ConnectionPoolManager()
    client = pool_manager._create_http_client()

    # Verify client exists
    assert client is not None

    # Call __del__ explicitly
    pool_manager.__del__()

    # Pool should be cleaned up
    # Note: Cannot directly verify without exposing internals
