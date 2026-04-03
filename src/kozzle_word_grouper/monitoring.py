"""Connection pool monitoring and metrics."""

from kozzle_word_grouper.connection_pool import get_pool_stats
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


def log_connection_pool_stats() -> None:
    """Log current connection pool statistics."""
    stats = get_pool_stats()

    logger.info("Connection Pool Statistics:")
    logger.info(f"  Max connections: {stats['max_connections']}")
    logger.info(f"  Max keepalive: {stats['max_keepalive']}")
    logger.info(f"  Idle timeout: {stats['idle_timeout_ms']}ms")
    logger.info(f"  Connect timeout: {stats['connect_timeout_s']}s")
    logger.info(f"  Read timeout: {stats['read_timeout_s']}s")
    logger.info(f"  Write timeout: {stats['write_timeout_s']}s")
    logger.info(f"  Pool timeout: {stats['pool_timeout_s']}s")
    logger.info(f"  Pool active: {stats['is_active']}")


def get_connection_pool_metrics() -> dict[str, int | float | bool]:
    """Get connection pool metrics for monitoring.

    Returns:
        Dictionary with connection pool metrics.
    """
    return get_pool_stats()
