"""Retry logic for external API calls."""

import os
import functools
from typing import Any, Callable

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import requests

from kozzle_word_grouper.exceptions import DataRetrievalError
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)

# Retry configuration from environment
MAX_RETRIES = int(os.getenv("SUPABASE_MAX_RETRIES", "3"))
RETRY_DELAY = float(os.getenv("SUPABASE_RETRY_DELAY", "1.0"))
RETRY_MAX_DELAY = float(os.getenv("SUPABASE_RETRY_MAX_DELAY", "10.0"))


def supabase_retry(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to retry Supabase operations on transient failures.

    Retries on:
    - Connection errors (requests.exceptions.ConnectionError)
    - Timeout errors (requests.exceptions.Timeout)
    - HTTP 503 errors (service unavailable)

    Uses exponential backoff with configurable limits.

    Args:
        func: Function to wrap with retry logic.

    Returns:
        Wrapped function with retry logic.
    """

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(
            multiplier=RETRY_DELAY,
            max=RETRY_MAX_DELAY,
        ),
        retry=retry_if_exception_type(
            (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.HTTPError,
            )
        ),
        before_sleep=before_sleep_log(logger, log_level=20),
        reraise=True,
    )
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except requests.exceptions.HTTPError as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 503:
                    logger.warning(
                        f"Supabase returned 503, retrying... "
                        f"(attempt {retry.statistics.get('attempt_number', 1)})"
                    )
                    raise
            raise
        except Exception as e:
            logger.error(f"Supabase error: {e}")
            raise

    return wrapper
