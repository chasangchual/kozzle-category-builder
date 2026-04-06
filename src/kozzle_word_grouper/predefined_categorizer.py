"""Classify Korean words into pre-defined categories using LLM binary classification."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from kozzle_word_grouper.categorizer import RateLimiter
from kozzle_word_grouper.exceptions import CategorizationError, OllamaConnectionError
from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class PredefinedCategorizer:
    """Classify Korean words into pre-defined categories using Yes/No classification."""

    CATEGORY_GROUPS = [
        "concept_categories",
        "function_categories",
        "usage_context_categories",
    ]

    def __init__(
        self,
        categories_file: Path | str,
        model_name: str = "exaone3.5:7.8b",
        ollama_host: str | None = None,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        cache_file: Path | str | None = None,
        rate_limit: float = 2.0,
        cache_save_interval: int = 50,
    ) -> None:
        """Initialize predefined categorizer.

        Args:
            categories_file: Path to kor_words_categories.json.
            model_name: Ollama model name.
            ollama_host: Ollama server URL.
            max_workers: Number of concurrent workers.
            max_retries: Number of retry attempts per request.
            retry_delay: Delay between retries in seconds.
            cache_file: Path to cache file for resume capability.
            rate_limit: Maximum API calls per second (default: 2.0).
            cache_save_interval: Save cache every N words (default: 50).

        Raises:
            OllamaConnectionError: If cannot connect to Ollama.
            FileNotFoundError: If categories_file not found.
        """
        self.categories_file = Path(categories_file)
        self.model_name = model_name
        self.ollama_host = ollama_host or "http://localhost:11434"
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.cache_file = Path(cache_file) if cache_file else None
        self.rate_limit = rate_limit
        self.cache_save_interval = cache_save_interval
        self.rate_limiter = RateLimiter(rate_limit)

        if not self.categories_file.exists():
            raise FileNotFoundError(
                f"Categories file not found: {self.categories_file}"
            )

        self.categories = self._load_categories(self.categories_file)
        self._validate_connection()

        logger.info(
            f"Initialized predefined categorizer with model: {model_name}, "
            f"max_workers={max_workers}, rate_limit={rate_limit} calls/sec"
        )
        logger.info(
            f"Loaded {self._count_categories()} categories from {self.categories_file}"
        )

    def _validate_connection(self) -> None:
        """Validate Ollama connection."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            response.raise_for_status()
            logger.info(f"Successfully connected to Ollama at {self.ollama_host}")
        except Exception as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.ollama_host}: {e}"
            ) from e

    def _load_categories(self, categories_file: Path) -> dict:
        """Load pre-defined categories from JSON file.

        Args:
            categories_file: Path to categories JSON file.

        Returns:
            Dictionary with category groups.

        Raises:
            CategorizationError: If file cannot be parsed.
        """
        try:
            with open(categories_file, encoding="utf-8") as f:
                data = json.load(f)

            categories = {
                "concept_categories": data.get("concept_categories", []),
                "function_categories": data.get("function_categories", []),
                "usage_context_categories": data.get("usage_context_categories", []),
            }

            return categories

        except Exception as e:
            raise CategorizationError(f"Failed to load categories file: {e}") from e

    def _count_categories(self) -> int:
        """Count total number of categories."""
        return (
            len(self.categories["concept_categories"])
            + len(self.categories["function_categories"])
            + len(self.categories["usage_context_categories"])
        )

    def _build_binary_prompt(
        self,
        lemma: str,
        definition: str | None,
        category: dict,
    ) -> str:
        """Build prompt for Yes/No classification.

        Args:
            lemma: Korean word.
            definition: Word definition (or None).
            category: Category dict with id, name, description.

        Returns:
            Formatted prompt string.
        """
        prompt = f"""다음 한국어 단어가 주어진 카테고리에 속하는지 판단하세요.

단어: {lemma}
"""
        if definition:
            prompt += f"정의: {definition}\n\n"
        else:
            prompt += "\n"

        prompt += f"""카테고리: {category["name"]}
설명: {category["description"]}

이 단어가 이 카테고리에 속한다면 'yes', 속하지 않는다면 'no'라고 답하세요.
다른 설명 없이 yes 또는 no만 답하세요."""

        return prompt

    def _ask_llm_binary(self, prompt: str) -> bool:
        """Call Ollama API and parse Yes/No response.

        Args:
            prompt: Prompt string.

        Returns:
            True if response is "yes", False otherwise.

        Raises:
            CategorizationError: If all retries fail.
        """
        # Apply rate limiting before making API call
        self.rate_limiter.acquire()

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False},
                    timeout=30,
                )
                response.raise_for_status()

                data = response.json()
                response_text = data.get("response", "").strip().lower()

                if not response_text:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    return False

                if "yes" in response_text or "네" in response_text:
                    return True
                elif "no" in response_text or "아니" in response_text:
                    return False
                else:
                    if attempt < self.max_retries - 1:
                        logger.warning(
                            f"Ambiguous response '{response_text[:50]}', retrying..."
                        )
                        time.sleep(self.retry_delay)
                        continue
                    return False

            except requests.Timeout as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Timeout, attempt {attempt + 1}/{self.max_retries}")
                    time.sleep(self.retry_delay)
                else:
                    raise CategorizationError(
                        f"Timeout after {self.max_retries} attempts: {e}"
                    ) from e

            except requests.ConnectionError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Connection error, attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise CategorizationError(
                        f"Connection error after {self.max_retries} attempts: {e}"
                    ) from e

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Error, attempt {attempt + 1}/{self.max_retries}: {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise CategorizationError(
                        f"Failed after {self.max_retries} attempts: {e}"
                    ) from e

        return False

    def classify_word(
        self,
        word: KoreanWord,
    ) -> dict[str, Any]:
        """Classify a word against all 150 pre-defined categories.

        Args:
            word: KoreanWord object with public_id, lemma, definition.

        Returns:
            Dictionary with categorization results:
                {
                    "public_id": "...",
                    "lemma": "...",
                    "definition": "...",
                    "concept_categories": [{"id": 1, "name": "..."}, ...],
                    "function_categories": [{"id": 1, "name": "..."}, ...],
                    "usage_context_categories": [{"id": 1, "name": "...}, ...]
                }
        """
        result = {
            "public_id": word.public_id,
            "lemma": word.lemma,
            "definition": word.definition,
            "concept_categories": [],
            "function_categories": [],
            "usage_context_categories": [],
        }

        for category_group in self.CATEGORY_GROUPS:
            categories = self.categories[category_group]

            for category in categories:
                prompt = self._build_binary_prompt(
                    word.lemma, word.definition, category
                )

                if self._ask_llm_binary(prompt):
                    result[category_group].append(
                        {"id": category["id"], "name": category["name"]}
                    )

        return result

    def load_cache(self) -> dict[str, Any]:
        """Load previously processed words from cache.

        Returns:
            Dictionary with processed words and their categorizations.
        """
        if not self.cache_file or not self.cache_file.exists():
            return {"processed_words": {}, "metadata": {}}

        try:
            with open(self.cache_file, encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('processed_words', {}))} cached words")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {"processed_words": {}, "metadata": {}}

    def save_cache(self, data: dict[str, Any]) -> None:
        """Save categorization results to cache.

        Args:
            data: Dictionary with processed words and categorizations.
        """
        if not self.cache_file:
            return

        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)

            data["metadata"]["last_updated"] = datetime.now().isoformat()
            data["metadata"]["total_processed"] = len(data.get("processed_words", {}))

            temp_file = self.cache_file.with_suffix(".json.tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            temp_file.replace(self.cache_file)
            logger.debug(f"Saved cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")

    def categorize_words(
        self,
        words: list[KoreanWord],
        show_progress: bool = True,
        resume: bool = True,
    ) -> list[dict]:
        """Categorize multiple words into pre-defined categories.

        Args:
            words: List of KoreanWord objects.
            show_progress: Whether to show progress.
            resume: Whether to resume from cache.

        Returns:
            List of categorization results.

        Raises:
            CategorizationError: If categorization fails critically.
        """
        cache_data = {"processed_words": {}, "metadata": {}}

        if resume:
            cache_data = self.load_cache()

        processed_words = cache_data.get("processed_words", {})
        cached_count = len([w for w in words if w.public_id in processed_words])

        if cached_count > 0:
            logger.info(f"Resuming from cache: {cached_count} words already processed")

        words_to_process = [w for w in words if w.public_id not in processed_words]

        if not words_to_process:
            logger.info("All words already processed (from cache)")
            return [processed_words[w.public_id] for w in words]

        total_categories = self._count_categories()
        logger.info(
            f"Categorizing {len(words_to_process)} words "
            f"({cached_count} from cache, {len(words_to_process)} new)"
        )
        logger.info(f"Total categories per word: {total_categories}")
        logger.info(f"Estimated LLM calls: {len(words_to_process) * total_categories}")
        logger.info(f"Using {self.max_workers} concurrent workers with rate limiting")

        results = list(processed_words.values())

        completed = 0
        total = len(words_to_process)

        # Process in smaller batches to prevent overwhelming Ollama
        # Since each word triggers 150 API calls, we need smaller batches
        batch_size = 5  # Only 5 words at a time (each word = 150 calls)
        batch_delay = 2.0  # Wait 2s between batches (more conservative)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for batch_start in range(0, len(words_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(words_to_process))
                batch_words = words_to_process[batch_start:batch_end]

                # Add delay between batches (except first batch)
                if batch_start > 0:
                    time.sleep(batch_delay)

                futures = {
                    executor.submit(self.classify_word, word): word
                    for word in batch_words
                }

                for future in as_completed(futures):
                    word = futures[future]

                    try:
                        result = future.result()

                        processed_words[word.public_id] = result
                        results.append(result)

                        completed += 1

                        if show_progress and completed % 1 == 0:
                            total_processed = cached_count + completed
                            percentage = (total_processed / len(words)) * 100
                            logger.info(
                                f"Categorized {completed}/{total} new words "
                                f"({total_processed}/{len(words)} total, {percentage:.1f}%)"
                            )

                        # Save cache less frequently (every cache_save_interval words)
                        if completed % self.cache_save_interval == 0:
                            cache_data["processed_words"] = processed_words
                            self.save_cache(cache_data)

                    except CategorizationError as e:
                        logger.error(f"Failed to categorize word '{word.lemma}': {e}")
                        results.append(
                            {
                                "public_id": word.public_id,
                                "lemma": word.lemma,
                                "definition": word.definition,
                                "concept_categories": [],
                                "function_categories": [],
                                "usage_context_categories": [],
                            }
                        )
                    completed += 1

        cache_data["processed_words"] = processed_words
        self.save_cache(cache_data)

        logger.info(
            f"Predefined categorization complete: {len(results)} words processed"
        )

        return results
