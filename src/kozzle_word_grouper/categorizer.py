"""LLM-based word categorization using Ollama."""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from kozzle_word_grouper.exceptions import CategorizationError, OllamaConnectionError
from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class Categorizer:
    """Categorize Korean words using LLM prompts to Ollama."""

    CLASSIFICATION_TYPES = ["하위개념", "기능", "사용맥락"]

    TYPE_DESCRIPTIONS = {
        "하위개념": "하위 개념(이 단어가 포함되는 더 넓은 범주)",
        "기능": "이 단어가 수행하는 기능이나 역할",
        "사용맥락": "이 단어가 사용되는 상황이나 맥락",
    }

    def __init__(
        self,
        model_name: str = "exaone3.5:7.8b",
        ollama_host: str | None = None,
        cache_file: Path | str | None = None,
        max_workers: int = 4,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """Initialize categorizer with Ollama.

        Args:
            model_name: Ollama model name.
            ollama_host: Ollama server URL.
            cache_file: Path to cache file for resume capability.
            max_workers: Number of concurrent workers.
            max_retries: Number of retry attempts per request.
            retry_delay: Delay between retries in seconds.

        Raises:
            OllamaConnectionError: If cannot connect to Ollama.
        """
        self.model_name = model_name
        self.ollama_host = ollama_host or "http://localhost:11434"
        self.cache_file = Path(cache_file) if cache_file else None
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._validate_connection()

        logger.info(f"Initialized categorizer with model: {model_name}")

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

    def _build_prompt(
        self,
        lemma: str,
        definition: str | None,
        classification_type: str,
    ) -> str:
        """Build prompt for Ollama.

        Args:
            lemma: Korean word.
            definition: Word definition (or None).
            classification_type: One of '하위개념', '기능', '사용맥락'.

        Returns:
            Formatted prompt string.
        """
        type_desc = self.TYPE_DESCRIPTIONS.get(classification_type, classification_type)

        base = (
            f"다음 한국어 단어를 {classification_type}({type_desc})으로 분류해주세요."
        )

        word_info = f"\n\n단어: {lemma}"
        if definition:
            word_info += f"\n정의: {definition}"
        word_info += "\n"

        json_format = """
JSON 형식으로 정확히 답변해주세요. 다른 텍스트는 포함하지 마세요.

{
  "categories": ["카테고리1", "카테고리2", "카테고리3"]
}

가장 적합한 카테고리 1-5개를 나열해주세요. 카테고리명은 간결하고 명확하게 작성해주세요."""

        return base + word_info + json_format

    def _parse_json_response(self, response: str) -> list[str]:
        """Parse JSON response from Ollama.

        Args:
            response: Raw response text from Ollama.

        Returns:
            List of categories.

        Raises:
            CategorizationError: If parsing fails and no categories found.
        """
        try:
            data = json.loads(response)
            categories = data.get("categories", [])
            if isinstance(categories, list):
                return [str(cat).strip() for cat in categories if cat]
            return []
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{[^}]*"categories"[^}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                categories = data.get("categories", [])
                if isinstance(categories, list):
                    return [str(cat).strip() for cat in categories if cat]
            except json.JSONDecodeError:
                pass

        list_match = re.search(r"\[[^\]]+\]", response)
        if list_match:
            try:
                categories = json.loads(list_match.group(0))
                if isinstance(categories, list):
                    return [str(cat).strip() for cat in categories if cat]
            except json.JSONDecodeError:
                pass

        categories = re.findall(r'"([^"]+)"', response)
        if categories:
            return [cat.strip() for cat in categories[:5] if cat.strip()]

        logger.warning(f"Could not parse categories from response: {response[:100]}")
        return []

    def _ask_ollama(
        self,
        lemma: str,
        definition: str | None,
        classification_type: str,
    ) -> list[str]:
        """Ask Ollama to categorize word by classification_type.

        Args:
            lemma: Korean word.
            definition: Word definition (or None).
            classification_type: One of '하위개념', '기능', '사용맥락'.

        Returns:
            List of categories.

        Raises:
            CategorizationError: If all retries fail.
        """
        prompt = self._build_prompt(lemma, definition, classification_type)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False},
                    timeout=60,
                )
                response.raise_for_status()

                data = response.json()
                response_text = data.get("response", "")

                if not response_text:
                    raise ValueError("Empty response from Ollama")

                categories = self._parse_json_response(response_text)

                if categories:
                    logger.debug(
                        f"Categorized '{lemma}' ({classification_type}): {categories}"
                    )
                    return categories

                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Empty categories for '{lemma}' ({classification_type}), "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(
                        f"No categories found for '{lemma}' ({classification_type}), "
                        "marking as '미분류'"
                    )
                    return ["미분류"]

            except requests.Timeout as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Timeout for '{lemma}' ({classification_type}), "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise CategorizationError(
                        f"Timeout categorizing '{lemma}' ({classification_type}) "
                        f"after {self.max_retries} attempts: {e}"
                    ) from e

            except requests.ConnectionError as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Connection error for '{lemma}' ({classification_type}), "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise CategorizationError(
                        f"Connection error categorizing '{lemma}' "
                        f"({classification_type}) after {self.max_retries} attempts: {e}"
                    ) from e

            except (ValueError, KeyError) as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Parse error for '{lemma}' ({classification_type}), "
                        f"attempt {attempt + 1}/{self.max_retries}: {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    raise CategorizationError(
                        f"Failed to parse response for '{lemma}' "
                        f"({classification_type}): {e}"
                    ) from e

        return ["미분류"]

    def categorize_word(self, word: KoreanWord) -> dict[str, list[str]]:
        """Categorize a word using 3 LLM prompts.

        Args:
            word: KoreanWord object with public_id, lemma, definition.

        Returns:
            Dictionary with classification types as keys:
                {
                    "하위개념": ["동물", "생물"],
                    "기능": ["애완동물"],
                    "사용맥락": ["일상대화"]
                }
        """
        categories = {}

        for classification_type in self.CLASSIFICATION_TYPES:
            try:
                cats = self._ask_ollama(
                    word.lemma, word.definition, classification_type
                )
                categories[classification_type] = cats
            except CategorizationError as e:
                logger.error(f"Failed to categorize '{word.lemma}': {e}")
                categories[classification_type] = ["미분류"]

        return categories

    def load_cache(self) -> dict[str, Any]:
        """Load previously processed words from cache.

        Returns:
            Dictionary with processed words and their categorizations.
        """
        if not self.cache_file or not self.cache_file.exists():
            return {"processed_words": {}, "metadata": {}}

        try:
            with open(self.cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data.get('processed_words', {}))} cached words")
            return data
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return {"processed_words": {}, "metadata": {}}

    def save_cache(self, data: dict[str, Any]) -> None:
        """Save categorization results to cache.

        Args:
            data: Dictionary with processed words and their categorizations.
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
    ) -> dict[str, dict]:
        """Categorize multiple words concurrently.

        Args:
            words: List of KoreanWord objects.
            show_progress: Whether to show progress.
            resume: Whether to resume from cache.

        Returns:
            Dictionary with public_id as key:
                {
                    "public_id_1": {
                        "lemma": "단어",
                        "definition": "...",
                        "categories": {
                            "하위개념": [...],
                            "기능": [...],
                            "사용맥락": [...]
                        }
                    },
                    ...
                }

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
            return processed_words

        logger.info(
            f"Categorizing {len(words_to_process)} words "
            f"({cached_count} from cache, {len(words_to_process)} new)"
        )

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.categorize_word, word): word
                for word in words_to_process
            }

            completed = 0
            total = len(words_to_process)

            for future in as_completed(futures):
                word = futures[future]

                try:
                    categories = future.result()

                    processed_words[word.public_id] = {
                        "lemma": word.lemma,
                        "definition": word.definition,
                        "categories": categories,
                    }

                    completed += 1

                    if show_progress and completed % 10 == 0:
                        total_processed = cached_count + completed
                        percentage = (total_processed / len(words)) * 100
                        logger.info(
                            f"Categorized {completed}/{total} new words "
                            f"({total_processed}/{len(words)} total, {percentage:.1f}%)"
                        )

                    if completed % 50 == 0:
                        cache_data["processed_words"] = processed_words
                        self.save_cache(cache_data)

                except CategorizationError as e:
                    logger.error(f"Failed to categorize word '{word.lemma}': {e}")
                    processed_words[word.public_id] = {
                        "lemma": word.lemma,
                        "definition": word.definition,
                        "categories": {
                            "하위개념": ["미분류"],
                            "기능": ["미분류"],
                            "사용맥락": ["미분류"],
                        },
                    }
                    completed += 1

        cache_data["processed_words"] = processed_words
        self.save_cache(cache_data)

        logger.info(f"Categorization complete: {len(processed_words)} words processed")

        return processed_words
