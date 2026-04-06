"""Compress and merge similar categories using LLM."""

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from kozzle_word_grouper.exceptions import CategorizationError, OllamaConnectionError
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class CategoryCompressor:
    """Compress categories by normalizing, merging duplicates, and semantic grouping."""

    CLASSIFICATION_TYPES = ["하위개념", "기능", "사용맥락"]

    def __init__(
        self,
        model_name: str = "exaone3.5:7.8b",
        ollama_host: str | None = None,
        batch_size: int = 50,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        """Initialize category compressor.

        Args:
            model_name: Ollama model name.
            ollama_host: Ollama server URL.
            batch_size: Number of categories per LLM call.
            max_retries: Number of retry attempts per request.
            retry_delay: Delay between retries in seconds.

        Raises:
            OllamaConnectionError: If cannot connect to Ollama.
        """
        self.model_name = model_name
        self.ollama_host = ollama_host or "http://localhost:11434"
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if model_name:
            self._validate_connection()

        logger.info(f"Initialized category compressor with model: {model_name}")

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

    def compress_categories(
        self,
        category_index: dict[str, dict[str, list]],
        categorizations: dict[str, dict],
        use_llm_merge: bool = True,
        min_word_count: int | None = None,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """Compress categories through normalization and optional LLM merging.

        Args:
            category_index: Category index from aggregator.
            categorizations: Original categorizations dict.
            use_llm_merge: Whether to use LLM for semantic merging.
            min_word_count: Minimum number of words to keep a category (None = no filter).
            show_progress: Whether to show progress.

        Returns:
            Dictionary with:
                - compressed_index: Compressed category index
                - statistics: Statistics for each classification type
                - merge_log: Log of what was merged
                - categorizations: Updated categorizations array
        """
        logger.info("Starting category compression...")

        original_stats = self._calculate_statistics(category_index)
        logger.info(
            f"Original categories: "
            f"{original_stats['하위개념']['total_categories']} 하위개념, "
            f"{original_stats['기능']['total_categories']} 기능, "
            f"{original_stats['사용맥락']['total_categories']} 사용맥락"
        )

        normalized_index = self._normalize_categories(category_index)

        after_normalize_stats = self._calculate_statistics(normalized_index)
        logger.info(
            f"After normalization: "
            f"{after_normalize_stats['하위개념']['total_categories']} 하위개념, "
            f"{after_normalize_stats['기능']['total_categories']} 기능, "
            f"{after_normalize_stats['사용맥락']['total_categories']} 사용맥락"
        )

        merged_index = self._merge_exact_duplicates(normalized_index)

        after_merge_stats = self._calculate_statistics(merged_index)
        logger.info(
            f"After merging duplicates: "
            f"{after_merge_stats['하위개념']['total_categories']} 하위개념, "
            f"{after_merge_stats['기능']['total_categories']} 기능, "
            f"{after_merge_stats['사용맥락']['total_categories']} 사용맥락"
        )

        if use_llm_merge:
            logger.info("Performing LLM-based semantic merging...")
            final_index = self._merge_semantic_similar(merged_index, show_progress)

            after_llm_stats = self._calculate_statistics(final_index)
            logger.info(
                f"After LLM merging: "
                f"{after_llm_stats['하위개념']['total_categories']} 하위개념, "
                f"{after_llm_stats['기능']['total_categories']} 기능, "
                f"{after_llm_stats['사용맥락']['total_categories']} 사용맥락"
            )
        else:
            final_index = merged_index

        sorted_index = self._sort_by_word_count(final_index)

        if min_word_count is not None and min_word_count > 0:
            before_filter_stats = self._calculate_statistics(sorted_index)
            logger.info(
                f"Before filtering: "
                f"{before_filter_stats['하위개념']['total_categories']} 하위개념, "
                f"{before_filter_stats['기능']['total_categories']} 기능, "
                f"{before_filter_stats['사용맥락']['total_categories']} 사용맥락"
            )

            sorted_index = self._filter_by_word_count(sorted_index, min_word_count)

            after_filter_stats = self._calculate_statistics(sorted_index)
            logger.info(
                f"After filtering (min_word_count={min_word_count}): "
                f"{after_filter_stats['하위개념']['total_categories']} 하위개념, "
                f"{after_filter_stats['기능']['total_categories']} 기능, "
                f"{after_filter_stats['사용맥락']['total_categories']} 사용맥락"
            )

        statistics = self._calculate_statistics(sorted_index)

        merge_log = self._generate_merge_log(
            category_index, sorted_index, use_llm_merge
        )

        updated_categorizations = self._update_categorizations(
            categorizations, sorted_index
        )

        logger.info("Category compression complete")

        return {
            "compressed_index": sorted_index,
            "statistics": statistics,
            "merge_log": merge_log,
            "categorizations": updated_categorizations,
            "original_stats": original_stats,
        }

    def _normalize_category_name(self, category: str) -> str:
        """Remove spaces and normalize category name.

        Args:
            category: Original category name.

        Returns:
            Normalized category name.

        Example:
            "동 물" → "동물"
            "동물" → "동물"
        """
        return category.replace(" ", "").strip()

    def _normalize_categories(
        self, category_index: dict[str, dict[str, list]]
    ) -> dict[str, dict[str, list]]:
        """Normalize all category names in the index.

        Args:
            category_index: Original category index.

        Returns:
            Normalized index with merged duplicate names.
        """
        normalized: dict[str, dict[str, list]] = {
            "하위개념": {},
            "기능": {},
            "사용맥락": {},
        }

        for class_type in self.CLASSIFICATION_TYPES:
            if class_type not in category_index:
                continue

            for category, words in category_index[class_type].items():
                normalized_name = self._normalize_category_name(category)

                if normalized_name not in normalized[class_type]:
                    normalized[class_type][normalized_name] = []

                normalized[class_type][normalized_name].extend(words)

        return normalized

    def _merge_exact_duplicates(
        self, normalized_index: dict[str, dict[str, list]]
    ) -> dict[str, dict[str, list]]:
        """Merge categories that are identical after normalization.

        Remove duplicate word entries (same public_id).

        Args:
            normalized_index: Normalized category index.

        Returns:
            Merged index with duplicates removed.
        """
        merged: dict[str, dict[str, list]] = {
            "하위개념": {},
            "기능": {},
            "사용맥락": {},
        }

        for class_type in self.CLASSIFICATION_TYPES:
            if class_type not in normalized_index:
                continue

            for category, words in normalized_index[class_type].items():
                unique_words = {}
                for word in words:
                    unique_words[word["public_id"]] = word

                merged[class_type][category] = list(unique_words.values())

        return merged

    def _merge_semantic_similar(
        self,
        merged_index: dict[str, dict[str, list]],
        show_progress: bool = True,
    ) -> dict[str, dict[str, list]]:
        """Use LLM to merge semantically similar categories.

        Process in batches of `batch_size` categories per LLM call.

        Args:
            merged_index: Index after merging exact duplicates.
            show_progress: Whether to show progress.

        Returns:
            Index with semantically merged categories.
        """
        final_index: dict[str, dict[str, list]] = {
            "하위개념": {},
            "기능": {},
            "사용맥락": {},
        }

        for class_type in self.CLASSIFICATION_TYPES:
            if show_progress:
                logger.info(f"Processing {class_type}...")

            if class_type not in merged_index:
                continue

            categories = list(merged_index[class_type].keys())

            if not categories:
                continue

            all_groupings = {}

            for i in range(0, len(categories), self.batch_size):
                batch = categories[i : i + self.batch_size]

                if show_progress:
                    logger.info(
                        f"Processing batch {i // self.batch_size + 1} "
                        f"({len(batch)} categories)..."
                    )

                groupings = self._get_llm_category_grouping(batch, class_type)

                all_groupings.update(groupings)

                if show_progress and i + self.batch_size < len(categories):
                    logger.info(
                        f"Processed {i + len(batch)}/{len(categories)} categories"
                    )

            for group_name, category_list in all_groupings.items():
                combined_words = []
                for cat in category_list:
                    if cat in merged_index[class_type]:
                        combined_words.extend(merged_index[class_type][cat])

                unique_words = {}
                for word in combined_words:
                    unique_words[word["public_id"]] = word

                final_index[class_type][group_name] = list(unique_words.values())

        return final_index

    def _get_llm_category_grouping(
        self, categories: list[str], classification_type: str
    ) -> dict[str, list[str]]:
        """Ask LLM to group categories and suggest unified names.

        Args:
            categories: List of category names to group.
            classification_type: One of '하위개념', '기능', '사용맥락'.

        Returns:
            Dictionary mapping group_name -> list of original categories.

        Example:
            Input: ["동물", "동물류", "생물", "짐승", "음식"]
            Output: {
                "동물": ["동물", "동물류", "생물", "짐승"],
                "음식": ["음식"]
            }
        """
        if len(categories) == 1:
            return {categories[0]: categories}

        prompt = f"""다음 카테고리들을 의미상 그룹화하세요.각 그룹에 적절한 대표카테고리명을 제안하세요.

분류 유형: {classification_type}

카테고리 목록:
{chr(10).join(f"- {cat}" for cat in categories)}

JSON 형식으로 정확히 답변해주세요. 다른 텍스트는 포함하지 마세요:

{{
  "groups": [
    {{
      "group_name": "대표카테고리명",
      "categories": ["카테고리1", "카테고리2"]
    }},
    {{
      "group_name": "다른대표카테고리명",
      "categories": ["카테고리3"]
    }}
  ]
}}

유의사항:
1. 의미가 비슷한 카테고리들을 같은 그룹으로 묶으세요
2. 각 그룹의 대표카테고리명은 가장 일반적이고 명확한 용어를 선택하세요
3. 너무 세분화하지 말고, 적절히 그룹화하세요
4. 단일 카테고리도 반드시 포함하세요 (categories 배열에 하나만 포함)
5. 모든 카테고리를 그룹에 포함하세요"""

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": self.model_name, "prompt": prompt, "stream": False},
                    timeout=120,
                )
                response.raise_for_status()

                data = response.json()
                response_text = data.get("response", "")

                if not response_text:
                    raise ValueError("Empty response from Ollama")

                groupings = self._parse_grouping_response(response_text)

                if groupings:
                    logger.debug(
                        f"Grouped {len(categories)} categories into {len(groupings)} groups"
                    )
                    return groupings

                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Failed to parse grouping response, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(
                        "Could not parse grouping response, using original categories"
                    )
                    return {cat: [cat] for cat in categories}

            except requests.Timeout as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Timeout for grouping request, "
                        f"attempt {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(f"Timeout after {self.max_retries} attempts: {e}")
                    return {cat: [cat] for cat in categories}

            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(
                        f"Error grouping categories, "
                        f"attempt {attempt + 1}/{self.max_retries}: {e}"
                    )
                    time.sleep(self.retry_delay)
                else:
                    logger.warning(f"Failed to group categories: {e}, using original")
                    return {cat: [cat] for cat in categories}

        return {cat: [cat] for cat in categories}

    def _parse_grouping_response(self, response: str) -> dict[str, list[str]]:
        """Parse LLM grouping response.

        Args:
            response: Raw response text from Ollama.

        Returns:
            Dictionary mapping group_name -> list of categories.
        """
        try:
            data = json.loads(response)
            if "groups" in data:
                groupings = {}
                for group in data["groups"]:
                    group_name = group.get("group_name", "")
                    categories = group.get("categories", [])

                    if group_name and categories:
                        groupings[group_name] = categories

                return groupings
        except json.JSONDecodeError:
            pass

        json_match = re.search(r'\{[^}]*"groups"[^}]*\[.*?\]\s*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                if "groups" in data:
                    groupings = {}
                    for group in data["groups"]:
                        group_name = group.get("group_name", "")
                        categories = group.get("categories", [])

                        if group_name and categories:
                            groupings[group_name] = categories

                    return groupings
            except json.JSONDecodeError:
                pass

        groups_pattern = (
            r'"group_name"\s*:\s*"([^"]+)"[^}]*"categories"\s*:\s*\[([^\]]+)\]'
        )
        matches = re.findall(groups_pattern, response, re.DOTALL)

        if matches:
            groupings = {}
            for group_name, cats_str in matches:
                cats = re.findall(r'"([^"]+)"', cats_str)
                if cats:
                    groupings[group_name] = cats
            return groupings

        return {}

    def _sort_by_word_count(
        self, merged_index: dict[str, dict[str, list]]
    ) -> dict[str, dict[str, list]]:
        """Sort categories by word count (descending).

        Args:
            merged_index: Index after merging.

        Returns:
            Index with categories sorted by word count.
        """
        sorted_index: dict[str, dict[str, list]] = {
            "하위개념": {},
            "기능": {},
            "사용맥락": {},
        }

        for class_type in self.CLASSIFICATION_TYPES:
            if class_type not in merged_index:
                continue

            sorted_categories = sorted(
                merged_index[class_type].items(),
                key=lambda x: len(x[1]),
                reverse=True,
            )

            for category, words in sorted_categories:
                sorted_index[class_type][category] = words

        return sorted_index

    def _filter_by_word_count(
        self,
        sorted_index: dict[str, dict[str, list]],
        min_word_count: int,
    ) -> dict[str, dict[str, list]]:
        """Filter categories by minimum word count.

        Args:
            sorted_index: Index sorted by word count.
            min_word_count: Minimum number of words to keep a category.

        Returns:
            Index with categories filtered by word count.
        """
        filtered_index: dict[str, dict[str, list]] = {
            "하위개념": {},
            "기능": {},
            "사용맥락": {},
        }

        for class_type in self.CLASSIFICATION_TYPES:
            if class_type not in sorted_index:
                continue

            filtered_count = 0
            kept_count = 0

            for category, words in sorted_index[class_type].items():
                if len(words) >= min_word_count:
                    filtered_index[class_type][category] = words
                    kept_count += 1
                else:
                    filtered_count += 1

            logger.debug(
                f"{class_type}: kept {kept_count} categories, "
                f"filtered out {filtered_count} categories (<{min_word_count} words)"
            )

        return filtered_index

    def _calculate_statistics(
        self, category_index: dict[str, dict[str, list]]
    ) -> dict[str, dict]:
        """Calculate statistics for each classification type.

        Args:
            category_index: Category index.

        Returns:
            Statistics dictionary.
        """
        statistics = {}

        for class_type in self.CLASSIFICATION_TYPES:
            if class_type not in category_index:
                statistics[class_type] = {
                    "total_categories": 0,
                    "total_words": 0,
                    "avg_words_per_category": 0,
                    "max_words": 0,
                    "min_words": 0,
                    "top_10_categories": [],
                }
                continue

            categories = category_index[class_type]

            if not categories:
                statistics[class_type] = {
                    "total_categories": 0,
                    "total_words": 0,
                    "avg_words_per_category": 0,
                    "max_words": 0,
                    "min_words": 0,
                    "top_10_categories": [],
                }
                continue

            word_counts = [len(words) for words in categories.values()]

            total_categories = len(categories)
            total_words = sum(word_counts)
            avg_words = total_words / len(word_counts) if word_counts else 0
            max_words = max(word_counts) if word_counts else 0
            min_words = min(word_counts) if word_counts else 0

            sorted_categories = sorted(
                categories.items(), key=lambda x: len(x[1]), reverse=True
            )
            top_10 = [
                {"category": cat, "count": len(words)}
                for cat, words in sorted_categories[:10]
            ]

            statistics[class_type] = {
                "total_categories": total_categories,
                "total_words": total_words,
                "avg_words_per_category": round(avg_words, 2),
                "max_words": max_words,
                "min_words": min_words,
                "top_10_categories": top_10,
            }

        return statistics

    def _generate_merge_log(
        self,
        original_index: dict[str, dict[str, list]],
        compressed_index: dict[str, dict[str, list]],
        use_llm_merge: bool,
    ) -> dict[str, list]:
        """Generate log of category merges.

        Args:
            original_index: Original category index.
            compressed_index: Compressed category index.
            use_llm_merge: Whether LLM merging was used.

        Returns:
            Merge log for each classification type.
        """
        merge_log: dict[str, list] = {
            "하위개념": [],
            "기능": [],
            "사용맥락": [],
        }

        for class_type in self.CLASSIFICATION_TYPES:
            if class_type not in compressed_index:
                continue

            for compressed_cat, words in compressed_index[class_type].items():
                word_ids = {w["public_id"] for w in words}

                original_cats = []
                for orig_cat, orig_words in original_index.get(class_type, {}).items():
                    orig_word_ids = {w["public_id"] for w in orig_words}

                    if orig_word_ids and orig_word_ids.issubset(word_ids):
                        original_cats.append(orig_cat)

                merge_log[class_type].append(
                    {
                        "final_category": compressed_cat,
                        "merged_from": original_cats,
                        "total_words": len(words),
                        "llm_merged": use_llm_merge,
                    }
                )

        return merge_log

    def _update_categorizations(
        self,
        categorizations: dict[str, dict],
        compressed_index: dict[str, dict[str, list]],
    ) -> list[dict]:
        """Update categorizations with compressed category names.

        Args:
            categorizations: Original categorizations dict.
            compressed_index: Compressed category index.

        Returns:
            Updated categorizations array.
        """
        category_mapping: dict[str, dict[str, str]] = {
            "하위개념": {},
            "기능": {},
            "사용맥락": {},
        }

        for class_type in self.CLASSIFICATION_TYPES:
            if class_type not in compressed_index:
                continue

            for compressed_cat, words in compressed_index[class_type].items():
                for word in words:
                    public_id = word["public_id"]
                    category_mapping[class_type][public_id] = compressed_cat

        categorizations_array = []
        for public_id, data in categorizations.items():
            original_categories = data.get("categories", {})
            updated_categories = {}

            for class_type in self.CLASSIFICATION_TYPES:
                original_cats = original_categories.get(class_type, [])

                compressed_cats = []
                for orig_cat in original_cats:
                    normalized = self._normalize_category_name(orig_cat)
                    if public_id in category_mapping.get(class_type, {}):
                        compressed = category_mapping[class_type][public_id]
                        if compressed not in compressed_cats:
                            compressed_cats.append(compressed)
                    else:
                        if normalized not in compressed_cats:
                            compressed_cats.append(normalized)

                if not compressed_cats:
                    compressed_cats = ["미분류"]

                updated_categories[class_type] = compressed_cats

            categorizations_array.append(
                {
                    "public_id": public_id,
                    "lemma": data.get("lemma", ""),
                    "definition": data.get("definition"),
                    "categories": updated_categories,
                    "original_categories": original_categories,
                    "processed_at": datetime.now().isoformat() + "Z",
                    "model_version": self.model_name,
                }
            )

        return categorizations_array

    def load_categorization_file(self, file_path: Path | str) -> dict[str, Any]:
        """Load categorization results from JSON file.

        Args:
            file_path: Path to word_categorization.json.

        Returns:
            Dictionary with categorizations and category_index.

        Raises:
            CategorizationError: If file not found or invalid.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise CategorizationError(f"Categorization file not found: {file_path}")

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            logger.info(f"Loaded categorization file: {file_path}")
            logger.info(f"Total words: {len(data.get('categorizations', []))}")

            return data

        except Exception as e:
            raise CategorizationError(f"Failed to load categorization file: {e}") from e
