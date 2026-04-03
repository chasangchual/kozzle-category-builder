"""Aggregate word categorizations by category type."""

from typing import Any

from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class CategoryAggregator:
    """Aggregate word categorizations by category type."""

    def aggregate(self, categorizations: dict[str, dict]) -> dict[str, Any]:
        """Group words by categories for each classification type.

        Args:
            categorizations: Dictionary with public_id as key:
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

        Returns:
            Dictionary with category index and statistics:
                {
                    "category_index": {
                        "하위개념": {
                            "동물": [
                                {"public_id": "...", "lemma": "개"},
                                ...
                            ],
                            ...
                        },
                        ...
                    },
                    "statistics": {
                        "하위개념": {
                            "total_categories": 50,
                            "avg_words_per_category": 117.5,
                            "max_words": 500,
                            "min_words": 1,
                            "singleton_categories": 10,
                            "top_10_categories": [...]
                        },
                        ...
                    }
                }
        """
        logger.info("Aggregating categories...")

        category_index = {"하위개념": {}, "기능": {}, "사용맥락": {}}

        for public_id, data in categorizations.items():
            for class_type, categories in data["categories"].items():
                for category in categories:
                    if category not in category_index[class_type]:
                        category_index[class_type][category] = []

                    category_index[class_type][category].append(
                        {"public_id": public_id, "lemma": data["lemma"]}
                    )

        statistics = self._calculate_statistics(category_index)

        logger.info(
            f"Aggregation complete: "
            f"{len(category_index['하위개념'])} 하위개념 categories, "
            f"{len(category_index['기능'])} 기능 categories, "
            f"{len(category_index['사용맥락'])} 사용맥락 categories"
        )

        return {"category_index": category_index, "statistics": statistics}

    def _calculate_statistics(
        self, category_index: dict[str, dict[str, list]]
    ) -> dict[str, dict]:
        """Calculate statistics for each classification type.

        Args:
            category_index: Category index from aggregate().

        Returns:
            Dictionary with statistics for each classification type.
        """
        statistics = {}

        for class_type in ["하위개념", "기능", "사용맥락"]:
            categories = category_index[class_type]

            if not categories:
                statistics[class_type] = {
                    "total_categories": 0,
                    "avg_words_per_category": 0,
                    "max_words": 0,
                    "min_words": 0,
                    "singleton_categories": 0,
                    "top_10_categories": [],
                }
                continue

            word_counts = [len(words) for words in categories.values()]

            total_categories = len(categories)
            avg_words = sum(word_counts) / len(word_counts)
            max_words = max(word_counts)
            min_words = min(word_counts)
            singleton_count = sum(1 for count in word_counts if count == 1)

            sorted_categories = sorted(
                categories.items(), key=lambda x: len(x[1]), reverse=True
            )
            top_10 = [
                {"category": cat, "count": len(words)}
                for cat, words in sorted_categories[:10]
            ]

            statistics[class_type] = {
                "total_categories": total_categories,
                "avg_words_per_category": round(avg_words, 2),
                "max_words": max_words,
                "min_words": min_words,
                "singleton_categories": singleton_count,
                "top_10_categories": top_10,
            }

        return statistics

    def get_words_in_category(
        self,
        category_index: dict[str, dict[str, list]],
        category: str,
        classification_type: str,
    ) -> list[dict]:
        """Get all words in a specific category.

        Args:
            category_index: Category index from aggregate().
            category: Category name.
            classification_type: One of '하위개념', '기능', '사용맥락'.

        Returns:
            List of word dictionaries with public_id and lemma.
        """
        if classification_type not in category_index:
            logger.warning(f"Unknown classification type: {classification_type}")
            return []

        if category not in category_index[classification_type]:
            logger.warning(f"Category '{category}' not found in {classification_type}")
            return []

        return category_index[classification_type][category]

    def get_all_categories(
        self, category_index: dict[str, dict[str, list]], classification_type: str
    ) -> list[str]:
        """Get all categories for a classification type.

        Args:
            category_index: Category index from aggregate().
            classification_type: One of '하위개념', '기능', '사용맥락'.

        Returns:
            List of category names.
        """
        if classification_type not in category_index:
            logger.warning(f"Unknown classification type: {classification_type}")
            return []

        return list(category_index[classification_type].keys())

    def get_multi_category_words(
        self,
        categorizations: dict[str, dict],
        classification_type: str,
        min_categories: int = 2,
    ) -> list[dict]:
        """Get words that belong to multiple categories.

        Args:
            categorizations: Categorizations dict.
            classification_type: One of '하위개념', '기능', '사용맥락'.
            min_categories: Minimum number of categories.

        Returns:
            List of word dictionaries with public_id, lemma, and categories.
        """
        multi_cat_words = []

        for public_id, data in categorizations.items():
            categories = data["categories"].get(classification_type, [])
            if len(categories) >= min_categories:
                multi_cat_words.append(
                    {
                        "public_id": public_id,
                        "lemma": data["lemma"],
                        "categories": categories,
                    }
                )

        return multi_cat_words

    def find_similar_categories(
        self,
        category_index: dict[str, dict[str, list]],
        classification_type: str,
        similarity_threshold: float = 0.7,
    ) -> list[tuple[str, str, float]]:
        """Find similar categories that might need merging in post-processing.

        Args:
            category_index: Category index from aggregate().
            classification_type: One of '하위개념', '기능', '사용망락'.
            similarity_threshold: Minimum Jaccard similarity (0-1).

        Returns:
            List of (category1, category2, similarity) tuples.
        """
        if classification_type not in category_index:
            return []

        categories = category_index[classification_type]
        similar_pairs = []

        cat_names = list(categories.keys())
        for i in range(len(cat_names)):
            for j in range(i + 1, len(cat_names)):
                cat1, cat2 = cat_names[i], cat_names[j]
                words1 = set(w["public_id"] for w in categories[cat1])
                words2 = set(w["public_id"] for w in categories[cat2])

                if not words1 or not words2:
                    continue

                intersection = len(words1 & words2)
                union = len(words1 | words2)
                similarity = intersection / union if union > 0 else 0

                if similarity >= similarity_threshold:
                    similar_pairs.append((cat1, cat2, similarity))

        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs
