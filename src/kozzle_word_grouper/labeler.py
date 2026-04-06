"""Generate Korean cluster labels using Ollama."""

import hashlib
import json
from pathlib import Path

import requests

from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class ClusterLabeler:
    """Generate Korean category names for word clusters using Ollama."""

    def __init__(
        self,
        ollama_host: str = "http://localhost:11434",
        model_name: str = "exaone3.5:7.8b",
        cache_file: Path | str | None = None,
    ) -> None:
        """Initialize cluster labeler.

        Args:
            ollama_host: Ollama server URL.
            model_name: Ollama model name.
            cache_file: Path to cache file for labels (optional).
        """
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.cache_file = Path(cache_file) if cache_file else None
        self._cache: dict[str, str] = {}

        if self.cache_file:
            self._load_cache()

        logger.info(f"Initialized cluster labeler with model: {model_name}")

    def _load_cache(self) -> None:
        """Load label cache from file."""
        if self.cache_file and self.cache_file.exists():
            try:
                with open(self.cache_file, encoding="utf-8") as f:
                    self._cache = json.load(f)
                logger.info(f"Loaded {len(self._cache)} cached labels")
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save label cache to file."""
        if self.cache_file:
            try:
                self.cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_file, "w", encoding="utf-8") as f:
                    json.dump(self._cache, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(self._cache)} cached labels")
            except Exception as e:
                logger.error(f"Failed to save cache: {e}")

    def _generate_label_with_ollama(
        self,
        words: list[dict[str, str]],
    ) -> str:
        """Generate Korean cluster label using Ollama.

        Args:
            words: List of word dicts with 'lemma' and 'definition'.

        Returns:
            Korean category name.

        Raises:
            LabelGenerationError: If label generation fails.
        """
        word_list = ", ".join([w["lemma"] for w in words[:20]])

        prompt = f"""다음 단어들을 가장 잘 나타내는 한국어 카테고리 이름을 한 단어로 답하세요. 단어만 답변하세요:

단어들: {word_list}

카테고리:"""

        try:
            response = requests.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            label = data.get("response", "").strip()

            label = label.split("\n")[0].strip()
            label = label.replace('"', "").replace("'", "")

            if not label:
                logger.warning("Empty label generated, using fallback")
                return f"클러스터_{hash(tuple(w['lemma'] for w in words[:5])) % 1000}"

            logger.info(f"Generated label: {label}")
            return label

        except Exception as e:
            logger.error(f"Failed to generate label: {e}")
            return f"클러스터_{hash(tuple(w['lemma'] for w in words[:5])) % 1000}"

    def generate_label(
        self,
        words: list[dict[str, str]],
        cluster_id: int,
    ) -> str:
        """Generate Korean label for cluster, using cache if available.

        Args:
            words: List of word dicts with 'lemma' and 'definition'.
            cluster_id: Cluster ID.

        Returns:
            Korean category name.
        """
        word_key = hashlib.md5(
            json.dumps([w["lemma"] for w in words[:10]], sort_keys=True).encode()
        ).hexdigest()

        if word_key in self._cache:
            logger.info(
                f"Using cached label for cluster {cluster_id}: {self._cache[word_key]}"
            )
            return self._cache[word_key]

        label = self._generate_label_with_ollama(words)

        self._cache[word_key] = label
        if self.cache_file:
            self._save_cache()

        return label

    def label_clusters(
        self,
        clusters: dict[int, list[dict[str, str]]],
    ) -> dict[int, str]:
        """Generate labels for all clusters.

        Args:
            clusters: Dict mapping cluster_id to list of words.

        Returns:
            Dict mapping cluster_id to Korean label.
        """
        labels = {}
        for cluster_id, words in clusters.items():
            label = self.generate_label(words, cluster_id)
            labels[cluster_id] = label
            logger.info(f"Cluster {cluster_id}: {label} ({len(words)} words)")

        return labels
