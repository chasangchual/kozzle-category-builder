"""Embedding generation using Ollama API."""

import os
from pathlib import Path
from typing import Any, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from numpy.typing import NDArray
import requests

from kozzle_word_grouper.exceptions import EmbeddingError, OllamaConnectionError
from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.utils import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings using Ollama API."""

    def __init__(
        self,
        model_name: str = "exaone3.5:7.8b",
        ollama_host: str | None = None,
        max_workers: int = 4,
    ) -> None:
        """Initialize embedding generator with Ollama.

        Args:
            model_name: Ollama model name (default: exaone3.5:7.8b).
            ollama_host: Ollama server URL (default: from env or localhost:11434).
            max_workers: Number of concurrent requests for batch processing.

        Raises:
            OllamaConnectionError: If cannot connect to Ollama.
        """
        self.model_name = model_name
        self.ollama_host = ollama_host or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        )
        self.max_workers = max_workers
        self._embedding_dim: int | None = None

        self._validate_connection()

        logger.info(f"Initialized Ollama embedding generator: {model_name}")

    def _validate_connection(self) -> None:
        """Validate Ollama connection."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise OllamaConnectionError(
                    f"Ollama API returned status {response.status_code}"
                )
            logger.info(f"Successfully connected to Ollama at {self.ollama_host}")
        except Exception as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at {self.ollama_host}: {e}"
            ) from e

    def _embed_single_text(self, text: str) -> NDArray[np.floating[Any]]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            response = requests.post(
                f"{self.ollama_host}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()

            if "embedding" not in data:
                raise EmbeddingError(f"No embedding in response: {data}")

            return np.array(data["embedding"], dtype=np.float32)

        except Exception as e:
            raise EmbeddingError(f"Failed to embed text '{text[:50]}...': {e}") from e

    def generate_embeddings(
        self,
        words: list[KoreanWord],
        batch_size: int = 256,
        show_progress: bool = True,
    ) -> NDArray[np.floating[Any]]:
        """Generate embeddings for Korean words.

        Args:
            words: List of KoreanWord objects.
            batch_size: Not used (for compatibility). Ollama processes sequentially.
            show_progress: Whether to show progress bar.

        Returns:
            Numpy array of embeddings with shape (n_words, embedding_dim).

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not words:
            logger.warning("Empty word list provided")
            return np.array([])

        logger.info(f"Generating embeddings for {len(words)} words")

        texts_to_embed = [word.get_text_for_embedding() for word in words]

        embeddings: list[tuple[int, NDArray[np.floating[Any]]]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._embed_single_text, text): i
                for i, text in enumerate(texts_to_embed)
            }

            completed = 0
            for future in as_completed(futures):
                try:
                    embedding = future.result()
                    idx = futures[future]
                    embeddings.append((idx, embedding))
                    completed += 1
                    if show_progress and completed % 10 == 0:
                        logger.info(f"Embedded {completed}/{len(words)} words")
                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
                    raise

        embeddings.sort(key=lambda x: x[0])
        embedding_array = np.vstack([e[1] for e in embeddings])

        self._embedding_dim = embedding_array.shape[1]

        logger.info(f"Generated embeddings with shape: {embedding_array.shape}")
        return embedding_array

    def save_embeddings(
        self,
        embeddings: NDArray[np.floating[Any]],
        output_path: Path | str,
    ) -> None:
        """Save embeddings to a numpy file.

        Args:
            embeddings: Embeddings array to save.
            output_path: Path to save embeddings (will add .npy extension).
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_file, embeddings)
        logger.info(f"Saved embeddings to {output_file}")

    def load_embeddings(self, input_path: Path | str) -> NDArray[np.floating[Any]]:
        """Load embeddings from a numpy file.

        Args:
            input_path: Path to embeddings file.

        Returns:
            Loaded embeddings array.

        Raises:
            EmbeddingError: If loading fails.
        """
        try:
            embeddings = np.load(input_path)
            logger.info(
                f"Loaded embeddings from {input_path} with shape: {embeddings.shape}"
            )
            return cast(NDArray[np.floating[Any]], embeddings)
        except Exception as e:
            raise EmbeddingError(
                f"Failed to load embeddings from {input_path}: {e}"
            ) from e

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension.

        Returns:
            Dimension of embeddings.
        """
        if self._embedding_dim is None:
            dummy = self._embed_single_text("test")
            self._embedding_dim = dummy.shape[0]
        return self._embedding_dim

    def compute_similarity(
        self,
        embedding1: NDArray[np.floating[Any]],
        embedding2: NDArray[np.floating[Any]],
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding.
            embedding2: Second embedding.

        Returns:
            Cosine similarity score.
        """
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
