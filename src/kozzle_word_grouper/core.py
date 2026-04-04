"""Core pipeline for Korean word grouping."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from kozzle_word_grouper.categorizer import Categorizer
from kozzle_word_grouper.category_aggregator import CategoryAggregator
from kozzle_word_grouper.category_compressor import CategoryCompressor
from kozzle_word_grouper.clustering import WordClusterer
from kozzle_word_grouper.connection_pool import close_connection_pool
from kozzle_word_grouper.embeddings import EmbeddingGenerator
from kozzle_word_grouper.exceptions import WordGrouperError
from kozzle_word_grouper.export import (
    WordGroupExporter,
    export_categorization_results,
    export_compressed_categories,
)
from kozzle_word_grouper.labeler import ClusterLabeler
from kozzle_word_grouper.models import KoreanWord
from kozzle_word_grouper.supabase_client import SupabaseClient
from kozzle_word_grouper.utils import ensure_directory, get_logger

logger = get_logger(__name__)


class WordGrouperPipeline:
    """Complete pipeline for Korean word grouping with Ollama."""

    def __init__(
        self,
        model_name: str = "exaone3.5:7.8b",
        min_cluster_size: int = 10,
        output_dir: Path | str = "./output",
        ollama_host: str | None = None,
    ) -> None:
        """Initialize the word grouper pipeline.

        Args:
            model_name: Name of the Ollama model.
            min_cluster_size: Minimum number of words per cluster.
            ollama_host: Ollama server URL (default: from env or localhost:11434).
            output_dir: Directory for output files.
        """
        self.model_name = model_name
        self.min_cluster_size = min_cluster_size
        self.output_dir = Path(output_dir)
        self.ollama_host = ollama_host or os.getenv(
            "OLLAMA_HOST", "http://localhost:11434"
        )

        self.supabase_client: SupabaseClient | None = None
        self.embedding_generator = EmbeddingGenerator(
            model_name=model_name,
            ollama_host=self.ollama_host,
        )
        self.clusterer = WordClusterer(min_cluster_size=min_cluster_size)
        self.labeler = ClusterLabeler(
            ollama_host=self.ollama_host,
            model_name=model_name,
            cache_file=self.output_dir / "label_cache.json",
        )
        self.exporter = WordGroupExporter(output_dir=self.output_dir)

        logger.info(f"Initialized WordGrouperPipeline with model: {model_name}")

    def fetch_korean_words(
        self,
        table_name: str = "kor_word",
        lemma_column: str = "lemma",
        definition_column: str = "definition",
        public_id_column: str = "public_id",
        level_column: str = "level",
        filter_level: list[int] | None = None,
        min_lemma_length: int | None = None,
    ) -> list[KoreanWord]:
        """Fetch Korean words from Supabase.

        Args:
            table_name: Name of the table.
            lemma_column: Name of the lemma column.
            definition_column: Name of the definition column.
            public_id_column: Name of the public ID column.
            level_column: Name of the level column for filtering.
            filter_level: List of levels to include (e.g., [1, 2] for level 1 or 2).
            min_lemma_length: Minimum lemma length (e.g., 2 for >= 2 characters).

        Returns:
            List of KoreanWord objects.
        """
        if self.supabase_client is None:
            self.supabase_client = SupabaseClient()

        return self.supabase_client.fetch_korean_words(
            table_name=table_name,
            lemma_column=lemma_column,
            definition_column=definition_column,
            public_id_column=public_id_column,
            level_column=level_column,
            filter_level=filter_level,
            min_lemma_length=min_lemma_length,
        )

    def fetch_and_cache_korean_words(
        self,
        cache_file: Path | str,
        table_name: str = "kor_word",
        lemma_column: str = "lemma",
        definition_column: str = "definition",
        public_id_column: str = "public_id",
        level_column: str = "level",
        filter_level: list[int] | None = None,
        min_lemma_length: int | None = None,
    ) -> list[KoreanWord]:
        """Fetch Korean words from Supabase and cache to local file.

        Args:
            cache_file: Path to cache file.
            table_name: Name of the table.
            lemma_column: Name of the lemma column.
            definition_column: Name of the definition column.
            public_id_column: Name of the public ID column.
            level_column: Name of the level column for filtering.
            filter_level: List of levels to include.
            min_lemma_length: Minimum lemma length.

        Returns:
            List of KoreanWord objects.
        """
        logger.info("Fetching Korean words from Supabase and caching to local file")

        words = self.fetch_korean_words(
            table_name=table_name,
            lemma_column=lemma_column,
            definition_column=definition_column,
            public_id_column=public_id_column,
            level_column=level_column,
            filter_level=filter_level,
            min_lemma_length=min_lemma_length,
        )

        self._save_words_to_cache(words, cache_file)

        return words

    def load_korean_words_from_cache(
        self,
        cache_file: Path | str,
    ) -> list[KoreanWord]:
        """Load Korean words from local cache file.

        Args:
            cache_file: Path to cache file.

        Returns:
            List of KoreanWord objects.

        Raises:
            WordGrouperError: If cache file not found or invalid.
        """
        cache_file = Path(cache_file)

        if not cache_file.exists():
            raise WordGrouperError(
                f"Cache file not found: {cache_file}. "
                "Run without --from-cache to fetch from Supabase first."
            )

        try:
            logger.info(f"Loading Korean words from cache: {cache_file}")

            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            words = []
            for word_data in data.get("words", []):
                words.append(
                    KoreanWord(
                        public_id=word_data["public_id"],
                        lemma=word_data["lemma"],
                        definition=word_data.get("definition"),
                    )
                )

            logger.info(
                f"Loaded {len(words)} words from cache "
                f"(cached at: {data.get('metadata', {}).get('cached_at', 'unknown')})"
            )

            return words

        except Exception as e:
            raise WordGrouperError(f"Failed to load cache file: {e}") from e

    def _save_words_to_cache(
        self,
        words: list[KoreanWord],
        cache_file: Path | str,
    ) -> None:
        """Save Korean words to cache file.

        Args:
            words: List of KoreanWord objects.
            cache_file: Path to cache file.
        """
        cache_file = Path(cache_file)
        ensure_directory(cache_file.parent)

        try:
            words_data = [
                {
                    "public_id": word.public_id,
                    "lemma": word.lemma,
                    "definition": word.definition,
                }
                for word in words
            ]

            cache_data = {
                "metadata": {
                    "cached_at": datetime.now().isoformat(),
                    "total_words": len(words),
                    "source": "supabase",
                },
                "words": words_data,
            }

            temp_file = cache_file.with_suffix(".json.tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            temp_file.replace(cache_file)

            logger.info(f"Cached {len(words)} words to {cache_file}")

        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            # Don't raise - caching is optional

    def run(
        self,
        table_name: str = "kor_word",
        lemma_column: str = "lemma",
        definition_column: str = "definition",
        public_id_column: str = "public_id",
        level_column: str = "level",
        filter_level: list[int] | None = None,
        min_lemma_length: int | None = None,
        show_progress: bool = True,
        output_format: list[str] | None = None,
    ) -> dict[int, dict[str, Any]]:
        """Run the complete pipeline for Korean word grouping.

        Args:
            table_name: Name of the table in Supabase.
            lemma_column: Name of the lemma column.
            definition_column: Name of the definition column.
            public_id_column: Name of the public ID column.
            level_column: Name of the level column for filtering.
            filter_level: List of levels to include (e.g., [1, 2] for level 1 or 2).
            min_lemma_length: Minimum lemma length (e.g., 2 for >= 2 characters).
            show_progress: Whether to show progress bars.
            output_format: List of output formats (json, csv, summary).

        Returns:
            Dictionary with cluster information (Korean labels).

        Raises:
            WordGrouperError: If pipeline fails.
        """
        try:
            # Step 1: Fetch Korean words from Supabase with filters
            logger.info(f"Fetching Korean words from table '{table_name}'")
            if filter_level:
                logger.info(f"Filtering by levels: {filter_level}")
            if min_lemma_length is not None:
                logger.info(f"Filtering by lemma length >= {min_lemma_length}")

            words = self.fetch_korean_words(
                table_name=table_name,
                lemma_column=lemma_column,
                definition_column=definition_column,
                public_id_column=public_id_column,
                level_column=level_column,
                filter_level=filter_level,
                min_lemma_length=min_lemma_length,
            )

            if not words:
                raise WordGrouperError("No words found in table")

            logger.info(f"Fetched {len(words)} words")

            # Step 2: Generate embeddings using Ollama
            logger.info(f"Generating embeddings using {self.model_name} via Ollama")
            embeddings = self.embedding_generator.generate_embeddings(
                words,
                show_progress=show_progress,
            )

            # Step 3: Cluster words using HDBSCAN
            logger.info(
                f"Clustering words with min_cluster_size={self.min_cluster_size}"
            )
            labels = self.clusterer.fit_predict(embeddings)

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            logger.info(f"Found {n_clusters} clusters")

            # Step 4: Generate Korean labels for each cluster
            logger.info("Generating Korean cluster labels")

            # Prepare cluster data for labeling
            clusters = {}
            for cluster_id in set(labels):
                if cluster_id != -1:  # Skip noise
                    clusters[cluster_id] = [
                        {"lemma": words[i].lemma, "definition": words[i].definition}
                        for i in range(len(words))
                        if labels[i] == cluster_id
                    ]

            # Generate labels using Ollama
            korean_labels = self.labeler.label_clusters(clusters)

            # Step 5: Get cluster info with Korean labels
            cluster_info = self.clusterer.get_cluster_info_korean(
                words, labels, korean_labels, embeddings
            )

            # Step 6: Calculate quality metrics
            quality_metrics = self.clusterer.calculate_cluster_quality(
                embeddings, labels
            )

            logger.info(f"Silhouette score: {quality_metrics['silhouette_score']:.4f}")
            logger.info(f"Noise ratio: {quality_metrics['noise_ratio']:.2%}")

            # Step 7: Export results
            logger.info(f"Exporting results to {self.output_dir}")
            exported_files = self.exporter.export_all(
                cluster_info,
                quality_metrics,
                output_format=output_format,
            )

            logger.info(f"Exported {len(exported_files)} files")
            logger.info("Pipeline completed successfully")

            return cluster_info

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise WordGrouperError(f"Pipeline failed: {e}") from e

        finally:
            # Close connection pool
            close_connection_pool()
            logger.info("Connection pool closed")

    def run_categorization(
        self,
        table_name: str = "kor_word",
        lemma_column: str = "lemma",
        definition_column: str = "definition",
        public_id_column: str = "public_id",
        level_column: str = "level",
        filter_level: list[int] | None = None,
        min_lemma_length: int | None = None,
        subset: int | None = None,
        resume: bool = True,
        show_progress: bool = True,
        from_cache: bool = False,
        cache_file: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run LLM-based categorization pipeline.

        Args:
            table_name: Name of the table in Supabase.
            lemma_column: Name of the lemma column.
            definition_column: Name of the definition column.
            public_id_column: Name of the public ID column.
            level_column: Name of the level column for filtering.
            filter_level: List of levels to include (e.g., [1, 2] for level 1 or 2).
            min_lemma_length: Minimum lemma length (e.g., 2 for >= 2 characters).
            subset: Number of words to process (None for all).
            resume: Whether to resume from cache.
            show_progress: Whether to show progress.
            from_cache: Whether to load words from cache instead of Supabase.
            cache_file: Path to cache file (default: output/words_cache.json).

        Returns:
            Dictionary with categorization results.

        Raises:
            WordGrouperError: If pipeline fails.
        """
        try:
            # Determine cache file path
            if cache_file is None:
                cache_file = self.output_dir / "words_cache.json"

            # Step 1: Fetch or load Korean words
            if from_cache:
                logger.info(f"Loading Korean words from cache: {cache_file}")
                words = self.load_korean_words_from_cache(cache_file)
            else:
                logger.info(f"Fetching Korean words from table '{table_name}'")
                if filter_level:
                    logger.info(f"Filtering by levels: {filter_level}")
                if min_lemma_length is not None:
                    logger.info(f"Filtering by lemma length >= {min_lemma_length}")

                words = self.fetch_and_cache_korean_words(
                    cache_file=cache_file,
                    table_name=table_name,
                    lemma_column=lemma_column,
                    definition_column=definition_column,
                    public_id_column=public_id_column,
                    level_column=level_column,
                    filter_level=filter_level,
                    min_lemma_length=min_lemma_length,
                )

            if not words:
                raise WordGrouperError("No words found")

            # Apply subset if specified
            if subset is not None and subset < len(words):
                words = words[:subset]
                logger.info(f"Processing subset of {subset} words")

            logger.info(f"Total words: {len(words)}")

            # Step 2: Categorize words using LLM
            logger.info(f"Categorizing words using {self.model_name} via Ollama")

            categorizer = Categorizer(
                model_name=self.model_name,
                ollama_host=self.ollama_host,
                cache_file=self.output_dir / "categorization_cache.json",
                max_workers=4,
                max_retries=3,
                retry_delay=2.0,
            )

            categorizations = categorizer.categorize_words(
                words,
                show_progress=show_progress,
                resume=resume,
            )

            # Step 3: Aggregate categories
            logger.info("Aggregating categories")
            aggregator = CategoryAggregator()
            aggregated = aggregator.aggregate(categorizations)

            # Step 4: Export results
            logger.info(f"Exporting results to {self.output_dir}")
            output_path = export_categorization_results(
                categorizations=categorizations,
                category_index=aggregated["category_index"],
                statistics=aggregated["statistics"],
                output_dir=self.output_dir,
                model_version=self.model_name,
            )

            logger.info(f"Categorization results exported to {output_path}")
            logger.info("Categorization pipeline completed successfully")

            return {
                "categorizations": categorizations,
                "category_index": aggregated["category_index"],
                "statistics": aggregated["statistics"],
                "output_path": output_path,
            }

        except Exception as e:
            logger.error(f"Categorization pipeline failed: {e}")
            raise WordGrouperError(f"Categorization pipeline failed: {e}") from e

        finally:
            close_connection_pool()
            logger.info("Connection pool closed")

    def run_category_compression(
        self,
        categorization_file: Path | str,
        use_llm_merge: bool = True,
        min_word_count: int | None = None,
        output_dir: Path | str | None = None,
        show_progress: bool = True,
        cycles: int = 1,
    ) -> dict[str, Any]:
        """Run category compression pipeline with multiple cycles.

        Args:
            categorization_file: Path to word_categorization.json.
            use_llm_merge: Whether to use LLM for semantic merging.
            min_word_count: Minimum number of words to keep a category (None = no filter).
            output_dir: Output directory (default: same as input file).
            show_progress: Whether to show progress.
            cycles: Number of compression cycles (default: 1).

        Returns:
            Dictionary with compressed results from final cycle.

        Raises:
            WordGrouperError: If compression fails.
        """
        try:
            categorization_file = Path(categorization_file)

            if output_dir is None:
                output_dir = categorization_file.parent
            else:
                output_dir = Path(output_dir)

            ensure_directory(output_dir)

            compressor = CategoryCompressor(
                model_name=self.model_name,
                ollama_host=self.ollama_host,
                batch_size=50,
                max_retries=3,
                retry_delay=2.0,
            )

            logger.info(f"Loading categorization file: {categorization_file}")
            data = compressor.load_categorization_file(categorization_file)

            categorizations = {}
            for item in data.get("categorizations", []):
                public_id = item.get("public_id")
                categorizations[public_id] = {
                    "lemma": item.get("lemma", ""),
                    "definition": item.get("definition"),
                    "categories": item.get("categories", {}),
                }

            aggregator = CategoryAggregator()

            cycle_stats = []

            category_index = data.get("category_index", {})

            if not category_index:
                logger.info("Category index not found, rebuilding from categorizations")
                aggregated = aggregator.aggregate(categorizations)
                category_index = aggregated["category_index"]

            original_stats = compressor._calculate_statistics(category_index)

            for cycle_num in range(1, cycles + 1):
                logger.info(f"=" * 60)
                logger.info(f"Starting compression cycle {cycle_num}/{cycles}")
                logger.info(f"=" * 60)

                if cycle_num > 1:
                    logger.info(f"Re-aggregating categories from cycle {cycle_num - 1}")
                    aggregated = aggregator.aggregate(categorizations)
                    category_index = aggregated["category_index"]

                logger.info(f"Compressing categories for cycle {cycle_num}")
                if use_llm_merge:
                    logger.info("Using LLM-based semantic merging")
                if min_word_count is not None:
                    logger.info(f"Filtering categories with <{min_word_count} words")

                result = compressor.compress_categories(
                    category_index=category_index,
                    categorizations=categorizations,
                    use_llm_merge=use_llm_merge,
                    min_word_count=min_word_count,
                    show_progress=show_progress,
                )

                categorizations = {}
                for item in result.get("categorizations", []):
                    public_id = item.get("public_id")
                    categorizations[public_id] = {
                        "lemma": item.get("lemma", ""),
                        "definition": item.get("definition"),
                        "categories": item.get("categories", {}),
                    }

                cycle_stat = {
                    "cycle_number": cycle_num,
                    "statistics": result.get("statistics", {}),
                    "use_llm_merge": use_llm_merge,
                    "min_word_count": min_word_count,
                }
                cycle_stats.append(cycle_stat)

                stats = result.get("statistics", {})
                logger.info(f"Cycle {cycle_num} complete:")
                for class_type in ["하위개념", "기능", "사용맥락"]:
                    type_stats = stats.get(class_type, {})
                    logger.info(
                        f"  {class_type}: {type_stats.get('total_categories', 0)} categories, "
                        f"{type_stats.get('total_words', 0)} words"
                    )

                if cycle_num < cycles:
                    logger.info(f"Preparing for cycle {cycle_num + 1}...")

            final_cycle_num = cycles
            result["cycle_info"] = {
                "total_cycles": cycles,
                "cycle_stats": cycle_stats,
                "original_stats": original_stats,
            }

            logger.info(
                f"Exporting final results (cycle {final_cycle_num}) to {output_dir}"
            )
            output_path = export_compressed_categories(
                result=result,
                output_dir=output_dir,
                model_version=self.model_name,
                use_llm_merge=use_llm_merge,
                cycle_number=final_cycle_num,
            )

            logger.info(f"Compressed results exported to {output_path}")
            logger.info(
                f"Category compression pipeline completed successfully ({cycles} cycles)"
            )

            result["output_path"] = output_path
            return result

        except Exception as e:
            logger.error(f"Category compression failed: {e}")
            raise WordGrouperError(f"Category compression failed: {e}") from e
