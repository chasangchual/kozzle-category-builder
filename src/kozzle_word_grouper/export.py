"""Export grouped words to various formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from kozzle_word_grouper.exceptions import ExportError
from kozzle_word_grouper.utils import ensure_directory, get_logger

logger = get_logger(__name__)


def convert_to_native_types(obj: Any) -> Any:
    """Convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object that may contain numpy types.

    Returns:
        Object with numpy types converted to Python types.
    """
    if isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class WordGroupExporter:
    """Export word groups to files."""

    def __init__(self, output_dir: Path | str = "./output") -> None:
        """Initialize exporter.

        Args:
            output_dir: Directory to save output files.
        """
        self.output_dir = Path(output_dir)
        ensure_directory(self.output_dir)
        logger.info(f"Output directory: {self.output_dir}")

    def export_to_json(
        self,
        cluster_info: dict[int, dict[str, Any]],
        filename: str = "word_groups.json",
        include_words: bool = True,
        pretty: bool = True,
    ) -> Path:
        """Export word groups to JSON file.

        Args:
            cluster_info: Dictionary with cluster information.
            filename: Output filename.
            include_words: Whether to include full word lists.
            pretty: Whether to pretty-print JSON.

        Returns:
            Path to exported file.

        Raises:
            ExportError: If export fails.
        """
        output_path = self.output_dir / filename

        try:
            # Prepare output data - use Korean labels as keys
            output_data = {}

            for cluster_id, info in cluster_info.items():
                # Use Korean label as key
                korean_label = info.get("label", f"cluster_{cluster_id}")

                output_data[korean_label] = {
                    "cluster_id": int(cluster_id),  # Convert numpy int64 to Python int
                    "word_count": int(
                        info.get("word_count", len(info.get("words", [])))
                    ),
                }

                if include_words:
                    # For Korean word output, words is list of {"public_id": ..., "lemma": ...}
                    output_data[korean_label]["words"] = info.get("words", [])

                if "representative_words" in info:
                    output_data[korean_label]["representative_words"] = info[
                        "representative_words"
                    ]

            # Convert all numpy types to native Python types
            output_data = convert_to_native_types(output_data)

            # Write JSON
            with open(output_path, "w", encoding="utf-8") as f:
                if pretty:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(output_data, f, ensure_ascii=False)

            logger.info(f"Exported JSON to {output_path}")
            return output_path

        except Exception as e:
            raise ExportError(f"Failed to export JSON: {e}") from e

    def export_to_csv(
        self,
        cluster_info: dict[int, dict[str, Any]],
        filename: str = "word_groups.csv",
    ) -> Path:
        """Export word groups to CSV file.

        Args:
            cluster_info: Dictionary with cluster information.
            filename: Output filename.

        Returns:
            Path to exported file.

        Raises:
            ExportError: If export fails.
        """
        output_path = self.output_dir / filename

        try:
            # Prepare DataFrame
            # For Korean words, format: group_name, public_id, lemma
            rows: list[dict[str, Any]] = []

            for cluster_id, info in cluster_info.items():
                words_list = info.get("words", [])
                label = info.get("label", f"cluster_{cluster_id}")

                for word in words_list:
                    # Check if word is dict (Korean word) or str (legacy)
                    if isinstance(word, dict):
                        rows.append(
                            {
                                "group_name": label,
                                "public_id": word.get("public_id", ""),
                                "lemma": word.get("lemma", ""),
                            }
                        )
                    else:
                        # Legacy format - just word string
                        rows.append(
                            {
                                "group_name": label,
                                "word": word,
                                "cluster_id": cluster_id,
                            }
                        )

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False, encoding="utf-8")

            logger.info(f"Exported CSV to {output_path}")
            return output_path

        except Exception as e:
            raise ExportError(f"Failed to export CSV: {e}") from e

    def export_summary(
        self,
        cluster_info: dict[int, dict[str, Any]],
        quality_metrics: dict[str, float] | None = None,
        filename: str = "cluster_summary.txt",
    ) -> Path:
        """Export cluster summary to text file.

        Args:
            cluster_info: Dictionary with cluster information.
            quality_metrics: Optional quality metrics from clustering.
            filename: Output filename.

        Returns:
            Path to exported file.

        Raises:
            ExportError: If export fails.
        """
        output_path = self.output_dir / filename

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write("한국어 단어 그룹 요약\n")
                f.write("Korean Word Group Summary\n")
                f.write("=" * 60 + "\n\n")

                if quality_metrics:
                    f.write("품질 지표 (Quality Metrics):\n")
                    f.write("-" * 40 + "\n")
                    f.write(
                        f"  그룹 수 (Number of clusters): {quality_metrics['n_clusters']}\n"
                    )
                    f.write(
                        f"  실루엣 점수 (Silhouette score): "
                        f"{quality_metrics['silhouette_score']:.4f}\n"
                    )
                    f.write(
                        f"  노이즈 비율 (Noise ratio): {quality_metrics['noise_ratio']:.2%}\n"
                    )
                    f.write("\n")

                f.write("그룹 상세 정보 (Cluster Details):\n")
                f.write("-" * 40 + "\n\n")

                # Sort clusters by size
                sorted_clusters = sorted(
                    cluster_info.items(),
                    key=lambda x: x[1].get("word_count", 0),
                    reverse=True,
                )

                for cluster_id, info in sorted_clusters:
                    korean_label = info.get("label", f"cluster_{cluster_id}")
                    word_count = info.get("word_count", len(info.get("words", [])))

                    f.write(f"## {korean_label}\n")
                    f.write(f"   단어 수 (Word count): {word_count}\n")

                    if "representative_words" in info:
                        rep_words = ", ".join(info["representative_words"][:10])
                        f.write(f"   대표 단어 (Representative words): {rep_words}\n")

                    # Show first few words with public_id
                    if "words" in info and len(info["words"]) > 0:
                        words = info["words"][:5]
                        f.write(f"   단어 목록 (Sample words):\n")
                        for word in words:
                            if isinstance(word, dict):
                                f.write(
                                    f"     - {word.get('lemma', '')} "
                                    f"(ID: {word.get('public_id', '')})\n"
                                )
                            else:
                                f.write(f"     - {word}\n")
                        if len(info["words"]) > 5:
                            f.write(f"     ... 외 {len(info['words']) - 5}개\n")

                    f.write("\n")

            logger.info(f"Exported summary to {output_path}")
            return output_path

        except Exception as e:
            raise ExportError(f"Failed to export summary: {e}") from e

    def export_all(
        self,
        cluster_info: dict[int, dict[str, Any]],
        quality_metrics: dict[str, float] | None = None,
        output_format: list[str] | None = None,
    ) -> list[Path]:
        """Export cluster info to all specified formats.

        Args:
            cluster_info: Dictionary with cluster information.
            quality_metrics: Optional quality metrics.
            output_format: List of output formats (json, csv, summary).

        Returns:
            List of paths to exported files.

        Raises:
            ExportError: If export fails.
        """
        if output_format is None:
            output_format = ["json", "csv", "summary"]

        # Convert numpy types to native Python types
        cluster_info = convert_to_native_types(cluster_info)
        if quality_metrics:
            quality_metrics = convert_to_native_types(quality_metrics)

        exported_files: list[Path] = []

        if "json" in output_format:
            path = self.export_to_json(cluster_info, include_words=True)
            exported_files.append(path)

        if "csv" in output_format:
            path = self.export_to_csv(cluster_info)
            exported_files.append(path)

        if "summary" in output_format:
            path = self.export_summary(cluster_info, quality_metrics)
            exported_files.append(path)

        logger.info(f"Exported {len(exported_files)} files")
        return exported_files


def export_categorization_results(
    categorizations: dict[str, dict],
    category_index: dict[str, dict[str, list]],
    statistics: dict[str, dict],
    output_dir: Path | str,
    model_version: str = "exaone3.5:7.8b",
) -> Path:
    """Export categorization results in database-ready format.

    Args:
        categorizations: Dictionary with public_id as key.
        category_index: Category index from aggregator.
        statistics: Statistics from aggregator.
        output_dir: Output directory.
        model_version: Model version string.

    Returns:
        Path to exported JSON file.

    Raises:
        ExportError: If export fails.
    """
    output_dir = Path(output_dir)
    ensure_directory(output_dir)

    output_path = output_dir / "word_categorization.json"

    try:
        categorizations_array = []
        for public_id, data in categorizations.items():
            categorizations_array.append(
                {
                    "public_id": public_id,
                    "lemma": data["lemma"],
                    "definition": data["definition"],
                    "categories": data["categories"],
                    "processed_at": datetime.now().isoformat() + "Z",
                    "model_version": model_version,
                }
            )

        output_data = {
            "version": "1.0",
            "metadata": {
                "total_words": len(categorizations),
                "processed_words": len(categorizations),
                "processed_at": datetime.now().isoformat() + "Z",
                "model": model_version,
                "classification_types": ["하위개념", "기능", "사용맥락"],
                "schema_version": "1.0",
            },
            "categorizations": categorizations_array,
            "category_index": category_index,
            "statistics": statistics,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported categorization results to {output_path}")
        return output_path

    except Exception as e:
        raise ExportError(f"Failed to export categorization results: {e}") from e


def export_compressed_categories(
    result: dict[str, Any],
    output_dir: Path | str,
    model_version: str = "exaone3.5:7.8b",
    use_llm_merge: bool = True,
    cycle_number: int | None = None,
) -> Path:
    """Export compressed categories to JSON file.

    Args:
        result: Result from CategoryCompressor.compress_categories().
        output_dir: Output directory.
        model_version: Model version string.
        use_llm_merge: Whether LLM merging was used.
        cycle_number: Cycle number if running multi-cycle compression.

    Returns:
        Path to exported JSON file.

    Raises:
        ExportError: If export fails.
    """
    output_dir = Path(output_dir)
    ensure_directory(output_dir)

    if cycle_number:
        filename = f"compressed_categories_cycle_{cycle_number}.json"
    else:
        filename = "compressed_categories.json"

    output_path = output_dir / filename

    try:
        compression_ratio = {}
        original_stats = result.get("original_stats", {})
        compressed_stats = result.get("statistics", {})

        for class_type in ["하위개념", "기능", "사용맥락"]:
            orig_count = original_stats.get(class_type, {}).get("total_categories", 0)
            comp_count = compressed_stats.get(class_type, {}).get("total_categories", 0)

            if orig_count > 0:
                ratio = (orig_count - comp_count) / orig_count
                compression_ratio[class_type] = round(ratio, 2)
            else:
                compression_ratio[class_type] = 0.0

        compressed_categories = {}
        for class_type in ["하위개념", "기능", "사용맥락"]:
            compressed_categories[class_type] = []
            if class_type in result.get("compressed_index", {}):
                for category, words in result["compressed_index"][class_type].items():
                    compressed_categories[class_type].append(
                        {
                            "category": category,
                            "word_count": len(words),
                            "words": words[:10],
                            "all_words_count": len(words),
                        }
                    )

        metadata = {
            "compression_timestamp": datetime.now().isoformat() + "Z",
            "model": model_version,
            "use_llm_merge": use_llm_merge,
            "original_categories": {
                class_type: original_stats.get(class_type, {}).get(
                    "total_categories", 0
                )
                for class_type in ["하위개념", "기능", "사용맥락"]
            },
            "compressed_categories": {
                class_type: compressed_stats.get(class_type, {}).get(
                    "total_categories", 0
                )
                for class_type in ["하위개념", "기능", "사용맥락"]
            },
            "compression_ratio": compression_ratio,
        }

        if cycle_number:
            metadata["cycle_number"] = cycle_number
            cycle_info = result.get("cycle_info", {})
            if cycle_info:
                metadata["total_cycles"] = cycle_info.get("total_cycles", cycle_number)
                if "cycle_stats" in cycle_info:
                    metadata["cycle_stats"] = cycle_info["cycle_stats"]

        output_data = {
            "version": "2.0",
            "metadata": metadata,
            "compressed_categories": compressed_categories,
            "merge_log": result.get("merge_log", {}),
            "statistics": compressed_stats,
            "categorizations": result.get("categorizations", []),
        }

        output_data = convert_to_native_types(output_data)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Exported compressed categories to {output_path}")
        return output_path

    except Exception as e:
        raise ExportError(f"Failed to export compressed categories: {e}") from e
