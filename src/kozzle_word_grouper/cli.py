"""CLI interface for kozzle-word-grouper."""

import logging
from pathlib import Path

import click
from dotenv import load_dotenv

from kozzle_word_grouper import __version__
from kozzle_word_grouper.core import WordGrouperPipeline
from kozzle_word_grouper.exceptions import WordGrouperError
from kozzle_word_grouper.utils import setup_logging


@click.group()
@click.version_option(version=__version__)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def main(verbose: bool) -> None:
    """Kozzle Word Grouper - Group Korean words by semantic meaning using Ollama."""
    load_dotenv()

    log_level = logging.DEBUG if verbose else logging.INFO
    setup_logging(log_level)


@main.command()
@click.option(
    "--table",
    "-t",
    default="kor_word",
    help="Name of the table containing Korean words",
)
@click.option(
    "--lemma-column",
    "-l",
    default="lemma",
    help="Name of the lemma column",
)
@click.option(
    "--definition-column",
    "-d",
    default="definition",
    help="Name of the definition column",
)
@click.option(
    "--public-id-column",
    "-p",
    default="public_id",
    help="Name of the public ID column",
)
@click.option(
    "--level-column",
    default="level",
    help="Name of the level column for filtering",
)
@click.option(
    "--filter-level",
    "-f",
    multiple=True,
    type=int,
    help="Filter by level (e.g., -f 1 -f 2 for level 1 or 2)",
)
@click.option(
    "--min-lemma-length",
    "-m",
    type=int,
    help="Minimum lemma length (e.g., 2 for >= 2 characters)",
)
@click.option(
    "--model",
    default="exaone3.5:7.8b",
    help="Ollama model for embeddings",
)
@click.option(
    "--min-cluster-size",
    "-c",
    default=10,
    help="Minimum words per cluster (adjust to get ~100 clusters)",
)
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Directory to save output files",
)
@click.option(
    "--ollama-host",
    default=None,
    help="Ollama server URL (default: from env or localhost:11434)",
)
@click.option(
    "--output-format",
    multiple=True,
    type=click.Choice(["json", "csv", "summary"], case_sensitive=False),
    default=["json", "csv", "summary"],
    help="Output format(s) for results",
)
def group(
    table: str,
    lemma_column: str,
    definition_column: str,
    public_id_column: str,
    level_column: str,
    filter_level: tuple[int, ...],
    min_lemma_length: int | None,
    model: str,
    min_cluster_size: int,
    output_dir: str,
    ollama_host: str | None,
    output_format: tuple[str, ...],
) -> None:
    """Group Korean words by semantic meaning using Ollama embeddings.

    This command will:
    1. Fetch words from Supabase (kor_word table)
    2. Filter by level and lemma length if specified
    3. Generate embeddings using Ollama (exaone3.5:7.8b)
    4. Cluster words using HDBSCAN
    5. Generate Korean cluster labels using Ollama
    6. Export results with Korean group names
    """
    logger = logging.getLogger(__name__)

    try:
        # Step 1: Initialize pipeline
        click.echo("Initializing Korean word grouper...")
        pipeline = WordGrouperPipeline(
            model_name=model,
            min_cluster_size=min_cluster_size,
            output_dir=Path(output_dir),
            ollama_host=ollama_host,
        )

        # Step 2: Run pipeline
        click.echo(f"Fetching Korean words from table '{table}'...")

        # Convert filter_level tuple to list
        level_filter = list(filter_level) if filter_level else None

        if level_filter:
            click.echo(f"Filtering by levels: {level_filter}")
        if min_lemma_length is not None:
            click.echo(f"Filtering by lemma length >= {min_lemma_length}")

        click.echo(f"Using Ollama model: {model}")
        click.echo(f"Minimum cluster size: {min_cluster_size}")

        output_formats = list(output_format) if output_format else None

        result = pipeline.run(
            table_name=table,
            lemma_column=lemma_column,
            definition_column=definition_column,
            public_id_column=public_id_column,
            level_column=level_column,
            filter_level=level_filter,
            min_lemma_length=min_lemma_length,
            show_progress=True,
            output_format=output_formats,
        )

        # Step 3: Show results summary
        click.echo("\n" + "=" * 60)
        click.echo("한국어 단어 그룹화 완료 (Korean Word Grouping Complete)")
        click.echo("=" * 60 + "\n")

        total_words = sum(info.get("word_count", 0) for info in result.values())
        click.echo(f"Total clusters: {len(result)}")
        click.echo(f"Total words grouped: {total_words}")

        # Show sample clusters
        click.echo("\nSample clusters:")
        for i, (cluster_id, cluster_data) in enumerate(list(result.items())[:5]):
            korean_label = cluster_data.get("label", f"cluster_{cluster_id}")
            word_count = cluster_data.get("word_count", 0)
            click.echo(f"  - {korean_label}: {word_count} words")

        click.echo("\n✓ Done! Check output files in: " + output_dir)

    except WordGrouperError as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Word grouper error")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        logger.exception("Unexpected error")
        raise SystemExit(1)


@main.command()
@click.option(
    "--table",
    "-t",
    default="kor_word",
    help="Name of the table containing Korean words",
)
@click.option(
    "--lemma-column",
    "-l",
    default="lemma",
    help="Name of the lemma column",
)
@click.option(
    "--definition-column",
    "-d",
    default="definition",
    help="Name of the definition column",
)
@click.option(
    "--public-id-column",
    "-p",
    default="public_id",
    help="Name of the public ID column",
)
@click.option(
    "--level-column",
    default="level",
    help="Name of the level column for filtering",
)
@click.option(
    "--filter-level",
    "-f",
    multiple=True,
    type=int,
    help="Filter by level (e.g., -f 1 -f 2 for level 1 or 2)",
)
@click.option(
    "--min-lemma-length",
    "-m",
    type=int,
    help="Minimum lemma length (e.g., 2 for >= 2 characters)",
)
@click.option(
    "--model",
    default="exaone3.5:7.8b",
    help="Ollama model for categorization",
)
@click.option(
    "--output-dir",
    "-o",
    default="./output",
    help="Directory to save output files",
)
@click.option(
    "--ollama-host",
    default=None,
    help="Ollama server URL (default: from env or localhost:11434)",
)
@click.option(
    "--subset",
    type=int,
    help="Number of words to process (for testing, None for all)",
)
@click.option(
    "--resume",
    is_flag=True,
    default=True,
    help="Resume from cache if available",
)
@click.option(
    "--from-cache",
    is_flag=True,
    default=False,
    help="Load words from local cache instead of Supabase",
)
@click.option(
    "--cache-file",
    type=click.Path(),
    default=None,
    help="Path to cache file (default: output/words_cache.json)",
)
def categorize(
    table: str,
    lemma_column: str,
    definition_column: str,
    public_id_column: str,
    level_column: str,
    filter_level: tuple[int, ...],
    min_lemma_length: int | None,
    model: str,
    output_dir: str,
    ollama_host: str | None,
    subset: int | None,
    resume: bool,
    from_cache: bool,
    cache_file: str | None,
) -> None:
    """Categorize Korean words using LLM-based classification.

    This command will:
    1. Fetch words from Supabase (kor_word table) or from local cache
    2. Filter by level and lemma length if specified
    3. Categorize each word using LLM (3 questions per word)
    4. Aggregate categories by classification type
    5. Export results in database-ready format

    Classification types:
    - 하위개념 (Hyponym): Classify by subordinate concepts
    - 기능 (Function): Classify by function/role
    - 사용맥락 (Usage Context): Classify by usage context

    Use --from-cache to load words from a previously cached file instead of
    fetching from Supabase. This is useful for rerunning categorization with
    different parameters without re-downloading the word list.
    """
    logger = logging.getLogger(__name__)

    try:
        # Step 1: Initialize pipeline
        click.echo("Initializing Korean word categorizer...")
        pipeline = WordGrouperPipeline(
            model_name=model,
            output_dir=Path(output_dir),
            ollama_host=ollama_host,
        )

        # Step 2: Run categorization
        if from_cache:
            click.echo("Loading Korean words from local cache...")
        else:
            click.echo(f"Fetching Korean words from table '{table}'...")

        # Convert filter_level tuple to list
        level_filter = list(filter_level) if filter_level else None

        if not from_cache:
            if level_filter:
                click.echo(f"Filtering by levels: {level_filter}")
            if min_lemma_length is not None:
                click.echo(f"Filtering by lemma length >= {min_lemma_length}")
        if subset is not None:
            click.echo(f"Processing subset of {subset} words")

        click.echo(f"Using Ollama model: {model}")
        click.echo(f"Resume from cache: {'Yes' if resume else 'No'}")
        click.echo(f"Load from cache: {'Yes' if from_cache else 'No'}")

        cache_file_path = Path(cache_file) if cache_file else None

        result = pipeline.run_categorization(
            table_name=table,
            lemma_column=lemma_column,
            definition_column=definition_column,
            public_id_column=public_id_column,
            level_column=level_column,
            filter_level=level_filter,
            min_lemma_length=min_lemma_length,
            subset=subset,
            resume=resume,
            show_progress=True,
            from_cache=from_cache,
            cache_file=cache_file_path,
        )

        # Step 3: Show results summary
        click.echo("\n" + "=" * 60)
        click.echo("한국어 단어 분류 완료 (Korean Word Categorization Complete)")
        click.echo("=" * 60 + "\n")

        stats = result["statistics"]

        click.echo("Classification Statistics:\n")
        for class_type in ["하위개념", "기능", "사용맥락"]:
            type_stats = stats.get(class_type, {})
            click.echo(f"  {class_type}:")
            click.echo(f"    Total categories: {type_stats.get('total_categories', 0)}")
            click.echo(
                f"    Avg words/category: "
                f"{type_stats.get('avg_words_per_category', 0):.1f}"
            )
            click.echo(f"    Max words: {type_stats.get('max_words', 0)}")
            click.echo(f"    Min words: {type_stats.get('min_words', 0)}")
            click.echo()

        click.echo("Top Categories:\n")
        for class_type in ["하위개념", "기능", "사용맥락"]:
            type_stats = stats.get(class_type, {})
            top_10 = type_stats.get("top_10_categories", [])
            if top_10:
                click.echo(f"  {class_type}:")
                for item in top_10[:5]:
                    click.echo(f"    - {item['category']}: {item['count']} words")
                click.echo()

        click.echo("✓ Done! Output file: " + str(result["output_path"]))

    except WordGrouperError as e:
        click.echo(f"Error: {e}", err=True)
        logger.exception("Word grouper error")
        raise SystemExit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {e}", err=True)
        logger.exception("Unexpected error")
        raise SystemExit(1)


@main.command()
def info() -> None:
    """Display information about Ollama models and configuration."""
    click.echo("Kozzle Word Grouper - Korean Word Clustering\n")

    click.echo("Ollama Models:")
    click.echo("  - exaone3.5:7.8b: Korean language model (default)")
    click.echo("  - Other models: Use 'ollama list' to see available models\n")

    click.echo("Environment Variables:")
    click.echo("  SUPABASE_URL: Your Supabase project URL")
    click.echo("  SUPABASE_KEY: Your Supabase anon/service key")
    click.echo("  OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)")
    click.echo("  OLLAMA_MODEL: Model for embeddings (default: exaone3.5:7.8b)\n")

    click.echo("Connection Pool Settings:")
    click.echo("  SUPABASE_MAX_CONNECTIONS: Max connections (default: 10)")
    click.echo("  SUPABASE_IDLE_TIMEOUT: Idle timeout in seconds (default: 0.1)")
    click.echo("  SUPABASE_MAX_RETRIES: Retry attempts (default: 3)\n")

    click.echo("Output Formats:")
    click.echo("  - json: JSON file with cluster information")
    click.echo("  - csv: CSV file with word-cluster mappings")
    click.echo("  - summary: Text summary of clusters\n")


@main.command()
def pool_info() -> None:
    """Display connection pool configuration and statistics."""
    from kozzle_word_grouper.monitoring import log_connection_pool_stats

    click.echo("Connection Pool Configuration:\n")
    log_connection_pool_stats()


if __name__ == "__main__":
    main()
