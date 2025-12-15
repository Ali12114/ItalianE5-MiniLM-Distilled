import os
import csv
import math
import logging
from pathlib import Path
from typing import Dict, List, Any, Union
import torch

import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from transformers import pipeline as hf_pipeline, AutoTokenizer

import shutil

# Import utility functions from utils module
from utils import (
    load_config,
    build_paths,
    ensure_spacy_model,
    split_paragraph_with_token_limit,
    split_paragraph_into_sentences
)


# ------------------------------
# Step 1: Download tiny dataset
# ------------------------------


def step_download(cfg: Dict[str, Any], out_csv: str) -> None:
    """Materialize the raw dataset into a CSV file.

    The step supports two data sources:
    1) If ``download.local_csv`` is configured, it is validated for the
       required text column and then copied to ``out_csv``.
    2) Otherwise, the dataset is fetched from Hugging Face Datasets using
       ``download.hf_dataset`` (with optional ``config_name``, ``split``, and
       ``streaming``). All rows and columns are written to ``out_csv``.

    The step is skipped when the output already exists and ``general.force`` is
    not set to true.

    Args:
        cfg: Configuration dictionary containing at least:
            - ``io.text_column``: Name of the text field expected for downstream task.
            - ``download.local_csv`` (optional): Path to a preexisting CSV.
            - ``download.hf_dataset`` (fallback): Hugging Face dataset identifier.
            - ``download.config_name`` (optional): Dataset config name.
            - ``download.split`` (optional): Split name, defaults to ``train``.
            - ``download.streaming`` (optional): Whether to stream the dataset.
            - ``general.force`` (optional): Recompute even if output exists.
        out_csv: Target path for the CSV to write.

    Returns:
        None. Writes the dataset to ``out_csv``.

    Raises:
        FileNotFoundError: If a configured ``local_csv`` does not exist.
        ValueError: If the configured ``io.text_column`` is missing in
            ``local_csv`` when provided.
        Exception: Propagated from Hugging Face ``load_dataset`` on failures.

        Note: Default settings for optional configuration are best. It should not be changed unless there is a specific reason.
    """
    # Skip if output already exists and force mode is not enabled
    if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
        logging.info(f"Download output exists at {out_csv}; skipping.")
        return

    # Extract download configuration section
    ds_cfg = cfg["download"]
    local_csv = ds_cfg.get("local_csv")

    # Branch 1: Use local CSV file if provided
    if local_csv:
        # Convert to absolute path, expanding ~ for home directory
        local_csv_path = Path(local_csv).expanduser().resolve()
        if not local_csv_path.exists():
            raise FileNotFoundError(
                f"Configured local CSV not found: {local_csv_path}")

        # Validate that the required text column exists in the CSV
        io_text_col = cfg["io"]["text_column"]
        # Read only the header row (nrows=0) to get column names efficiently
        cols = pd.read_csv(str(local_csv_path), nrows=0).columns.tolist()
        if io_text_col not in cols:
            raise ValueError(
                f"Local CSV missing required text column '{io_text_col}'. Columns found: {cols}"
            )

        # Prepare target directory structure
        target_path = Path(out_csv).expanduser().resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy file only if source and destination are different
        if local_csv_path == target_path:
            logging.info(
                f"Local CSV already at target location: {target_path}. Skipping copy."
            )
        else:
            # Use copy2 to preserve metadata (timestamps, permissions)
            shutil.copy2(str(local_csv_path), str(target_path))
            logging.info(f"Copied local CSV to: {target_path}")
        logging.info("Download step completed using local CSV.")
        return

    # Branch 2: Download from Hugging Face Datasets if no local CSV provided
    logging.info("Downloading the dataset (this may take a while)...")
    dataset = load_dataset(
        # Dataset identifier (e.g., "squad")
        ds_cfg["hf_dataset"],
        # Optional config (e.g., "plain_text")
        ds_cfg.get("config_name", None),
        # Dataset split to use
        split=ds_cfg.get("split", "train"),
        # Stream data to save memory
        streaming=bool(ds_cfg.get("streaming", False)),
        # Allow custom dataset code execution
        trust_remote_code=True
    )

    # Get total number of examples for progress tracking
    total_rows = len(dataset)
    logging.info(
        f"Loaded {total_rows} examples from '{ds_cfg.get('config_name', 'default')}' split."
    )

    # Extract column names from the dataset schema
    fieldnames = dataset.column_names
    logging.info(f"Writing to CSV: {out_csv}")

    # Write dataset to CSV with proper encoding and escaping
    with open(out_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            quoting=csv.QUOTE_MINIMAL,    # Only quote when necessary
            escapechar="\\"               # Use backslash for escaping special chars
        )
        writer.writeheader()

        # Process each row with progress tracking
        for row in tqdm(dataset, total=total_rows, desc="Writing rows"):
            # Clean text data by removing line breaks that could break CSV format
            writer.writerow(
                {
                    key: str(row.get(key, "")).replace(
                        "\r", " ").replace("\n", " ")
                    for key in fieldnames
                }
            )

    logging.info("Download step completed.")


# ------------------------------
# Step 2: Keyword filter
# ------------------------------


def step_keyword_filter(cfg: Dict[str, Any], in_csv: str, out_csv: str) -> None:
    """Filter rows by keyword presence in the configured text column.

    Reads the input CSV in chunks to limit memory usage and performs a
    case-insensitive substring search for any of the configured keywords within
    ``io.text_column``. Matching rows are appended to the output CSV. The step
    is skipped when the output already exists and ``general.force`` is not set to true.

    Args:
        cfg: Configuration dictionary with:
            - ``io.text_column``: Column to search for keywords.
            - ``keyword_filter.chunk_size``: Chunk size for streaming reads.
            - ``keyword_filter.keywords``: List of keywords (strings).
            - ``general.force`` (optional): Recompute even if output exists.
        in_csv: Path to the input CSV to scan.
        out_csv: Path to the CSV where matched rows will be written.

    Returns:
        None. Writes matched (i.e. filtered) rows to ``out_csv`` and logs summary statistics.
    """
    # Skip if output already exists and force mode is not enabled
    if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
        logging.info(f"Keyword-filter output exists at {out_csv}; skipping.")
        return

    # Extract configuration parameters
    # Column name containing text to search
    io_text_col = cfg["io"]["text_column"]
    kw_cfg = cfg["keyword_filter"]
    # Process CSV in chunks to manage memory
    chunk_size = int(kw_cfg.get("chunk_size", 100000))
    # Convert all keywords to lowercase
    keywords = [str(k).lower() for k in kw_cfg.get("keywords", [])]

    logging.info(
        f"Keyword-filtering {in_csv} -> {out_csv} with {len(keywords)} keywords"
    )

    # Initialize counters for statistics
    matched_rows = 0
    total_rows = 0

    # Process CSV file in chunks to handle large datasets efficiently
    with pd.read_csv(in_csv, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            logging.info(f"Processing chunk {i + 1} of keyword filtering")

            # Convert text column to lowercase strings for case-insensitive matching
            chunk[io_text_col] = chunk[io_text_col].astype(str).str.lower()

            # Create boolean mask: True if any keyword is found in the text
            mask = chunk[io_text_col].apply(
                lambda x: any(keyword in x for keyword in keywords)
            )

            # Filter chunk to keep only rows where mask is True
            filtered_chunk = chunk[mask]

            # Update statistics
            matched_rows += len(filtered_chunk)
            total_rows += len(chunk)

            # Write filtered chunk to output CSV
            # First chunk: write mode with header, subsequent chunks: append mode without header
            mode = "w" if i == 0 else "a"
            header = i == 0
            filtered_chunk.to_csv(out_csv, index=False,
                                  mode=mode, header=header)

    logging.info(
        f"Keyword filter finished. {matched_rows} / {total_rows} rows matched."
    )


# ------------------------------
# Step 3: Zero-shot classification
# ------------------------------


def step_zero_shot(cfg: Dict[str, Any], in_csv: str, out_csv: str) -> None:
    """Select rows whose text is predicted as the positive label via zero-shot.

    This step loads a Hugging Face zero-shot classification pipeline and runs
    batched inference on ``io.text_column`` from ``in_csv``. For each example,
    if the top predicted label equals the first label in ``zero_shot.labels``,
    the original row is retained. Selected rows are appended to ``out_csv``.

    The step is skipped when the output already exists and ``general.force`` is
    not set to true.

    Args:
        cfg: Configuration dictionary including:
            - ``io.text_column``: Column containing the input text.
            - ``zero_shot.model_name``: HF model (e.g., ``facebook/bart-large-mnli``).
            - ``zero_shot.labels``: Ordered labels, first is treated as positive.
            - ``zero_shot.hypothesis_template``: Natural language template.
            - ``zero_shot.chunk_size``: CSV read chunksize.
            - ``zero_shot.batch_size``: Inference batch size.
            - ``zero_shot.device`` (optional): CUDA device index; defaults to 3
              when CUDA is available, otherwise CPU (``-1``).
            - ``general.force`` (optional): Recompute even if output exists.
        in_csv: Path to the input CSV produced by the keyword filter.
        out_csv: Path for the CSV of rows predicted as positive.

    Returns:
        None. Writes the filtered subset to ``out_csv``.

    Raises:
        RuntimeError: If model initialization or inference fails.


    """
    # Skip if output already exists and force mode is not enabled
    if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
        logging.info(f"Zero-shot output exists at {out_csv}; skipping.")
        return

    # Extract zero-shot classification configuration
    zs_cfg = cfg["zero_shot"]
    io_text_col = cfg["io"]["text_column"]
    # CSV processing chunk size
    chunk_size = int(zs_cfg.get("chunk_size", 1000))
    # Model inference batch size
    batch_size = int(zs_cfg.get("batch_size", 8))
    # Classification labels (order matters!)
    labels = zs_cfg.get("labels", ["crimine", "non crimine"])
    hypothesis_template = zs_cfg.get(
        "hypothesis_template", "Questo testo riguarda {}.")  # Template for zero-shot

    # Configure device for model inference (prefer GPU for speed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Use GPU device 3 by default
    device_index = -1 if device == 'cpu' else int(zs_cfg.get("device", 0))

    logging.info("Loading zero-shot model on GPU device=%s", device_index)
    # Initialize Hugging Face zero-shot classification pipeline
    classifier = hf_pipeline(
        "zero-shot-classification",
        # Default to BART-MNLI model
        model=zs_cfg.get("model_name", "facebook/bart-large-mnli"),
        device=device_index,
    )

    # Process input CSV in chunks to manage memory usage
    with pd.read_csv(in_csv, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            logging.info(f"Classifying chunk {i + 1} ({len(chunk)} rows)")

            # Clean data: remove rows with missing text and reset index
            chunk = chunk.dropna(subset=[io_text_col]).reset_index(drop=True)
            texts: List[str] = chunk[io_text_col].astype(str).tolist()
            results: List[dict] = []  # Store rows that match positive class

            # Process texts in batches for efficient GPU utilization
            for j in tqdm(
                range(0, len(texts), batch_size), desc=f"Zero-shot batch {i + 1}"
            ):
                # Extract current batch of texts
                batch = texts[j: j + batch_size]
                try:
                    # Run zero-shot classification on the batch
                    batch_results = classifier(
                        batch,
                        candidate_labels=labels,                    # Labels to classify against
                        # Template for natural language inference
                        hypothesis_template=hypothesis_template,
                    )

                    # Ensure batch_results is always a list (single input returns dict)
                    if isinstance(batch_results, dict):
                        batch_results = [batch_results]

                    # Process each classification result in the batch
                    for k, result in enumerate(batch_results):
                        # Get highest confidence label
                        top_label = result["labels"][0]
                        # Keep only rows classified as positive class (first label in config)
                        # treat first label as positive class
                        if top_label == labels[0]:
                            row_idx = j + k  # Calculate original row index
                            if row_idx < len(chunk):
                                row = chunk.iloc[row_idx]
                                # Store matching row
                                results.append(row.to_dict())

                except Exception as e:
                    logging.warning(
                        f"Error in batch {j}-{j + batch_size}: {e}")
                    continue  # Skip failed batches and continue processing

            # Save results from current chunk to CSV
            result_df = pd.DataFrame(results)
            if not result_df.empty:
                # First chunk: write mode with header, subsequent chunks: append mode
                mode = "w" if i == 0 else "a"
                header = i == 0
                result_df.to_csv(out_csv, index=False,
                                 mode=mode, header=header)
                logging.info(
                    f"Saved {len(result_df)} rows from chunk {i + 1} to {out_csv}"
                )
            else:
                logging.info(f"No crime-related rows found in chunk {i + 1}")

    logging.info("Zero-shot step completed.")


# ------------------------------
# Step 4: Sentence chunking
# ------------------------------


def step_process(cfg: Dict[str, Any], in_csv: str, out_csv: str) -> None:
    """Split paragraphs into sentence chunks constrained by token counts.

    This step loads a spaCy model for sentence segmentation and a Hugging Face
    tokenizer for token counting. Each paragraph in ``io.text_column`` from
    ``in_csv`` is segmented into sentences and then grouped into contiguous
    chunks such that the total token count per chunk does not exceed
    ``process.max_tokens``. Chunks with a token count below ``process.min_tokens``
    are discarded. The resulting chunks are written one per row to ``out_csv``
    under ``process.output_column``. Existing outputs are skipped unless
    ``general.force`` is set to true.

    Args:
        cfg: Configuration dictionary containing:
            - ``io.text_column``: Column with paragraphs of text.
            - ``process.chunksize``: Input CSV chunk size for streaming.
            - ``process.max_tokens``: Upper bound on tokens per output chunk.
            - ``process.min_tokens``: Minimum tokens required to keep a chunk/sentence.
            - ``process.output_column``: Name of the output CSV column.
            - ``process.spacy_model``: spaCy pipeline name for sentence splitting.
            - ``process.tokenizer_name``: HF tokenizer for token counting.
            - ``process.auto_download_spacy_model``: Auto-download missing model.
            - ``process.mode``: Processing mode - "chunks" or "sentences".
            - ``general.force`` (optional): Recompute even if output exists.
        in_csv: Path to the CSV produced by the zero-shot step.
        out_csv: Path to write the processed text (chunks or sentences).

    Returns:
        None. Writes the processed chunks to ``out_csv``.
    """
    # Skip if output already exists and force mode is not enabled
    if os.path.exists(out_csv) and not cfg.get("general", {}).get("force", False):
        logging.info(f"Processed output exists at {out_csv}; skipping.")
        return

    # Extract processing configuration parameters
    proc_cfg = cfg["process"]
    io_text_col = cfg["io"]["text_column"]
    # CSV processing chunk size
    chunksize_csv = int(proc_cfg.get("chunksize", 100))
    # Maximum tokens per output chunk
    max_tokens = int(proc_cfg.get("max_tokens", 250))
    # Minimum tokens to keep a chunk/sentence
    min_tokens = int(proc_cfg.get("min_tokens", 80))
    # Name of output column in final CSV
    output_column = proc_cfg.get("output_column", "sentence1")
    # Processing mode: "chunks" or "sentences"
    processing_mode = proc_cfg.get("mode", "chunks").lower()

    logging.info("Loading spaCy model and tokenizer for token counting...")
    # Load spaCy model for sentence segmentation (language-specific)
    nlp = ensure_spacy_model(
        # Default to Italian model
        proc_cfg.get("spacy_model", "it_core_news_lg"),
        # Auto-download if missing
        auto_download=bool(proc_cfg.get("auto_download_spacy_model", True)),
    )
    # Load Hugging Face tokenizer for accurate token counting
    tokenizer = AutoTokenizer.from_pretrained(
        # Default multilingual tokenizer
        proc_cfg.get("tokenizer_name", "intfloat/multilingual-e5-large")
    )

    # Initialize output CSV file with header
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(f"{output_column}\n")  # Write column header

    mode_description = "individual sentences" if processing_mode == "sentences" else "token-limited chunks"
    logging.info(f"Processing input file into {mode_description}: {in_csv}")
    # Read input CSV in chunks, only loading the text column to save memory
    chunk_iter = pd.read_csv(
        in_csv, usecols=[io_text_col], chunksize=chunksize_csv)

    # Process each CSV chunk
    for i, chunk in enumerate(tqdm(chunk_iter, desc="Processing chunks")):
        # Collect all sentence chunks from this CSV chunk
        output_rows: List[str] = []

        # Process each paragraph in the current CSV chunk
        for _, row in chunk.iterrows():
            paragraph = str(row[io_text_col])  # Extract paragraph text

            # Choose processing method based on configuration
            if processing_mode == "sentences":
                # Split into individual sentences (ignore max_tokens)
                processed_items = split_paragraph_into_sentences(
                    paragraph, nlp, tokenizer, min_tokens
                )
                logging.debug(
                    f"Sentence mode: extracted {len(processed_items)} sentences")
            else:
                # Default: split into token-limited chunks
                processed_items = split_paragraph_with_token_limit(
                    paragraph, nlp, tokenizer, max_tokens, min_tokens
                )
                logging.debug(
                    f"Chunk mode: extracted {len(processed_items)} chunks")

            # Add all processed items from this paragraph to output list
            output_rows.extend(processed_items)

        # Write all chunks from this CSV chunk to output file
        df_out = pd.DataFrame(output_rows, columns=[output_column])
        # Append to file (header=False since we wrote it manually)
        df_out.to_csv(out_csv, mode="a", index=False, header=False)
        logging.info(
            f"Processed chunk {i + 1}, saved {len(output_rows)} samples.")

    logging.info("Processing completed.")


# ------------------------------
# Orchestrator
# ------------------------------


def run_pipeline(cfg: Dict[str, Any], steps_map: Dict[str, bool]) -> None:
    """
    Run pipeline using a pre-loaded cfg dict and an explicit steps_map mapping
    step_name -> bool (download, keyword_filter, zero_shot, process).

    This function tracks files produced during the current run so subsequent
    steps don't incorrectly fail existence checks when upstream steps also ran.

    Args:
        config_path: Path to the YAML configuration file.
        steps_map: Optional step selection. When ``None`` or empty, the function
            reads a mapping from ``general.steps`` in the config. The mapping is
            expected to contain boolean flags for ``download``, ``keyword_filter``,
            ``zero_shot``, and ``process``. If a custom mapping is provided,
            it should follow the same structure.

    Returns:
        None. Executes enabled steps and writes files to disk.
    """
    # Ensure keys exist
    for k in ["download", "keyword_filter", "zero_shot", "process"]:
        steps_map.setdefault(k, False)

    paths = build_paths(cfg)
    produced = {}  # path -> True if produced in this run

    logging.info("Using outputs directory: %s", Path(cfg["io"]["base_dir"]))

    # Helper to test if an input is available (either produced earlier or on disk)
    def _input_available(path: str) -> bool:
        return bool(produced.get(path, False)) or os.path.exists(path)

    # Step 1: download
    if steps_map.get("download", False):
        step_download(cfg, paths["download_csv"])
        produced[paths["download_csv"]] = True

    # Step 2: keyword_filter
    if steps_map.get("keyword_filter", False):
        # keyword_filter always consumes raw download CSV
        in_csv = paths["download_csv"]
        if not _input_available(in_csv):
            raise RuntimeError(
                f"keyword_filter requires input at {in_csv} but it does not exist. "
                "Run the download step or provide the file."
            )
        step_keyword_filter(cfg, in_csv, paths["filtered_csv"])
        produced[paths["filtered_csv"]] = True

    # Step 3: zero_shot
    if steps_map.get("zero_shot", False):
        # Prefer filtered_csv if keyword_filter also ran (or exists), else fall back to download_csv
        if steps_map.get("keyword_filter", False):
            in_csv = paths["filtered_csv"]
        else:
            in_csv = paths["download_csv"]

        if not _input_available(in_csv):
            raise RuntimeError(
                f"zero_shot requires input at {in_csv} but it does not exist. "
                "Run required upstream step(s) or provide the file."
            )
        step_zero_shot(cfg, in_csv, paths["zero_shot_csv"])
        produced[paths["zero_shot_csv"]] = True

    # Step 4: process
    if steps_map.get("process", False):
        # Prefer zero_shot -> keyword_filter -> download
        if steps_map.get("zero_shot", False):
            in_csv = paths["zero_shot_csv"]
        elif steps_map.get("keyword_filter", False):
            in_csv = paths["filtered_csv"]
        else:
            in_csv = paths["download_csv"]

        if not _input_available(in_csv):
            raise RuntimeError(
                f"process requires input at {in_csv} but it does not exist. "
                "Run required upstream step(s) or provide the file."
            )
        step_process(cfg, in_csv, paths["processed_csv"])
        produced[paths["processed_csv"]] = True

    logging.info("Pipeline run finished.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocessing pipeline")
    parser.add_argument(
        "--config", type=str, default="./config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--steps",
        type=str,
        nargs="*",
        default=None,
        help="Subset of steps to run (names): download keyword_filter zero_shot process. If omitted, uses config general.steps.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-run steps even if outputs exist"
    )
    parser.add_argument("--log-level", type=str,
                        default="INFO", help="Logging level")

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load config once
    cfg = load_config(args.config)
    cfg_steps = cfg.get("general", {}).get("steps", {})

    # Build explicit steps_map from CLI or config
    STEP_NAMES = ["download", "keyword_filter", "zero_shot", "process"]
    if args.steps:
        requested = [s.lower() for s in args.steps]
        steps_map = {name: (name in requested) for name in STEP_NAMES}
    else:
        steps_map = {name: bool(cfg_steps.get(name, False))
                     for name in STEP_NAMES}

    # If process requested but no upstream step provided, enable download as fallback.
    if steps_map.get("process", False) and not any(
        steps_map.get(x, False) for x in ["zero_shot", "keyword_filter", "download"]
    ):
        steps_map["download"] = True
        logging.info(
            "No upstream steps were enabled; enabling 'download' so 'process' has an input file."
        )

    # If force requested, flip the flag in cfg then call the single run_pipeline function.
    if args.force:
        cfg.setdefault("general", {})["force"] = True
        logging.info("Force mode enabled. Re-running steps: %s", steps_map)
        run_pipeline(cfg, steps_map)
    else:
        # Normal execution: call the same run_pipeline function
        logging.info("Running pipeline with steps: %s", steps_map)
        run_pipeline(cfg, steps_map)
