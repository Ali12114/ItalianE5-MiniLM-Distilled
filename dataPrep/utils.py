"""
Utility functions for the preprocessing pipeline.

This module contains reusable utility functions that support the main preprocessing
pipeline, including configuration management, model loading, and text processing utilities.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any

import yaml
import spacy
from spacy.cli import download as spacy_download
from transformers import AutoTokenizer


# ------------------------------
# Configuration utilities
# ------------------------------


def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file from disk.

    This function reads a YAML file and deserializes it into a Python
    dictionary. The configuration governs I/O paths and parameters for
    each step of the preprocessing pipeline (download, keyword filter,
    zero-shot classification, and sentence chunking).

    Args:
        config_path: Absolute or relative path to the YAML configuration file.

    Returns:
        A dictionary containing the parsed configuration. Common top-level keys
        include:
        - ``io``: Base directory and CSV filenames, plus ``text_column``.
        - ``general``: Global flags such as ``force`` and per-step enable flags.
        - ``download``: Dataset source options (local CSV or Hugging Face dataset info).
        - ``keyword_filter``: Chunk size and keywords list.
        - ``zero_shot``: Model name, labels, batch sizes, and device index.
        - ``process``: spaCy model, tokenizer name, and token limits for chunks.

    Raises:
        FileNotFoundError: If ``config_path`` does not exist.
        yaml.YAMLError: If the file cannot be parsed as valid YAML.

    Notes:
        The function does not validate schema beyond returning the parsed
        content. Downstream steps will access expected keys and may raise
        ``KeyError`` if required fields are missing.
    """
    # Open config file with UTF-8 encoding to handle international characters
    with open(config_path, "r", encoding="utf-8") as f:
        # Use safe_load to prevent execution of arbitrary Python code in YAML
        config = yaml.safe_load(f)
    return config


def build_paths(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Construct absolute file paths for all pipeline outputs.

    Ensures the configured base directory exists, then builds full
    paths for intermediate and final CSV data used by the pipeline.

    Args:
        cfg: The loaded configuration dictionary. Must contain
            ``io.base_dir`` and the CSV filenames under ``io``:
            ``download_csv``, ``filtered_csv``, ``zero_shot_csv``,
            and ``processed_csv``.

    Returns:
        A dictionary mapping logical names to absolute file paths:
        ``{"download_csv": str, "filtered_csv": str, "zero_shot_csv": str, "processed_csv": str}``.

    Raises:
        KeyError: If required keys under ``io`` are missing.

    Side Effects:
        Creates the base output directory if it does not already exist.
    """
    # Convert base_dir to absolute path, expanding ~ for home directory
    base_dir = Path(cfg["io"]["base_dir"]).expanduser().resolve()
    # Create the directory structure if it doesn't exist (parents=True for nested dirs)
    base_dir.mkdir(parents=True, exist_ok=True)

    # Build absolute paths for each pipeline stage output file
    paths = {
        # Raw dataset
        "download_csv": str(base_dir / cfg["io"]["download_csv"]),
        # After keyword filtering
        "filtered_csv": str(base_dir / cfg["io"]["filtered_csv"]),
        # After zero-shot classification
        "zero_shot_csv": str(base_dir / cfg["io"]["zero_shot_csv"]),
        # Final processed chunks
        "processed_csv": str(base_dir / cfg["io"]["processed_csv"]),
    }
    return paths


# ------------------------------
# Model management utilities
# ------------------------------


def ensure_spacy_model(model_name: str, auto_download: bool = True):
    """Load a spaCy language model, optionally downloading it if missing.

    Args:
        model_name: Name of the spaCy pipeline to load (e.g., ``it_core_news_lg``).
        auto_download: If True, attempt to download the model if not found
            locally before retrying the load.

    Returns:
        A loaded spaCy ``Language`` object.

    Raises:
        OSError: If the model cannot be found and ``auto_download`` is False,
            or the model fails to load after download.

    """
    try:
        # Attempt to load the spaCy model from local installation
        return spacy.load(model_name)
    except OSError:
        # Model not found locally
        if auto_download:
            logging.info(
                f"spaCy model '{model_name}' not found. Downloading...")
            # Download the model using spaCy's CLI download function
            spacy_download(model_name)
            # Retry loading after download
            return spacy.load(model_name)
        # Re-raise the error if auto_download is disabled
        raise


# ------------------------------
# Text processing utilities
# ------------------------------


def count_tokens(text: str, tokenizer: AutoTokenizer) -> int:
    """Compute a rough token count for a piece of text.

    Uses the provided Hugging Face tokenizer to encode the text and
    returns the length of the resulting token IDs sequence.

    Args:
        text: The input string whose token count should be estimated.
        tokenizer: The Hugging Face tokenizer to use for encoding.

    Returns:
        The number of tokens produced by ``tokenizer.encode(text)``.

    """
    # Use the tokenizer to convert text to token IDs and count them
    # This gives an accurate count for the specific model being used
    return len(tokenizer.encode(text))


def split_paragraph_with_token_limit(
    paragraph: str,
    nlp,
    tokenizer: AutoTokenizer,
    max_tokens: int,
    min_tokens: int
) -> List[str]:
    """Split a paragraph into token-limited sentence chunks.

    The paragraph is segmented into sentences via spaCy, then sentences are
    accumulated into a chunk until adding the next sentence would exceed
    ``max_tokens``. When a chunk would overflow, it is finalized (if it
    meets ``min_tokens``) and a new chunk is started with the current
    sentence. After iteration, the remaining chunk is finalized if it meets
    the minimum token threshold.

    Args:
        paragraph: A single paragraph of text to segment and group.
        nlp: A loaded spaCy Language model for sentence segmentation.
        tokenizer: The Hugging Face tokenizer for token counting.
        max_tokens: Maximum tokens allowed per output chunk.
        min_tokens: Minimum tokens required to keep a chunk.

    Returns:
        A list of chunk strings that respect token limits and thresholds.

    """
    try:
        # Use spaCy to parse the paragraph and segment into sentences
        doc = nlp(paragraph.strip())
        # Extract non-empty sentences as strings
        sentences = [sent.text.strip()
                     for sent in doc.sents if sent.text.strip()]

        # Initialize variables for chunk building
        result_chunks: List[str] = []     # Final list of token-limited chunks
        current_chunk: List[str] = []     # Current chunk being built
        current_token_count = 0           # Running token count for current chunk

        # Process each sentence and group into chunks
        for sent in sentences:
            sent_token_count = count_tokens(sent, tokenizer)

            # Check if adding this sentence would exceed max_tokens limit
            if current_token_count + sent_token_count > max_tokens:
                # Finalize current chunk if it has content and meets minimum threshold
                if current_chunk:
                    chunk_text = " ".join(current_chunk)
                    if count_tokens(chunk_text, tokenizer) >= min_tokens:
                        result_chunks.append(chunk_text)

                # Start new chunk with current sentence
                current_chunk = [sent]
                current_token_count = sent_token_count
            else:
                # Add sentence to current chunk
                current_chunk.append(sent)
                current_token_count += sent_token_count

        # Handle remaining chunk after processing all sentences
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            # Only keep chunk if it meets minimum token requirement
            if count_tokens(chunk_text, tokenizer) >= min_tokens:
                result_chunks.append(chunk_text)

        return result_chunks
    except Exception as e:
        logging.warning(f"Error processing paragraph: {e}")
        return []  # Return empty list on processing errors


def split_paragraph_into_sentences(
    paragraph: str,
    nlp,
    tokenizer: AutoTokenizer,
    min_tokens: int
) -> List[str]:
    """Split a paragraph into individual sentences.

    The paragraph is segmented into sentences via spaCy. Each sentence that
    meets the minimum token threshold is kept as a separate item.

    Args:
        paragraph: A single paragraph of text to segment.
        nlp: A loaded spaCy Language model for sentence segmentation.
        tokenizer: The Hugging Face tokenizer for token counting.
        min_tokens: Minimum tokens required to keep a sentence.

    Returns:
        A list of individual sentences that meet the minimum token threshold.

    """
    try:
        # Use spaCy to parse the paragraph and segment into sentences
        doc = nlp(paragraph.strip())
        # Extract non-empty sentences as strings
        sentences = [sent.text.strip()
                     for sent in doc.sents if sent.text.strip()]

        # Filter sentences by minimum token count
        result_sentences: List[str] = []
        for sent in sentences:
            sent_token_count = count_tokens(sent, tokenizer)
            # Only keep sentences that meet minimum token requirement
            if sent_token_count >= min_tokens:
                result_sentences.append(sent)

        return result_sentences
    except Exception as e:
        logging.warning(f"Error processing paragraph: {e}")
        return []  # Return empty list on processing errors
