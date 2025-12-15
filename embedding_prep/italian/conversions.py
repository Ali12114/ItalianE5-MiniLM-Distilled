import os
import csv
import pickle
import random
from pathlib import Path
from typing import Any, List

import torch
from sentence_transformers import SentenceTransformer


# =========================================================
# CONFIG
# =========================================================
DEFAULT_BATCH_SIZE = 2048   # <-- IMPORTANT: set this to the SAME batch_size you used in encode()


# =========================================================
# STREAMING / ENCODING
# =========================================================
def stream_sentences(file_path: str, batch_size: int, starting_batch: int = 0):
    """
    Generator that yields sentences from CSV in batches.
    
    Reads sentences from a CSV file containing a 'sentence1' column and yields them
    in batches of the specified size. Useful for resuming encoding from a specific
    batch number when processing large datasets.
    
    Args:
        file_path (str): Path to the CSV file containing sentences in 'sentence1' column.
        batch_size (int): Number of sentences to include in each batch.
        starting_batch (int): Batch number to start from (0-indexed). Defaults to 0.
            Batches before this number are skipped.
    
    Yields:
        list[str]: A list of sentences of length batch_size (or less for the final batch).
    
    Example:
        >>> for batch in stream_sentences("data.csv", batch_size=2048, starting_batch=5):
        ...     print(f"Processing {len(batch)} sentences")
    """
    batch = []
    current_batch_num = 0

    with open(file_path, newline='', encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            batch.append(row["sentence1"])
            if len(batch) == batch_size:
                if current_batch_num >= starting_batch:
                    yield batch
                current_batch_num += 1
                batch = []
        if batch and current_batch_num >= starting_batch:
            yield batch


def save_group(tensors: List[torch.Tensor], group_num: int, output_dir: str):
    """
    Saves a list of tensors as a pickle file in the output directory.
    
    Creates the output directory if it doesn't exist and saves the tensors
    with a filename format: {group_num}.pkl. This is used to organize
    embeddings into groups for efficient storage and loading.
    
    Args:
        tensors (List[torch.Tensor]): List of tensor embeddings to save.
        group_num (int): Numeric identifier for this group (used in filename).
        output_dir (str): Directory path where the pickle file will be saved.
    
    Example:
        >>> embeddings = [torch.randn(2048, 384) for _ in range(50)]
        >>> save_group(embeddings, group_num=0, output_dir="./data/embeddings")
        >>> # Saves to ./data/embeddings/0.pkl
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{group_num}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(tensors, f)


def encode(model: Any,
           input_csv: str,
           output_dir: str,
           batch_size: int = 2048,
           group_size: int = 50,
           starting_group: int = 0,
           device: str = None):
    """
    Encodes sentences from a CSV file in batches using a sentence transformer model.
    
    Reads sentences from a CSV file, encodes them in batches using the provided model,
    and saves the resulting embeddings in organized groups. Each group contains
    group_size batches of embeddings. Useful for processing large datasets incrementally
    and resuming from a specific group if interrupted.
    
    Args:
        model (Any): Either a SentenceTransformer model instance or a string path
            to a HuggingFace model identifier. If a string, loads the model.
        input_csv (str): Path to CSV file with sentences in 'sentence1' column.
        output_dir (str): Directory where encoded embeddings will be saved as pickle files.
        batch_size (int): Number of sentences to encode in each batch. Defaults to 2048.
            IMPORTANT: Must match the batch_size used in shuffle_and_align.
        group_size (int): Number of batches to combine into one group before saving.
            Defaults to 50. Each group file will contain group_size batches.
        starting_group (int): Group number to resume from (0-indexed). Defaults to 0.
            Useful for resuming interrupted encoding sessions.
        device (str, optional): Device to run encoding on (e.g., "cuda:0", "cpu"). 
            If None, automatically uses CUDA if available, else CPU.
    
    Note:
        The batch_size parameter is critical - it must match the batch_size used
        in shuffle_and_align() to ensure proper alignment of sentences and embeddings.
    
    Example:
        >>> encode(
        ...     model="intfloat/multilingual-e5-large",
        ...     input_csv="data.csv",
        ...     output_dir="./embeddings",
        ...     batch_size=2048,
        ...     group_size=50
        ... )
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if isinstance(model, str):
        model = SentenceTransformer(model, device=device)

    tensors_group = []
    group_num = starting_group
    total_sentences = group_num * group_size * batch_size
    batch_count = group_num * group_size
    starting_batch = batch_count
    print("Device:", device)

    for batch in stream_sentences(input_csv, batch_size, starting_batch=starting_batch):
        batch_count += 1
        total_sentences += len(batch)

        print(f"[Batch {batch_count}] Encoding {len(batch)} sentences "
              f"(Total so far: {total_sentences})...")

        embeddings = model.encode(batch,
                                  convert_to_tensor=True,
                                  device=device,
                                  show_progress_bar=False)
        tensors_group.append(embeddings)

        if len(tensors_group) == group_size:
            save_group(tensors_group, group_num, output_dir)
            print(f"Saved group {group_num}")
            tensors_group = []
            group_num += 1

    if tensors_group:
        save_group(tensors_group, group_num, output_dir)
        print(f"Saved final group {group_num} "
              f"({total_sentences} sentences total)")

    print("\n--- Encoding complete ---")
    print(f"Total sentences encoded: {total_sentences}")
    print(f"Total groups saved: {group_num + (1 if tensors_group else 0)}")


# =========================================================
# LOADING / ALIGNING
# =========================================================
def load_sentences(batch_size=DEFAULT_BATCH_SIZE,
                   path=""):
    """
    Loads sentences from a CSV file and truncates to a multiple of batch_size.
    
    Reads the 'sentence1' column from a CSV file and returns a list of sentences.
    The list is truncated to ensure its length is a multiple of batch_size, which
    is required for proper batching in downstream processing. This ensures that
    all batches have exactly batch_size sentences (except possibly the last one
    if we don't truncate, but we do truncate here).
    
    Args:
        batch_size (int): Batch size used for processing. The returned list length
            will be truncated to the largest multiple of batch_size that fits.
            Defaults to DEFAULT_BATCH_SIZE (2048).
        path (str): Path to the CSV file containing sentences in 'sentence1' column.
            Defaults to a specific path in the project.
    
    Returns:
        list[str]: List of sentences, with length truncated to a multiple of batch_size.
    
    Example:
        >>> sentences = load_sentences(batch_size=32)
        >>> len(sentences) % 32 == 0  # True
    """
    import pandas as pd
    df = pd.read_csv(path, usecols=['sentence1'])
    sentences = df["sentence1"].astype(str).tolist()
    total = len(sentences)
    keep = (total // batch_size) * batch_size
    return sentences[:keep]


def load_embeddings(labels_dir, batch_size=DEFAULT_BATCH_SIZE):
    """
    Loads embedding tensors from pickle files and flattens them into a list.
    
    Reads all .pkl files from the specified directory, sorts them numerically,
    and extracts individual embedding vectors from each tensor. The embeddings
    are moved to CPU and collected into a single list. The list is truncated
    to ensure its length is a multiple of batch_size for proper batching.
    
    Args:
        labels_dir (str): Directory containing pickle files with embedding tensors.
            Each pickle file should contain a list of tensors. Files are sorted
            numerically by their filename (without .pkl extension).
        batch_size (int): Batch size used for processing. The returned list length
            will be truncated to the largest multiple of batch_size that fits.
            Defaults to DEFAULT_BATCH_SIZE (2048).
    
    Returns:
        list[torch.Tensor]: List of individual embedding tensors (on CPU), with
            length truncated to a multiple of batch_size.
    
    Note:
        Each pickle file typically contains a list of tensors (e.g., 50 tensors
        of shape (2048, dim)). This function flattens them into individual
        embedding vectors.
    
    Example:
        >>> embeddings = load_embeddings("./data/pca_labels", batch_size=32)
        >>> len(embeddings) % 32 == 0  # True
    """
    all_embeddings = []
    files = sorted(Path(labels_dir).glob('*.pkl'), key=lambda x: int(x.stem))
    for pkl_file in files:
        with open(pkl_file, 'rb') as f:
            tensors_list = pickle.load(f)
        for tensor in tensors_list:
            for i in range(tensor.shape[0]):
                all_embeddings.append(tensor[i].cpu())
    total = len(all_embeddings)
    keep = (total // batch_size) * batch_size
    return all_embeddings[:keep]


def shuffle_in_unison(list1, list2, seed=42):
    """
    Shuffles two lists in the same random order to maintain correspondence.
    
    Shuffles both lists using the same random seed, ensuring that elements
    at corresponding indices remain paired. This is essential for maintaining
    alignment between sentences and their embeddings after shuffling.
    
    Args:
        list1: First list to shuffle (e.g., sentences).
        list2: Second list to shuffle (e.g., embeddings). Must have same length as list1.
        seed (int): Random seed for reproducibility. Defaults to 42.
    
    Returns:
        tuple[list, list]: A tuple of (shuffled_list1, shuffled_list2) with the
            same correspondence as the original lists but in randomized order.
    
    Raises:
        ValueError: If list1 and list2 have different lengths.
    
    Example:
        >>> sentences = ["a", "b", "c"]
        >>> embeddings = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
        >>> s_shuffled, e_shuffled = shuffle_in_unison(sentences, embeddings, seed=42)
        >>> # Correspondence is maintained: sentence[i] still matches embedding[i]
    """
    if len(list1) != len(list2):
        raise ValueError(f"Length mismatch: {len(list1)} vs {len(list2)}")
    random.seed(seed)
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    if not combined:
        return [], []
    l1, l2 = zip(*combined)
    return list(l1), list(l2)


def batch_sentences(sentences, batch_size=DEFAULT_BATCH_SIZE):
    """
    Splits a list of sentences into batches of the specified size.
    
    Divides the input list of sentences into consecutive chunks (batches)
    of the specified size. The final batch may be smaller than batch_size
    if the total number of sentences is not evenly divisible.
    
    Args:
        sentences (list[str]): List of sentences to batch.
        batch_size (int): Number of sentences per batch. Defaults to DEFAULT_BATCH_SIZE (2048).
    
    Returns:
        list[list[str]]: List of batches, where each batch is a list of sentences.
    
    Example:
        >>> sentences = ["s1", "s2", "s3", "s4", "s5"]
        >>> batches = batch_sentences(sentences, batch_size=2)
        >>> # Returns [["s1", "s2"], ["s3", "s4"], ["s5"]]
    """
    return [sentences[i:i+batch_size] for i in range(0, len(sentences), batch_size)]


def batch_embeddings(embeddings, batch_size=DEFAULT_BATCH_SIZE):
    """
    Groups individual embedding tensors into batches and stacks them.
    
    Takes a list of individual embedding tensors, groups them into batches
    of the specified size, and stacks each batch into a single tensor.
    The resulting batches are ready for training or evaluation.
    
    Args:
        embeddings (list[torch.Tensor]): List of individual embedding tensors,
            each of shape (embedding_dim,).
        batch_size (int): Number of embeddings per batch. Defaults to DEFAULT_BATCH_SIZE (2048).
    
    Returns:
        list[torch.Tensor]: List of batched tensors, each of shape (batch_size, embedding_dim).
            The final batch may have fewer than batch_size embeddings.
    
    Example:
        >>> embeddings = [torch.randn(384) for _ in range(100)]
        >>> batches = batch_embeddings(embeddings, batch_size=32)
        >>> batches[0].shape  # torch.Size([32, 384])
    """
    batches = []
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i+batch_size]
        stacked = torch.stack(batch)
        batches.append(stacked)
    return batches


def group_batches(sent_batches, emb_batches, group_size=3200):
    """
    Groups sentence and embedding batches into larger groups for storage.
    
    Combines consecutive batches into groups to organize data for efficient
    storage and loading. Both sentence batches and embedding batches are
    grouped together to maintain alignment. This is used to create the
    final data structure for training.
    
    Args:
        sent_batches (list[list[str]]): List of sentence batches, where each
            batch is a list of sentences.
        emb_batches (list[torch.Tensor]): List of embedding batches, where each
            batch is a tensor of shape (batch_size, embedding_dim).
            Must have the same length as sent_batches.
        group_size (int): Number of batches to combine into one group.
            Defaults to 3200.
    
    Returns:
        tuple[list, list]: A tuple of (grouped_sentences, grouped_embeddings).
            grouped_sentences: List of groups, each group containing group_size batches.
            grouped_embeddings: List of groups, each group containing group_size batches.
    
    Example:
        >>> sent_batches = [["s1"], ["s2"], ["s3"], ["s4"]]
        >>> emb_batches = [torch.randn(1, 384) for _ in range(4)]
        >>> groups_s, groups_e = group_batches(sent_batches, emb_batches, group_size=2)
        >>> len(groups_s)  # 2 (4 batches / 2 per group)
    """
    grouped_sentences = []
    grouped_embeddings = []
    for i in range(0, len(sent_batches), group_size):
        grouped_sentences.append(sent_batches[i:i+group_size])
        grouped_embeddings.append(emb_batches[i:i+group_size])
    return grouped_sentences, grouped_embeddings


def save_groups(grouped_sentences, grouped_embeddings, base_path):
    """
    Saves grouped sentence and embedding batches to disk for training.
    
    Organizes and saves the grouped data into a structured directory format
    where each group is stored in its own numbered folder. Each folder contains:
    - sentences.pkl: List of sentence batches for this group
    - embeddings.pkl: Stacked tensor of embedding batches (num_batches, batch_size, dim)
    
    This format is expected by LabelledBatchedSentencesDataset for training.
    
    Args:
        grouped_sentences (list[list[list[str]]]): List of groups, where each group
            contains multiple sentence batches, and each batch is a list of sentences.
        grouped_embeddings (list[list[torch.Tensor]]): List of groups, where each group
            contains multiple embedding batch tensors. Must have same length as grouped_sentences.
        base_path (str): Base directory path where groups will be saved.
            Each group will be saved in a subdirectory named by its index (0, 1, 2, ...).
    
    Example:
        >>> grouped_sentences = [[["s1"], ["s2"]], [["s3"], ["s4"]]]
        >>> grouped_embeddings = [[torch.randn(1, 384), torch.randn(1, 384)], ...]
        >>> save_groups(grouped_sentences, grouped_embeddings, "./data/aligned_labels")
        >>> # Creates: ./data/aligned_labels/0/{sentences.pkl, embeddings.pkl}
        >>> #         ./data/aligned_labels/1/{sentences.pkl, embeddings.pkl}
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    for group_num, (sent_group, emb_group) in enumerate(zip(grouped_sentences, grouped_embeddings)):
        group_folder = base_path / str(group_num)
        group_folder.mkdir(exist_ok=True)

        with open(group_folder / 'sentences.pkl', 'wb') as f:
            pickle.dump(sent_group, f)

        emb_tensor = torch.stack(emb_group)  # (num_batches, batch_size, dim)
        with open(group_folder / 'embeddings.pkl', 'wb') as f:
            pickle.dump(emb_tensor, f)


def split_groups_into_train_val(
    groups_base_path,
    output_base_path,
    dataset_name,
    val_ratio=0.1,
    seed=42
):
    """
    Splits groups into train and validation sets and reorganizes them.
    
    Takes all groups from groups_base_path, randomly splits them into train/val,
    and reorganizes them into a structured format with continuous numbering starting
    from 0 in each split.
    
    The output structure is:
    output_base_path/
        dataset_name/
            train/
                0/{sentences.pkl, embeddings.pkl}
                1/{sentences.pkl, embeddings.pkl}
                ...
            val/
                0/{sentences.pkl, embeddings.pkl}
                1/{sentences.pkl, embeddings.pkl}
                ...
    
    Args:
        groups_base_path (str): Path containing numbered group folders (0, 1, 2, ...)
            Each folder should contain sentences.pkl and embeddings.pkl.
        output_base_path (str): Base directory where the dataset folder will be created.
        dataset_name (str): Name of the dataset folder to create inside output_base_path.
        val_ratio (float): Proportion of groups to use for validation (0.0 to 1.0).
            Defaults to 0.1 (10%).
        seed (int): Random seed for reproducibility. Defaults to 42.
    
    Example:
        >>> split_groups_into_train_val(
        ...     groups_base_path="./data/mc4/aligned_labels_temp",
        ...     output_base_path="./data/mc4/aligned_labels",
        ...     dataset_name="my_dataset",
        ...     val_ratio=0.1
        ... )
    """
    import random
    import shutil
    
    groups_base_path = Path(groups_base_path)
    output_dataset_path = Path(output_base_path) / dataset_name
    train_path = output_dataset_path / "train"
    val_path = output_dataset_path / "val"
    
    # Create output directories
    train_path.mkdir(parents=True, exist_ok=True)
    val_path.mkdir(parents=True, exist_ok=True)
    
    # Find all group folders
    group_folders = sorted(
        [f for f in groups_base_path.iterdir() if f.is_dir() and f.name.isdigit()],
        key=lambda x: int(x.name)
    )
    
    if not group_folders:
        raise RuntimeError(f"No group folders found in {groups_base_path}")
    
    print(f"Found {len(group_folders)} groups")
    
    # Split groups into train/val
    random.seed(seed)
    shuffled_folders = group_folders.copy()
    random.shuffle(shuffled_folders)
    
    n_val = max(1, int(len(shuffled_folders) * val_ratio))
    val_groups = shuffled_folders[:n_val]
    train_groups = shuffled_folders[n_val:]
    
    print(f"Splitting: {len(train_groups)} train groups, {len(val_groups)} val groups")
    
    # Copy and renumber train groups
    print("Copying train groups...")
    for new_idx, old_folder in enumerate(train_groups):
        new_folder = train_path / str(new_idx)
        new_folder.mkdir(exist_ok=True)
        
        # Copy sentences.pkl and embeddings.pkl
        shutil.copy(old_folder / "sentences.pkl", new_folder / "sentences.pkl")
        shutil.copy(old_folder / "embeddings.pkl", new_folder / "embeddings.pkl")
        
        if (new_idx + 1) % 100 == 0:
            print(f"  Copied {new_idx + 1}/{len(train_groups)} train groups...")
    
    # Copy and renumber val groups
    print("Copying val groups...")
    for new_idx, old_folder in enumerate(val_groups):
        new_folder = val_path / str(new_idx)
        new_folder.mkdir(exist_ok=True)
        
        # Copy sentences.pkl and embeddings.pkl
        shutil.copy(old_folder / "sentences.pkl", new_folder / "sentences.pkl")
        shutil.copy(old_folder / "embeddings.pkl", new_folder / "embeddings.pkl")
        
        if (new_idx + 1) % 10 == 0:
            print(f"  Copied {new_idx + 1}/{len(val_groups)} val groups...")
    
    print(f"Train/val split complete!")
    print(f"Train: {len(train_groups)} groups in {train_path}")
    print(f"Val: {len(val_groups)} groups in {val_path}")


def shuffle_and_align(
    batch_size=32,
    labels_path=None,
    aligned_base_path=None,
    sentences_csv_path=None,
    group_size=3200,
    seed=42
):
    """
    Main function to shuffle and align sentences with their embeddings for training.
    
    This function performs the complete pipeline to prepare data for distillation training:
    1. Loads sentences from CSV and embeddings from pickle files
    2. Shuffles them together to maintain correspondence
    3. Batches them into the specified batch size
    4. Groups batches into larger groups (3200 batches per group)
    5. Saves the organized data in the format expected by the training dataset
    
    The output structure saves groups directly to aligned_base_path:
    aligned_base_path/
        0/{sentences.pkl, embeddings.pkl}
        1/{sentences.pkl, embeddings.pkl}
        ...
    
    Args:
        batch_size (int): Batch size for creating batches. This determines how many
            sentences/embeddings go into each training batch. Defaults to 32.
            IMPORTANT: Must match the batch_size used when encoding embeddings.
        labels_path (str, optional): Path to PCA-reduced embeddings directory.
            If None, defaults to "./data/mc4/pca_labels".
        aligned_base_path (str, optional): Path where aligned groups will be saved.
            If None, defaults to "./data/mc4/aligned_labels/".
        sentences_csv_path (str, optional): Path to CSV file with sentences.
            If None, uses default from load_sentences().
        group_size (int): Number of batches per group. Defaults to 3200.
        seed (int): Random seed for shuffling. Defaults to 42.
    
    Raises:
        RuntimeError: If the number of sentences and embeddings don't match.
    
    Note:
        The batch_size here is the training batch size (typically 32), which may
        differ from the encoding batch_size (typically 2048) used in encode().
        This function uses the encoding batch_size to load and truncate data, but
        then rebatches at the training batch_size.
        For train/val split, use split_groups_into_train_val() after this function.
    
    Example:
        >>> shuffle_and_align(batch_size=32)
        >>> # Creates organized data structure ready for training
    """
    # if labels_path is None:
    #     labels_path = "./data/mc4/pca_labels"
    # if aligned_base_path is None:
    #     aligned_base_path = "./data/mc4/aligned_labels/"

    print("Loading sentences ...")
    # Use DEFAULT_BATCH_SIZE for loading/truncating (encoding batch size)
    # This ensures proper alignment with embeddings that were encoded with that batch size
    if sentences_csv_path is None:
        sentences = load_sentences(batch_size=DEFAULT_BATCH_SIZE)
    else:
        # Import here to avoid circular dependency
        import pandas as pd
        df = pd.read_csv(sentences_csv_path, usecols=['sentence1'])
        sentences_list = df["sentence1"].astype(str).tolist()
        total = len(sentences_list)
        keep = (total // DEFAULT_BATCH_SIZE) * DEFAULT_BATCH_SIZE
        sentences = sentences_list[:keep]

    print(f"Loading embeddings from {labels_path} ...")
    embeddings = load_embeddings(labels_path, batch_size=DEFAULT_BATCH_SIZE)

    if len(sentences) != len(embeddings):
        raise RuntimeError(f"Mismatch in sentences and embeddings! "
                           f"{len(sentences)} vs {len(embeddings)}")

    print("Shuffling sentences and embeddings together ...")
    sentences_shuffled, embeddings_shuffled = shuffle_in_unison(sentences, embeddings, seed=seed)

    print(f"Batching into size {batch_size} ...")
    sentences_batches = batch_sentences(sentences_shuffled, batch_size=batch_size)
    embeddings_batches = batch_embeddings(embeddings_shuffled, batch_size=batch_size)

    print(f"Grouping batches into groups of {group_size} batches ...")
    grouped_sentences, grouped_embeddings = group_batches(sentences_batches, embeddings_batches, group_size=group_size)

    print(f"Saving {len(grouped_sentences)} groups into {aligned_base_path} ...")
    save_groups(grouped_sentences, grouped_embeddings, aligned_base_path)

    print("Done!")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="shuffle",
                        choices=["encode", "shuffle"], help="Which step to run")
    parser.add_argument("--model", type=str, default='intfloat/multilingual-e5-large')
    parser.add_argument("--input_csv", type=str,
                        default="/big_storage/ali/big_storage/DataPreparationScript/Data Preparation Italian Embedding Model/distilled_miniLM_it-master/_PATH_TO_SAVE_OUTPUT_FILES_/data_filtered_zero_shot_chunk.csv")
    parser.add_argument("--output_dir", type=str, default="./data/mc4/labels")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--group_size", type=int, default=50)
    parser.add_argument("--starting_group", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu, or cuda:0). If not specified, automatically uses CUDA if available, else CPU.")
    args = parser.parse_args()

    if args.mode == "encode":
        # Auto-detect device if not specified
        device = args.device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Auto-detected device: {device}")
        
        encode(model=args.model,
               input_csv=args.input_csv,
               output_dir=args.output_dir,
               batch_size=args.batch_size,
               group_size=args.group_size,
               starting_group=args.starting_group,
               device=device)
    else:
        shuffle_and_align(batch_size=32)
