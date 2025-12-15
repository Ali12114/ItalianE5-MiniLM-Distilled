import pickle
from typing import Union
from torch.utils.data import Dataset
from pathlib import Path


class LabelledBatchedSentencesDataset(Dataset):
    """
    PyTorch Dataset for pre-batched sentences with corresponding embeddings.

    This dataset is designed to work with data organized as created by conversions.py's
    shuffle_and_align() function. The data structure expected is:

    base_path/
        0/{sentences.pkl, embeddings.pkl}
        1/{sentences.pkl, embeddings.pkl}
        ...

    Each group folder contains:
    - sentences.pkl: List of sentence batches, where each batch is a list of strings
    - embeddings.pkl: Stacked tensor of shape (num_batches, batch_size, embedding_dim)

    This dataset implements lazy loading - it loads one group into memory at a time
    to handle large datasets efficiently. Each __getitem__ call returns a tuple of
    (sentence_batch, embedding_batch) where the batch size is typically 32.

    Important Notes:
    - Does NOT support external shuffling by DataLoader. Shuffling should be done
      during data preparation. Using shuffle=True in DataLoader will cause
      extremely poor performance due to random group loading.
    - Does NOT support external batching. Each item is already a pre-batched tuple
      of (sentences, embeddings), so DataLoader should use batch_size=1 with
      a custom collate function (see organized_training.py's flat_collate_fn).
    """

    def __init__(self, base_path, group_size=3200):
        """
        Initializes the LabelledBatchedSentencesDataset.

        Args:
            base_path (str): Path to the directory containing numbered group folders (0, 1, 2, ...).
                Each group folder must contain sentences.pkl and embeddings.pkl files.
            group_size (int): Number of batches per group. This should match the group_size
                used when creating the data with conversions.py. Defaults to 3200.
                Note: This parameter is kept for backward compatibility but actual group sizes
                are tracked individually since the last group may be smaller.

        Note:
            On initialization, the dataset counts the total number of batches by loading
            all sentences.pkl files. This can be slow if there are many groups.
            Group 0 is immediately loaded into memory.

        Raises:
            FileNotFoundError: If base_path doesn't exist or group folders are missing required files.

        Example:
            >>> dataset = LabelledBatchedSentencesDataset(
            ...     base_path="/data/aligned_labels/train",
            ...     group_size=3200
            ... )
            >>> len(dataset)  # Total number of batches across all groups
        """
        self.root = Path(base_path)
        self.group_size = group_size

        # Track actual group sizes and build cumulative sum for correct indexing
        # This handles cases where the last group is smaller than group_size
        group_dirs = sorted([f for f in self.root.iterdir() if f.is_dir()],
                            key=lambda x: int(x.name) if x.name.isdigit() else float('inf'))

        self.group_sizes = []
        # Cumulative sum: group_cumsum[i] = sum of sizes of groups 0 to i-1
        self.group_cumsum = [0]
        # Map from group_num (index) to actual directory name
        self.group_dir_names = []

        self.total = 0
        for group_dir in group_dirs:
            sentences_path = group_dir / 'sentences.pkl'
            if sentences_path.exists():
                with open(sentences_path, "rb") as pklfile:
                    group_len = len(pickle.load(pklfile))
                    self.group_sizes.append(group_len)
                    # Store actual directory name
                    self.group_dir_names.append(group_dir.name)
                    self.total += group_len
                    self.group_cumsum.append(self.total)

        print("Size of dataset: ", self.total)
        print(f"Number of groups: {len(self.group_sizes)}")
        if len(self.group_sizes) > 0:
            print(
                f"Group sizes: min={min(self.group_sizes)}, max={max(self.group_sizes)}")

        self.loaded_group_num = None
        self.loaded_group = None
        if len(self.group_sizes) > 0:
            self._load_group(0)

    def __len__(self):
        """
        Returns the total number of batches in the dataset.

        Returns:
            int: Total number of pre-batched items across all groups.
                This is computed during initialization by counting batches
                in all group folders.
        """
        return self.total

    def _group_position(self, item_id):
        """
        Calculates which group and position within that group an item_id corresponds to.

        Uses cumulative sums to correctly handle groups of different sizes, especially
        when the last group is smaller than group_size.

        Args:
            item_id (int): Global item index (0-indexed across all groups).

        Returns:
            tuple[int, int]: A tuple of (group_num, group_index) where:
                - group_num: Which numbered group folder contains this item
                - group_index: Position of the item within that group (0 to group_size-1)

        Raises:
            IndexError: If item_id is out of range.

        Example:
            >>> dataset._group_position(3205)  # If group 0 has 3200 items
            >>> # Returns (1, 5)  # Group 1, position 5 within that group
        """
        if item_id < 0 or item_id >= self.total:
            raise IndexError(
                f"item_id {item_id} is out of range [0, {self.total})")

        # Binary search to find which group contains this item_id
        # group_cumsum[i] is the cumulative count up to (but not including) group i
        # So we want the largest i such that group_cumsum[i] <= item_id
        group_num = 0
        for i in range(len(self.group_cumsum) - 1):
            if self.group_cumsum[i + 1] > item_id:
                group_num = i
                break
        else:
            # If we didn't break, item_id is in the last group
            group_num = len(self.group_cumsum) - 2

        # Calculate position within the group
        group_index = item_id - self.group_cumsum[group_num]

        return (group_num, group_index)

    def _load_group(self, group_num):
        """
        Loads a specific group into memory for lazy loading.

        Loads both sentences.pkl and embeddings.pkl from the specified group folder
        and stores them in memory. This method is called automatically when __getitem__
        requests an item from a different group than the one currently loaded.

        Args:
            group_num (int): Index of the group to load (0-indexed in sorted order).
                The actual directory name is looked up from self.group_dir_names.

        Raises:
            FileNotFoundError: If the group folder or required pickle files don't exist.
            IndexError: If group_num is out of range.

        Note:
            This method clears the previously loaded group before loading the new one.
            The loaded data is stored as:
            - self.loaded_group_num: The group number currently in memory
            - self.loaded_group: Tuple of (sentences, embeddings) where:
              - sentences: List of sentence batches (list[list[str]])
              - embeddings: Tensor of shape (num_batches, batch_size, embedding_dim)
        """
        if group_num < 0 or group_num >= len(self.group_dir_names):
            raise IndexError(
                f"group_num {group_num} is out of range [0, {len(self.group_dir_names)})")

        group_dir_name = self.group_dir_names[group_num]
        print("LOADING NEW GROUP", group_num, f"(directory: {group_dir_name})")
        self.loaded_group_num = None
        self.loaded_group = None

        with (self.root / group_dir_name / "sentences.pkl").open("rb") as pklfile:
            sentences = pickle.load(pklfile)
        with (self.root / group_dir_name / "embeddings.pkl").open("rb") as pklfile:
            embeddings = pickle.load(pklfile)

        self.loaded_group_num = group_num
        self.loaded_group = (sentences, embeddings)

    def __getitem__(self, item_id):
        """
        Returns a pre-batched item (sentences and embeddings) by index.

        Implements lazy loading: if the requested item is in a different group than
        the one currently loaded, that group is loaded into memory first. This allows
        the dataset to handle very large datasets without loading everything at once.

        Args:
            item_id (int): Global index of the batch to retrieve (0-indexed).

        Returns:
            tuple[list[str], torch.Tensor]: A tuple containing:
                - sentences: List of strings representing a batch of sentences
                  (typically 32 sentences)
                - embeddings: Tensor of shape (batch_size, embedding_dim) containing
                  the corresponding target embeddings for those sentences

        Raises:
            IndexError: If item_id is out of range.
            FileNotFoundError: If the required group folder or files don't exist.

        Example:
            >>> dataset = LabelledBatchedSentencesDataset(base_path="./data/train")
            >>> sentences, embeddings = dataset[100]
            >>> len(sentences)  # Typically 32 (batch size)
            >>> embeddings.shape  # (32, 384) for example
        """
        group_num, group_index = self._group_position(item_id)

        if self.loaded_group_num != group_num:
            self._load_group(group_num)
        sentences, embeddings = self.loaded_group

        return sentences[group_index], embeddings[group_index]


# Example usage
if __name__ == "__main__":
    pass
