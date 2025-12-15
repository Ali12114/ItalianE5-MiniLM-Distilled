"""
Configuration file for Italian data preparation pipeline.

This file contains all paths and hyperparameters needed for the complete
data preparation pipeline: encoding, PCA, alignment, and train/val split.
"""

# =========================================================
# PATHS
# =========================================================

# Input paths
INPUT_CSV = "PATH_TO_YOUR_DATASET/dataset/data_filtered_zero_shot_chunk.csv"

# Intermediate output paths
LABELS_DIR = "./data/mc4/labels"  # Where encoded embeddings are saved (step 1)
PCA_MODELS_DIR = "./data/mc4/pca_models"  # Where PCA model is saved (step 2)
PCA_LABELS_DIR = "./data/mc4/pca_labels"  # Where PCA-reduced embeddings are saved (step 2)

# Final output path
ALIGNED_LABELS_BASE = "./data/mc4/aligned_labels"  # Base directory for aligned labels
DATASET_NAME = "my_dataset"  # Name of the dataset folder inside aligned_labels

# =========================================================
# ENCODING HYPERPARAMETERS (Step 1)
# =========================================================

ENCODE_MODEL = "intfloat/multilingual-e5-large"  # Model to use for encoding
ENCODE_BATCH_SIZE = 2048  # IMPORTANT: Must match DEFAULT_BATCH_SIZE in conversions.py
ENCODE_GROUP_SIZE = 50  # Number of batches per group
ENCODE_STARTING_GROUP = 0  # Group to start from (for resuming)
ENCODE_DEVICE = None  # Device for encoding (None = auto-detect: CUDA if available, else CPU)

# =========================================================
# PCA HYPERPARAMETERS (Step 2)
# =========================================================

PCA_K = 384  # Number of principal components to retain
PCA_DEVICE = None  # Device for PCA computation (None = auto-detect: CUDA if available, else CPU)

# =========================================================
# ALIGNMENT HYPERPARAMETERS (Step 3)
# =========================================================

TRAINING_BATCH_SIZE = 32  # Batch size for training (can differ from encoding batch size)
ALIGNMENT_GROUP_SIZE = 3200  # Number of batches per group for training
ALIGNMENT_SEED = 42  # Random seed for shuffling

# =========================================================
# TRAIN/VAL SPLIT HYPERPARAMETERS (Step 4)
# =========================================================

VAL_RATIO = 0.1  # Proportion of groups to use for validation (e.g., 0.1 = 10%)
VAL_SEED = 42  # Random seed for train/val split (for reproducibility)

