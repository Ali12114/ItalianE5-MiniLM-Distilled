"""
Main script for complete MC4 data preparation pipeline.

This script automates all 4 steps of data preparation:
1. Encode sentences using a sentence transformer model
2. Apply PCA dimensionality reduction to embeddings
3. Shuffle and align sentences with their embeddings
4. Split groups into train/val sets and organize them

All configuration is loaded from config.py.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

import config
from conversions import encode, shuffle_and_align, split_groups_into_train_val
from pca import (
    incremental_pca_from_pickles,
    save_pca_model,
    apply_pca_to_all_pickles
)
import torch


def step1_encode():
    """
    Step 1: Encode sentences using the specified model.
    
    Reads sentences from CSV, encodes them in batches, and saves embeddings
    in organized groups.
    """
    print("\n" + "="*80)
    print("STEP 1: Encoding Sentences")
    print("="*80)
    
    print(f"Model: {config.ENCODE_MODEL}")
    print(f"Input CSV: {config.INPUT_CSV}")
    print(f"Output directory: {config.LABELS_DIR}")
    print(f"Batch size: {config.ENCODE_BATCH_SIZE}")
    print(f"Group size: {config.ENCODE_GROUP_SIZE}")
    # Auto-detect device if config specifies None or use config value
    if config.ENCODE_DEVICE is None:
        encode_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected device: {encode_device}")
    else:
        encode_device = config.ENCODE_DEVICE
    
    print(f"Device: {encode_device}")
    
    # Create output directory if it doesn't exist
    os.makedirs(config.LABELS_DIR, exist_ok=True)
    
    encode(
        model=config.ENCODE_MODEL,
        input_csv=config.INPUT_CSV,
        output_dir=config.LABELS_DIR,
        batch_size=config.ENCODE_BATCH_SIZE,
        group_size=config.ENCODE_GROUP_SIZE,
        starting_group=config.ENCODE_STARTING_GROUP,
        device=encode_device
    )
    
    print("Step 1 complete!\n")


def step2_pca():
    """
    Step 2: Apply PCA dimensionality reduction to embeddings.
    
    Computes PCA model from the encoded embeddings and applies it to reduce
    dimensionality (e.g., from 1024 to 384 dimensions).
    """
    print("\n" + "="*80)
    print("STEP 2: Applying PCA")
    print("="*80)
    
    print(f"Input folder: {config.LABELS_DIR}")
    print(f"Output folder: {config.PCA_LABELS_DIR}")
    print(f"PCA model path: {config.PCA_MODELS_DIR}/pca_model.pkl")
    print(f"Number of components (k): {config.PCA_K}")
    # Auto-detect device if config specifies None or use config value
    if config.PCA_DEVICE is None:
        pca_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Auto-detected PCA device: {pca_device}")
    else:
        pca_device = config.PCA_DEVICE
    
    print(f"Device: {pca_device}")
    
    # Create output directories
    os.makedirs(config.PCA_LABELS_DIR, exist_ok=True)
    os.makedirs(config.PCA_MODELS_DIR, exist_ok=True)
    
    pca_model_path = os.path.join(config.PCA_MODELS_DIR, "pca_model.pkl")
    # Convert device string to torch.device, but ensure it's valid
    if pca_device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(pca_device)
    
    # Step 2a: Compute PCA model
    print("\n--- Computing PCA model ---")
    mean, components = incremental_pca_from_pickles(
        folder_path=config.LABELS_DIR,
        k=config.PCA_K,
        device=device
    )
    save_pca_model(mean, components, pca_model_path)
    
    # Step 2b: Apply PCA to all pickles
    print("\n--- Applying PCA to all embeddings ---")
    apply_pca_to_all_pickles(
        input_folder=config.LABELS_DIR,
        output_folder=config.PCA_LABELS_DIR,
        mean=mean,
        components=components,
        k=config.PCA_K,
        device=device
    )
    
    print("Step 2 complete!\n")


def step3_shuffle_and_align():
    """
    Step 3: Shuffle and align sentences with their embeddings.
    
    Loads sentences and PCA-reduced embeddings, shuffles them together,
    batches them, and saves in organized groups.
    """
    print("\n" + "="*80)
    print("STEP 3: Shuffling and Aligning")
    print("="*80)
    
    # Create a temporary directory for aligned groups (before train/val split)
    temp_aligned_path = os.path.join(config.ALIGNED_LABELS_BASE, "temp_groups")
    
    print(f"Sentences CSV: {config.INPUT_CSV}")
    print(f"PCA labels: {config.PCA_LABELS_DIR}")
    print(f"Temporary output: {temp_aligned_path}")
    print(f"Training batch size: {config.TRAINING_BATCH_SIZE}")
    print(f"Group size: {config.ALIGNMENT_GROUP_SIZE}")
    
    shuffle_and_align(
        batch_size=config.TRAINING_BATCH_SIZE,
        labels_path=config.PCA_LABELS_DIR,
        aligned_base_path=temp_aligned_path,
        sentences_csv_path=config.INPUT_CSV,
        group_size=config.ALIGNMENT_GROUP_SIZE,
        seed=config.ALIGNMENT_SEED
    )
    
    print("Step 3 complete!\n")
    
    return temp_aligned_path


def step4_split_train_val(temp_aligned_path):
    """
    Step 4: Split groups into train/val sets and organize them.
    
    Takes all groups from the temporary aligned path, splits them into
    train/val sets, and reorganizes them with continuous numbering.
    """
    print("\n" + "="*80)
    print("STEP 4: Splitting into Train/Val")
    print("="*80)
    
    print(f"Input groups: {temp_aligned_path}")
    print(f"Output base: {config.ALIGNED_LABELS_BASE}")
    print(f"Dataset name: {config.DATASET_NAME}")
    print(f"Validation ratio: {config.VAL_RATIO}")
    
    split_groups_into_train_val(
        groups_base_path=temp_aligned_path,
        output_base_path=config.ALIGNED_LABELS_BASE,
        dataset_name=config.DATASET_NAME,
        val_ratio=config.VAL_RATIO,
        seed=config.VAL_SEED
    )
    
    # Optionally clean up temporary directory
    # Uncomment the following lines if you want to remove temp_groups after splitting
    # import shutil
    # print(f"\nCleaning up temporary directory: {temp_aligned_path}")
    # shutil.rmtree(temp_aligned_path)
    
    print("Step 4 complete!\n")


def main():
    """
    Main function that orchestrates all 4 steps of data preparation.
    """
    print("\n" + "="*80)
    print("MC4 Data Preparation Pipeline")
    print("="*80)
    print(f"\nConfiguration loaded from config.py")
    print(f"Dataset will be saved to: {config.ALIGNED_LABELS_BASE}/{config.DATASET_NAME}")
    print("\nStarting pipeline...")
    
    try:
        # Step 1: Encode sentences
        step1_encode()
        
        # Step 2: Apply PCA
        step2_pca()
        
        # Step 3: Shuffle and align
        temp_aligned_path = step3_shuffle_and_align()
        
        # Step 4: Split into train/val
        step4_split_train_val(temp_aligned_path)
        
        print("\n" + "="*80)
        print("ALL STEPS COMPLETE!")
        print("="*80)
        print(f"\nFinal dataset location:")
        print(f"  Train: {config.ALIGNED_LABELS_BASE}/{config.DATASET_NAME}/train/")
        print(f"  Val: {config.ALIGNED_LABELS_BASE}/{config.DATASET_NAME}/val/")
        print("\nYou can now use this dataset for training!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed at step")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="MC4 data preparation pipeline - automates encoding, PCA, alignment, and train/val split"
    )
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 2, 3, 4],
        default=None,
        help="Run only a specific step (1=encode, 2=PCA, 3=align, 4=split). If not specified, runs all steps."
    )
    parser.add_argument(
        "--skip",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4],
        default=[],
        help="Skip specific steps (can specify multiple, e.g., --skip 1 2)"
    )
    
    args = parser.parse_args()
    
    if args.step:
        # Run only specific step
        print(f"\nRunning only Step {args.step}\n")
        if args.step == 1:
            step1_encode()
        elif args.step == 2:
            step2_pca()
        elif args.step == 3:
            temp_path = step3_shuffle_and_align()
            print(f"Temporary aligned path: {temp_path}")
        elif args.step == 4:
            # For step 4, we need the temp path - assume it exists
            temp_path = os.path.join(config.ALIGNED_LABELS_BASE, "temp_groups")
            if not os.path.exists(temp_path):
                print(f"ERROR: Temporary aligned path not found: {temp_path}")
                print("Please run Step 3 first or provide the path manually.")
                sys.exit(1)
            step4_split_train_val(temp_path)
    else:
        # Run all steps, skipping specified ones
        if args.skip:
            print(f"\nSkipping steps: {args.skip}\n")
        
        try:
            if 1 not in args.skip:
                step1_encode()
            else:
                print("Skipping Step 1 (Encoding)\n")
            
            if 2 not in args.skip:
                step2_pca()
            else:
                print("Skipping Step 2 (PCA)\n")
            
            if 3 not in args.skip:
                temp_aligned_path = step3_shuffle_and_align()
            else:
                print("Skipping Step 3 (Alignment)\n")
                temp_aligned_path = os.path.join(config.ALIGNED_LABELS_BASE, "temp_groups")
            
            if 4 not in args.skip:
                step4_split_train_val(temp_aligned_path)
            else:
                print("Skipping Step 4 (Train/Val Split)\n")
            
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETE!")
            print("="*80)
            print(f"\nFinal dataset location:")
            print(f"  Train: {config.ALIGNED_LABELS_BASE}/{config.DATASET_NAME}/train/")
            print(f"  Val: {config.ALIGNED_LABELS_BASE}/{config.DATASET_NAME}/val/")
            print("\nYou can now use this dataset for training!")
            print("="*80 + "\n")
            
        except Exception as e:
            print(f"\nERROR: Pipeline failed")
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

