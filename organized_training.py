from eval import get_italian_miniLM
from sentence_transformers.util import batch_to_device
from sentence_datasets import LabelledBatchedSentencesDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import random
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from transformers import get_linear_schedule_with_warmup

torch.set_float32_matmul_precision("high")

# from sentence_transformers import SentenceTransformer


def flat_collate_fn(batch):
    """
    Collate function that extracts a single item from the batch.

    Since the underlying dataset already returns pre-batched items (batches of 32),
    this function simply returns the first (and only) item from the DataLoader's
    batch list. This avoids double-batching and preserves the dataset's batch structure.

    Args:
        batch: A list containing a single pre-batched item from the dataset.
            The item should be a tuple of (sentences, target_embeddings) where
            sentences is a list of strings and target_embeddings is a tensor.

    Returns:
        The first item from the batch (typically a tuple of sentences and embeddings).

    Example:
        >>> batch = [(sentences, embeddings)]  # DataLoader wraps in a list
        >>> data = flat_collate_fn(batch)  # Returns (sentences, embeddings)
    """
    return batch[0]


def set_seed(seed=42):
    """
    Sets random seeds for reproducibility across random number generators.

    Ensures deterministic behavior by setting seeds for Python's random module,
    PyTorch's CPU and CUDA random number generators. Also configures CuDNN
    to use deterministic algorithms and disables benchmarking for consistency.

    Args:
        seed (int): Random seed value to use. Defaults to 42.

    Note:
        Setting cudnn.deterministic=True and benchmark=False may reduce performance
        but ensures reproducible results across runs.

    Example:
        >>> set_seed(42)
        >>> # All random operations will now be reproducible
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class DistillationLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for knowledge distillation training.

    This module trains a student model (typically a smaller, faster model) to
    mimic the embeddings produced by a teacher model through knowledge distillation.
    The student learns by minimizing a loss function that compares its embeddings
    to target embeddings (from the teacher).

    The module handles:
    - Forward pass: Computes embeddings for input sentences
    - Training step: Computes loss and logs metrics
    - Validation step: Evaluates on validation data
    - Optimizer configuration: Sets up Adam optimizer with linear decay scheduler
    """

    def __init__(self, student, loss_fn, lr=1e-3, eval_fns=None):
        """
        Initializes the distillation lightning module.

        Args:
            student: SentenceTransformer model to train (the student in distillation).
                This is the model that will learn to produce similar embeddings
                to the teacher's embeddings.
            loss_fn (callable): Loss function that takes (student_embeddings, target_embeddings)
                and returns a scalar tensor. Typically cosine_loss or comparative_loss.
            lr (float): Learning rate for the Adam optimizer. Defaults to 1e-3.
            eval_fns (list[callable], optional): List of evaluation functions to run
                during training. Each function should take (student_emb, target_emb)
                and return a scalar value. Results are logged as train_eval_{fn_name}.
                Defaults to empty list.
        """
        super().__init__()
        self.student = student
        self.loss_fn = loss_fn
        self.lr = lr
        self.eval_fns = eval_fns or []

    def forward(self, x: list[str]):
        """
        Forward pass: computes embeddings for input sentences.

        Args:
            x (list[str]): List of sentences to encode. The length corresponds
                to the batch size (typically 32 from the dataset's pre-batching).

        Returns:
            torch.Tensor: Embeddings tensor of shape (batch_size, embedding_dim).
        """
        return self._compute_embeddings(sentences=x)

    def _compute_embeddings(self, sentences: list[str]) -> torch.Tensor:
        """
        Internal method to compute embeddings for sentences using the student model.

        Tokenizes the input sentences, moves tokens to the appropriate device,
        and passes them through the student model to get sentence embeddings.
        This follows the standard SentenceTransformer encoding pipeline.

        Args:
            sentences (list[str]): List of sentences to encode.

        Returns:
            torch.Tensor: Sentence embeddings tensor of shape (len(sentences), embedding_dim).
        """
        # model.tokenize() returns a dict of input_ids, token_type_ids and attention_mask, we can feed it directly to model()
        tokens = self.student.tokenize(sentences)

        # function used in official code of SentenceTransformer after tokenization.
        tokens = batch_to_device(tokens, self.device)

        embeddings = self.student(tokens)["sentence_embedding"]

        # model.forward() also returns a dict of things including token_embeddings, for us sentence_embedding is relevant.
        return embeddings

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Performs a single training step.

        Computes student embeddings, calculates loss against target embeddings,
        and logs metrics. Also runs any evaluation functions if provided.

        Args:
            batch: A tuple of (sentences, target_embeddings) where:
                - sentences: list[str] of batch_size sentences
                - target_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
                  containing teacher embeddings (or PCA-reduced embeddings)
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed loss value (scalar).
        """
        sentences, target_embeddings = batch

        student_embeddings = self(sentences)

        loss = self.loss_fn(student_embeddings, target_embeddings)

        self.log("train_loss", loss, on_step=True, on_epoch=True)

        for eval_fn in self.eval_fns:
            value = eval_fn(student_embeddings, target_embeddings)
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.log(f"train_eval_{eval_fn.__name__}",
                     value, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Performs a single validation step.

        Computes student embeddings and validation loss. The loss is logged
        at the epoch level (not step level) for validation.

        Args:
            batch: A tuple of (sentences, target_embeddings) where:
                - sentences: list[str] of batch_size sentences
                - target_embeddings: torch.Tensor of shape (batch_size, embedding_dim)
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The computed validation loss value (scalar).
        """
        sentences, target_embeddings = batch
        student_embeddings = self(sentences)
        loss = self.loss_fn(student_embeddings, target_embeddings)
        self.log("val_loss", loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Sets up an Adam optimizer with a linear learning rate decay schedule.
        The learning rate decays linearly from the initial lr to 0 over the
        course of training, with no warmup period.

        Returns:
            dict: Dictionary containing optimizer and lr_scheduler configuration
                for PyTorch Lightning. The scheduler steps after each batch.
        """
        optimizer = optim.Adam(self.student.parameters(), lr=self.lr)

        # Total number of training steps
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = 0  # No warmup, decay starts from step 0

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # Step every batch
            "frequency": 1,
            "name": "linear_decay"
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}


class DistillationDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning data module for distillation training.

    Manages loading and organizing training and validation datasets for knowledge
    distillation. The data structure is expected to be organized as:

    base_path/
        train/
            0/{sentences.pkl, embeddings.pkl}
            1/{sentences.pkl, embeddings.pkl}
            ...
        val/
            0/{sentences.pkl, embeddings.pkl}
            ...

    Each group folder contains pre-batched sentences and their corresponding
    target embeddings, as created by conversions.py.
    """

    def __init__(self, base_path, seed):
        """
        Initializes the data module.

        Args:
            base_path (str): Path to the base directory containing 'train' and 'val'
                subdirectories. Each subdirectory should contain numbered group folders
                (0, 1, 2, ...) with sentences.pkl and embeddings.pkl files.
                This structure is created by conversions.py's shuffle_and_align() function.
            seed (int): Random seed (currently unused but stored for potential future use).
        """
        super().__init__()
        self.seed = seed
        self.base_path = base_path

    def setup(self, stage=None):
        """
        Sets up train and validation datasets.

        Called by PyTorch Lightning to initialize the datasets. Creates
        LabelledBatchedSentencesDataset instances for both train and validation sets.

        Args:
            stage (str, optional): Stage of training (e.g., 'fit', 'test'). Not used here.
                Defaults to None.

        Note:
            The datasets are loaded from {base_path}/train and {base_path}/val,
            where each directory contains numbered group folders with the data.
        """
        self.train_dataset = LabelledBatchedSentencesDataset(
            base_path=f"{self.base_path}/train")
        self.val_dataset = LabelledBatchedSentencesDataset(
            base_path=f"{self.base_path}/val")

    def train_dataloader(self):
        """
        Returns the training DataLoader.

        Returns:
            DataLoader: DataLoader for training data. Uses batch_size=1 because
                the dataset already provides pre-batched items (batches of 32).
                The flat_collate_fn extracts the single item from the DataLoader's list.

        Note:
            shuffle=False because the data is already shuffled during preprocessing.
            batch_size=1 means each DataLoader iteration returns one pre-batched item
            from the dataset (which itself contains a batch of 32 sentences/embeddings).
        """
        return DataLoader(self.train_dataset, batch_size=1, shuffle=False, collate_fn=flat_collate_fn)

    def val_dataloader(self):
        """
        Returns the validation DataLoader.

        Returns:
            DataLoader: DataLoader for validation data. Uses batch_size=1 because
                the dataset already provides pre-batched items (batches of 32).
                The flat_collate_fn extracts the single item from the DataLoader's list.

        Note:
            Same configuration as train_dataloader but for validation data.
        """
        return DataLoader(self.val_dataset, batch_size=1, shuffle=False, collate_fn=flat_collate_fn)


if __name__ == "__main__":

    base_path_emb = "./embedding_prep/italian/data/mc4/aligned_labels/my_dataset"

    master_seed = 42
    set_seed(master_seed)

    # Determine device automatically
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    student = get_italian_miniLM().to(device)

    data_module = DistillationDataModule(
        base_path_emb, seed=master_seed)

    def cosine_sim(x1, x2):
        return nn.functional.cosine_similarity(x1, x2, dim=1).mean()

    def cosine_loss(x1, x2):
        return 1-cosine_sim(x1, x2)

    def comparative_loss(x1, x2):
        # Normalize each row to unit norm → cosine similarity = dot product
        x1_norm = nn.functional.normalize(x1, p=2, dim=1)
        x2_norm = nn.functional.normalize(x2, p=2, dim=1)

        # Compute cosine similarity matrices: (batch, batch)
        sim_x1 = x1_norm @ x1_norm.T
        sim_x2 = x2_norm @ x2_norm.T

        # Mean squared error between similarity matrices
        loss = nn.functional.mse_loss(sim_x1, sim_x2)

        return loss

    lightning_model = DistillationLightningModule(
        student=student,
        loss_fn=cosine_loss,
        lr=5e-5,
        eval_fns=[]
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # <- Metric to monitor
        mode="min",  # <- "min" means lower is better (loss)
        save_top_k=1,  # <- Only keep best model
        filename="{step}_{val_loss:.8f}",  # Optional custom filename
        every_n_epochs=None,  # <- Optional (for full-epoch saving)
        save_on_train_epoch_end=False  # <- Important when validating mid-epoch
    )

    # Configure accelerator based on device availability
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = 1  # Use only 1 GPU
    else:
        accelerator = "cpu"
        devices = 1

    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        max_epochs=10,
        # validate every 10000 train batches (fixed from 2000 to be more reasonable)
        val_check_interval=10000,
        default_root_dir="checkpoints",
        log_every_n_steps=1,
        accelerator=accelerator,
        devices=devices,
        enable_checkpointing=True,
        logger=pl.loggers.TensorBoardLogger("runs")
    )

    trainer.fit(lightning_model, datamodule=data_module)
