import os
import pickle
import torch
import numpy as np


def incremental_pca_from_pickles(folder_path, k=384, device=None):
    """
    Performs full PCA using batch-wise covariance matrix computation.

    Each .pkl file contains a list of 50 tensors of shape (2048, 1024).

    Args:
        folder_path (str): Folder containing the .pkl files.
        k (int): Number of principal components to retain.
        device (torch.device or str): CUDA or CPU device to use.

    Returns:
        mean (torch.Tensor): Shape (1024,)
        components (torch.Tensor): Shape (1024, k)
        retained_variance (float): Fraction of variance retained by top-k components
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.pkl')])
    dtype = torch.float32

    total_samples = 0
    running_mean = None

    # === Step 1: Compute Global Mean ===
    for file in files:
        print(f"Processing mean for file: {file}")
        with open(os.path.join(folder_path, file), "rb") as f:
            tensor_list = pickle.load(f)  # list of 50 tensors (2048, 1024)

        X = torch.cat(tensor_list, dim=0).to(device)  # (102400, 1024)

        if running_mean is None:
            running_mean = torch.zeros(X.shape[1], dtype=dtype, device=device)

        batch_samples = X.shape[0]
        batch_mean = X.mean(dim=0)
        running_mean = (running_mean * total_samples + batch_mean * batch_samples) / (total_samples + batch_samples)
        total_samples += batch_samples

    print(f"Mean computed over {total_samples:,} samples")

    # === Step 2: Compute Covariance Matrix ===
    cov = torch.zeros((running_mean.shape[0], running_mean.shape[0]), dtype=dtype, device=device)

    total_samples = 0  # reset for covariance normalization
    for file in files:
        print(f"Processing covariance for file: {file}")
        with open(os.path.join(folder_path, file), "rb") as f:
            tensor_list = pickle.load(f)

        X = torch.cat(tensor_list, dim=0).to(device)
        X_centered = X - running_mean
        cov += X_centered.T @ X_centered
        total_samples += X.shape[0]

    cov /= total_samples
    print("Covariance matrix computed")

    # === Step 3: Eigen Decomposition ===
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)  # ascending order
    if k > eigenvectors.shape[1]:
        raise ValueError(f"Requested k={k} components but data dimensionality is {eigenvectors.shape[1]}")

    topk_components = eigenvectors[:, -k:]  # shape: (1024, k)

    # === Step 4: Explained Variance ===
    explained_variance = eigenvalues[-k:].sum().item()
    total_variance = eigenvalues.sum().item()
    retained_variance = explained_variance / total_variance

    print("Eigen decomposition complete")
    print(f"Retained variance with top {k} components: {retained_variance:.4f} ({retained_variance*100:.2f}%)")

    return running_mean.cpu(), topk_components.cpu()

def project_array(array: np.ndarray, mean: torch.Tensor, components: torch.Tensor) -> np.ndarray:
    """
    Projects a (N, 1024)-shaped NumPy array using PCA components (mean and components in torch).

    Args:
        array (np.ndarray): Input array to project, shape (N, 1024)
        mean (torch.Tensor): Mean used for centering, shape (1024,)
        components (torch.Tensor): PCA components, shape (1024, k)

    Returns:
        np.ndarray: Projected array of shape (N, k)
    """
    device = mean.device

    # Convert NumPy array to torch tensor and move to the same device
    tensor = torch.from_numpy(array).to(device)

    # Perform projection: (tensor - mean) @ components
    projected = (tensor - mean) @ components

    # Move back to CPU and convert to NumPy
    return projected.cpu().numpy()

def save_pca_model(mean, components, out_path="pca_model.pkl"):
    """
    Saves the PCA model (mean and components) to a pickle file.
    
    Stores the mean vector and principal component matrix in a dictionary
    format that can be loaded later using load_pca_model(). This allows
    the PCA transformation to be applied to new data without recomputing
    the decomposition.
    
    Args:
        mean (torch.Tensor): Mean vector used for centering, shape (dim,).
        components (torch.Tensor): Principal component matrix, shape (dim, k).
        out_path (str): Path where the PCA model will be saved. Defaults to "pca_model.pkl".
    
    Example:
        >>> mean = torch.randn(1024)
        >>> components = torch.randn(1024, 384)
        >>> save_pca_model(mean, components, "my_pca_model.pkl")
    """
    with open(out_path, "wb") as f:
        pickle.dump({
            "mean": mean,
            "components": components
        }, f)
    print(f"PCA model saved to: {out_path}")


def load_pca_model(path="pca_model_old.pkl"):
    """
    Loads the PCA model (mean and components) from a pickle file.
    
    Reads a previously saved PCA model that was stored using save_pca_model().
    The file should contain a dictionary with 'mean' and 'components' keys.
    
    Args:
        path (str): Path to the pickle file containing the PCA model.
            Defaults to "pca_model_old.pkl".
    
    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple of (mean, components) where:
            mean: Mean vector for centering, shape (dim,).
            components: Principal component matrix, shape (dim, k).
    
    Example:
        >>> mean, components = load_pca_model("data/wiki_it/pca_model.pkl")
        >>> mean.shape  # torch.Size([1024])
        >>> components.shape  # torch.Size([1024, 384])
    """
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model["mean"], model["components"]


def project_tensor(tensor, mean, components, device=None):
    """
    Projects a (N, 1024)-shaped tensor using PCA components.

    Args:
        tensor (torch.Tensor): Input tensor to project.
        mean (torch.Tensor): Mean used for centering.
        components (torch.Tensor): PCA components (1024, k)
        device (str or torch.device): Device for projection.

    Returns:
        torch.Tensor: Projected tensor of shape (N, k)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tensor = tensor.to(device)
    mean = mean.to(device)
    components = components.to(device)

    return (tensor - mean) @ components


def apply_pca_to_all_pickles(input_folder, output_folder, mean, components, k=384, device=None):
    """
    Applies PCA projection to all .pkl files in input_folder and saves the reduced data to output_folder.

    Args:
        input_folder (str): Folder with original pickle files.
        output_folder (str): Folder to save PCA reduced pickle files.
        mean (torch.Tensor): Mean tensor for centering.
        components (torch.Tensor): PCA components matrix.
        k (int): Number of components.
        device (torch.device or str): Device for computations.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_folder, exist_ok=True)

    files = sorted([f for f in os.listdir(input_folder) if f.endswith('.pkl')])

    for file in files:
        print(f"Applying PCA to file: {file}")
        with open(os.path.join(input_folder, file), "rb") as f:
            tensor_list = pickle.load(f)  # list of 50 tensors (2048, 1024)

        # Concatenate to shape (102400, 1024)
        X = torch.cat(tensor_list, dim=0).to(device)

        # Project using PCA components: (102400, k)
        X_pca = (X - mean.to(device)) @ components.to(device)

        # Split back into list of 50 tensors (each 2048, k)
        split_tensors = torch.split(X_pca, 2048, dim=0)

        # Move tensors to CPU and convert to list for pickling
        reduced_list = [t.cpu() for t in split_tensors]

        # Save to output folder with the same file name
        out_path = os.path.join(output_folder, file)
        with open(out_path, "wb") as f:
            pickle.dump(reduced_list, f)

        print(f"Saved PCA reduced data to {out_path}")


if __name__ == "__main__":

    folder = "./data/mc4/labels/"
    os.makedirs("./data/mc4/pca_labels", exist_ok=True)
    out_model_path = "./data/mc4/pca_models/pca_model.pkl"
    k = 384
    # Auto-detect device: use CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Step 1: Compute PCA model
    mean, components = incremental_pca_from_pickles(folder_path=folder, k=k, device=device)
    save_pca_model(mean, components, out_model_path)

    # Step 2: Apply PCA to all pickles and save reduced tensors
    apply_pca_to_all_pickles(input_folder=folder, output_folder="./data/mc4/pca_labels/", mean=mean, components=components, k=k, device=device)


