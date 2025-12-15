import os
from collections import OrderedDict

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from embedding_prep.italian.pca import load_pca_model, project_array
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import mteb


class TruncatedE5:
    """
    Wrapper around an E5 model that truncates sentences to 170 words.
    
    This class ensures fair comparison by limiting sentence length to 170 words,
    preventing longer sentences from having an advantage in embedding quality.
    Useful when comparing models with different maximum sequence lengths.
    """
    
    def __init__(self, base_model):
        """
        Initializes the TruncatedE5 wrapper.
        
        Args:
            base_model: The base SentenceTransformer model (typically E5) to wrap.
        """
        self.base_model = base_model
    
    def encode(self, sentences,  **kwargs):
        """
        Encodes sentences after truncating them to 170 words.
        
        Truncates each sentence to the first 170 words before encoding.
        This ensures consistent input length for fair model comparison.
        
        Args:
            sentences (list[str]): List of sentences to encode.
            **kwargs: Additional arguments passed to base_model.encode().
        
        Returns:
            numpy.ndarray: Embeddings array of shape (len(sentences), embedding_dim).
        """
        sentences = [ " ".join(sentence.split(" ")[:170]) for sentence in sentences]
        
        return self.base_model.encode(sentences,  convert_to_numpy=True, **kwargs)
        
    


class E5LargePCA:
    """
    Wrapper around E5 model that applies PCA dimensionality reduction to embeddings.
    
    Encodes sentences using the base E5 model, then projects the embeddings
    through PCA to reduce dimensionality (e.g., from 1024 to 384 dimensions).
    This is useful for fair comparison when the target embedding space is smaller.
    """
    
    def __init__(self, base_model, pca_model, fair=True):
        """
        Initializes the E5LargePCA wrapper.
        
        Args:
            base_model: The base SentenceTransformer model (typically E5-large) to use.
            pca_model: Tuple of (mean, components) for PCA transformation.
                mean: torch.Tensor of shape (original_dim,) for centering.
                components: torch.Tensor of shape (original_dim, k) with PCA components.
            fair (bool): If True, truncates sentences to 170 words before encoding
                for fair comparison. Defaults to True.
        """
        self.model_name = "e5-large-pca"
        self.base_model = base_model
        self.pca_model = pca_model
        self.fair = fair
    
    def encode(self, sentences, batch_size=32, **kwargs):
        """
        Encodes sentences and applies PCA dimensionality reduction.
        
        First encodes sentences using the base model, then projects the embeddings
        through PCA to reduce dimensionality. Optionally truncates sentences
        to 170 words if fair=True.
        
        Args:
            sentences (list[str]): List of sentences to encode.
            batch_size (int): Batch size for encoding. Defaults to 32.
            **kwargs: Additional arguments passed to base_model.encode().
        
        Returns:
            numpy.ndarray: PCA-reduced embeddings of shape (len(sentences), k)
                where k is the number of PCA components.
        
        Note:
            The PCA mean and components are moved to CUDA for efficient computation.
            The final embeddings are returned as NumPy arrays on CPU.
        """
        if self.fair:
            sentences = [ " ".join(sentence.split(" ")[:170]) for sentence in sentences]
        np_array = self.base_model.encode(sentences, batch_size=batch_size, convert_to_numpy=True, **kwargs)
        if isinstance(np_array, torch.Tensor): # idk why its not a numpy array in the first place
            np_array = np_array.cpu().numpy()
        
        mean,components = self.pca_model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mean = mean.to(device)
        components = components.to(device)
        return project_array(np_array, mean, components)
        

def comparative_loss(s_emb, t_emb):
    """
    Computes comparative loss between student and teacher embeddings.
    
    This loss function compares the relational similarity structures of student
    and teacher embeddings rather than direct embedding similarity. It computes
    similarity matrices for both embeddings (using dot product, which equals
    cosine similarity for normalized embeddings) and measures the MSE between them.
    
    This loss encourages the student to preserve the relative relationships
    between embeddings, even if absolute values differ.
    
    Args:
        s_emb (torch.Tensor): Student embeddings, shape (N, embedding_dim).
        t_emb (torch.Tensor): Teacher embeddings, shape (N, embedding_dim).
    
    Returns:
        float: Mean squared error between the similarity matrices (scalar).
    
    Note:
        The embeddings should be normalized (L2) for dot product to equal cosine similarity.
        This function expects the embeddings to be on the same device.
    """
    # Compute relational similarity matrices (dot product ~= cosine)
    teacher_sim = t_emb @ t_emb.T  # (N x N)
    student_sim = s_emb @ s_emb.T  # (N x N)

    # Comparative relational MSE loss
    loss = F.mse_loss(student_sim, teacher_sim).item()

    return loss

def mean_cosine_sim(s_emb, t_emb):
    """
    Computes the mean cosine similarity between corresponding student and teacher embeddings.
    
    Calculates cosine similarity between each pair of corresponding embeddings
    (student[i] vs teacher[i]) and returns the average. This measures how well
    the student embeddings align with teacher embeddings in direction.
    
    Args:
        s_emb (torch.Tensor): Student embeddings, shape (N, embedding_dim).
        t_emb (torch.Tensor): Teacher embeddings, shape (N, embedding_dim).
            Must have the same shape as s_emb.
    
    Returns:
        float: Mean cosine similarity value in [-1, 1], where 1 means perfect alignment.
    
    Example:
        >>> s_emb = torch.randn(10, 384)
        >>> t_emb = torch.randn(10, 384)
        >>> similarity = mean_cosine_sim(s_emb, t_emb)
        >>> # Returns a value between -1 and 1
    """
    return F.cosine_similarity(s_emb, t_emb, dim=1).mean().item()

def track_evals(sentences:list[str], batch_size, teacher, student, eval_fns:list, device):
    """
    Evaluates student and teacher models on a set of sentences in batches.
    
    Processes sentences in batches, computes embeddings from both teacher and student
    models, and evaluates them using the provided evaluation functions. Prints
    progress and evaluation results for each batch.
    
    Args:
        sentences (list[str]): List of sentences to evaluate on.
        batch_size (int): Number of sentences to process in each batch.
        teacher: Teacher model (must have encode() method).
        student: Student model (must have encode() method).
        eval_fns (list[callable]): List of evaluation functions, each taking
            (student_embeddings, teacher_embeddings) and returning a scalar value.
        device (str): Device to run encoding on (e.g., "cuda", "cpu").
    
    Note:
        Both teacher and student embeddings are normalized before evaluation.
        Evaluation is done with torch.no_grad() for efficiency.
    
    Example:
        >>> eval_fns = [mean_cosine_sim, comparative_loss]
        >>> track_evals(sentences, batch_size=32, teacher=teacher, student=student, 
        ...             eval_fns=eval_fns, device="cuda")
    """

    for start_idx in range(0, len(sentences), batch_size):
        end_idx = min(start_idx + batch_size, len(sentences))
        batch = sentences[start_idx:end_idx]
        print(f"Processing batch {start_idx}–{end_idx}")

        with torch.no_grad():
            # Get normalized embeddings for cosine-like dot products
            t_emb = teacher.encode(batch, convert_to_tensor=True, device=device, normalize_embeddings=True)
            s_emb = student.encode(batch, convert_to_tensor=True, device=device, normalize_embeddings=True)

            # # Compute relational similarity matrices (dot product ~= cosine)
            # teacher_sim = t_emb @ t_emb.T  # (N x N)
            # student_sim = s_emb @ s_emb.T  # (N x N)
            #
            # # Comparative relational MSE loss
            # loss = F.mse_loss(student_sim, teacher_sim).item()
            evals = []
            for eval_fn in eval_fns:
                evals.append((eval_fn.__name__,eval_fn(s_emb, t_emb)))

            print(f"evals: {evals}")
def get_italian_only_tasks():
    """
    Retrieves MTEB tasks that are exclusively for Italian language.
    
    Filters MTEB (Massive Text Embedding Benchmark) tasks to include only
    those that support Italian and are single-language (not multilingual).
    This is useful for evaluating Italian-specific embedding models.
    
    Returns:
        list: List of MTEB task objects that are exclusively Italian.
    
    Note:
        Uses mteb.get_tasks() to get tasks, then filters to tasks where
        len(task.languages) == 1 to ensure they're single-language tasks.
    """
    tasks = mteb.get_tasks(languages=["ita"], modalities=["text"])
    tasks = [task for task in tasks if len(task.languages)==1]
    return tasks

def benchmark_italian_task(model, task_class):
    """
    Benchmarks a model on a specific Italian MTEB task.
    
    Finds the Italian-only task matching the given task class and runs
    the MTEB evaluation on it. This is a workaround to get the Italian
    subset of multilingual tasks that don't directly support filtering.
    
    Args:
        model: Embedding model to evaluate (must implement encode() method).
        task_class: Class type of the MTEB task to run (e.g., mteb.WikipediaRerankingMultilingual).
    
    Raises:
        AssertionError: If no matching Italian-only task is found.
    
    Note:
        This function searches through Italian-only tasks and finds the one
        that is an instance of task_class. The results are saved to "results" folder.
    
    Example:
        >>> benchmark_italian_task(model, mteb.WikipediaRerankingMultilingual)
    """
    # a very stupid way to get the italian subset of wikipediarerankingmultlingual, but ah well
    tasks = get_italian_only_tasks()
    for task in tasks:
        print(task)
    required_task = None
    for task in tasks:
        if isinstance(task, task_class):
            required_task = task
            break
    assert required_task is not None

    evaluator = mteb.MTEB(tasks=[required_task])

    evaluator.run(model, output_folder="results")

def get_model_state_dict(ckpt):
    """
    Extracts the student model's state_dict from a PyTorch Lightning checkpoint.
    
    PyTorch Lightning checkpoints store model parameters with a "student." prefix
    because the model is nested within the DistillationLightningModule. This function
    removes the prefix to get the raw model state_dict that can be loaded directly
    into a SentenceTransformer model.
    
    Args:
        ckpt (dict): PyTorch Lightning checkpoint dictionary, typically loaded
            with torch.load(). Must contain a "state_dict" key.
    
    Returns:
        OrderedDict: State dictionary with "student." prefixes removed, ready
            to be loaded into the student model.
    
    Example:
        >>> ckpt = torch.load("checkpoint.ckpt")
        >>> state_dict = get_model_state_dict(ckpt)
        >>> model.load_state_dict(state_dict)
    """
    raw_dict: OrderedDict = ckpt["state_dict"]
    new_dict = OrderedDict()
    for key, value in raw_dict.items():
        new_dict[key.removeprefix("student.")] = value
    return new_dict

def compare_similarities(sentences, student, teacher):
    """
    Compares similarity matrices between student and teacher embeddings.
    
    Computes embeddings for sentences using both models and prints various
    similarity metrics:
    1. Teacher-teacher similarity matrix (diagonal should be 1.0)
    2. Student-student similarity matrix (diagonal should be 1.0)
    3. Student-teacher cross-similarity matrix
    4. Mean cosine similarity between corresponding embeddings
    
    Useful for debugging and understanding how well the student mimics the teacher.
    
    Args:
        sentences (list[str]): List of sentences to compare.
        student: Student model (must have encode() and similarity() methods).
        teacher: Teacher model (must have encode() method).
    
    Note:
        All embeddings are converted to tensors and moved to the available device (CUDA if available, else CPU).
        The similarity matrices show pairwise similarities between all sentences.
        The ith row shows how similar sentence i is to all other sentences.
    
    Example:
        >>> compare_similarities(["sentence 1", "sentence 2"], student, teacher)
        >>> # Prints similarity matrices and mean cosine similarity
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    teacher_embeddings = teacher.encode(sentences, convert_to_tensor=True)
    if not isinstance(teacher_embeddings, torch.Tensor):
        teacher_embeddings = torch.from_numpy(teacher_embeddings).to(device)
    student_embeddings = student.encode(sentences, convert_to_tensor=True)

    # ith row indicates the similarities of ith sentence with all other sentences.
    print(student.similarity(teacher_embeddings, teacher_embeddings))
    print(student.similarity(student_embeddings, student_embeddings))
    print(student.similarity(student_embeddings, teacher_embeddings))
    print(mean_cosine_sim(teacher_embeddings, student_embeddings))


def get_fair_e5():
    """
    Creates a fair E5-large teacher model with PCA dimensionality reduction.
    
    Loads the multilingual E5-large model, applies PCA reduction using a pre-trained
    PCA model, and optionally truncates sentences to 170 words for fair comparison.
    This creates a teacher model that produces 384-dimensional embeddings (via PCA)
    from the original 1024-dimensional E5-large embeddings.
    
    Returns:
        E5LargePCA: Wrapped teacher model that encodes to PCA-reduced embeddings.
    
    Note:
        The PCA model is loaded from "data/wiki_it/pca_model.pkl".
        The model is moved to the available device (CUDA if available, else CPU) and configured for fair comparison (truncates to 170 words).
    
    Example:
        >>> teacher = get_fair_e5()
        >>> embeddings = teacher.encode(["sentence 1", "sentence 2"])
        >>> embeddings.shape  # (2, 384)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = SentenceTransformer("intfloat/multilingual-e5-large").to(device)
    pca_model = load_pca_model("data/wiki_it/pca_model.pkl")
    teacher = E5LargePCA(teacher, pca_model)
    return teacher

def get_italian_miniLM():
    """
    Loads an Italian fine-tuned MiniLM model (or returns base model if checkpoint not loaded).
    
    Loads a SentenceTransformer model, optionally from a checkpoint if provided.
    Currently, the checkpoint loading is commented out, so it returns the base
    all-MiniLM-L6-v2 model. When checkpoint loading is enabled, it extracts
    the student model weights from the PyTorch Lightning checkpoint.
    
    Args:
        ckpt_path (str): Path to the checkpoint file. Defaults to a specific checkpoint path.
            Note: Currently checkpoint loading is disabled (commented out).
    
    Returns:
        SentenceTransformer: The student model (either from checkpoint or base model),
            moved to the available device (CUDA if available, else CPU).
    
    Note:
        To enable checkpoint loading, uncomment the lines that load and apply the state_dict.
        The checkpoint should be from a DistillationLightningModule training run.
    
    Example:
        >>> model = get_italian_miniLM("path/to/checkpoint.ckpt")
        >>> embeddings = model.encode(["Ciao mondo"])
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    student = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)

    return student

if __name__=="__main__":

    tasks = [
        mteb.tasks.Clustering.multilingual.SIB200ClusteringS2S.SIB200ClusteringFast().filter_languages(languages=["ita"], exclusive_language_filter=True),
        mteb.tasks.Reranking.multilingual.WikipediaRerankingMultilingual.WikipediaRerankingMultilingual().filter_languages(languages=["ita"], exclusive_language_filter=True),
        mteb.tasks.Retrieval.multilingual.BelebeleRetrieval.BelebeleRetrieval().filter_languages(languages=["ita"], exclusive_language_filter=True),
        mteb.tasks.Retrieval.multilingual.MultiLongDocRetrieval.MultiLongDocRetrieval().filter_languages(languages=["ita"], exclusive_language_filter=True),
        mteb.tasks.STS.multilingual.STS22CrosslingualSTS.STS22CrosslingualSTS().filter_languages(languages=["ita"], exclusive_language_filter=True),
        mteb.tasks.STS.multilingual.STSBenchmarkMultilingualSTS.STSBenchmarkMultilingualSTS().filter_languages(languages=["ita"], exclusive_language_filter=True)  ,
        mteb.tasks.Retrieval.multilingual.WebFAQRetrieval.WebFAQRetrieval().filter_languages(languages=["ita"], exclusive_language_filter=True),
    ]
    
    eval = mteb.MTEB(tasks =tasks)
    
    model = get_italian_miniLM()
    
    eval.run(model)

    
    
    