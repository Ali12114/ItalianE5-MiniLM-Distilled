import os
from collections import OrderedDict

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from embedding_prep.italian.pca import load_pca_model, project_array
import torch
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
import mteb


class TruncatedE5:
    
    def __init__(self, base_model):
        self.base_model = base_model
    
    def encode(self, sentences,  **kwargs):
        
        sentences = [ " ".join(sentence.split(" ")[:170]) for sentence in sentences]
        
        return self.base_model.encode(sentences,  convert_to_numpy=True, **kwargs)
        
    


class E5LargePCA:
    
    def __init__(self, base_model, pca_model, fair=True):
        self.model_name = "e5-large-pca"
        self.base_model = base_model
        self.pca_model = pca_model
        self.fair = fair
    
    def encode(self, sentences, batch_size=32, **kwargs):
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

    # Compute relational similarity matrices (dot product ~= cosine)
    teacher_sim = t_emb @ t_emb.T  # (N x N)
    student_sim = s_emb @ s_emb.T  # (N x N)

    # Comparative relational MSE loss
    loss = F.mse_loss(student_sim, teacher_sim).item()

    return loss

def mean_cosine_sim(s_emb, t_emb):
    return F.cosine_similarity(s_emb, t_emb, dim=1).mean().item()

def track_evals(sentences:list[str], batch_size, teacher, student, eval_fns:list, device):

    for start_idx in range(0, len(sentences), batch_size):
        end_idx = min(start_idx + batch_size, len(sentences))
        batch = sentences[start_idx:end_idx]
        print(f"Processing batch {start_idx}–{end_idx}")

        with torch.no_grad():
            # Get normalized embeddings for cosine-like dot products
            t_emb = teacher.encode(batch, convert_to_tensor=True, device=device, normalize_embeddings=True)
            s_emb = student.encode(batch, convert_to_tensor=True, device=device, normalize_embeddings=True)
            evals = []
            for eval_fn in eval_fns:
                evals.append((eval_fn.__name__,eval_fn(s_emb, t_emb)))

            print(f"✓ evals: {evals}")
def get_italian_only_tasks():
    tasks = mteb.get_tasks(languages=["ita"], modalities=["text"])
    tasks = [task for task in tasks if len(task.languages)==1]
    return tasks

#[mteb.WikipediaRerankingMultilingual, mteb.STSBenchmarkMultilingualSTS, WebFAQRetrieval]
def benchmark_italian_task(model, task_class):
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

# get model's state_dict from distillation module
def get_model_state_dict(ckpt):
    raw_dict: OrderedDict = ckpt["state_dict"]
    new_dict = OrderedDict()
    for key, value in raw_dict.items():
        new_dict[key.removeprefix("student.")] = value
    return new_dict

def compare_similarities(sentences, student, teacher):
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    teacher = SentenceTransformer("intfloat/multilingual-e5-large").to(device)
    pca_model = load_pca_model("data/wiki_it/pca_model.pkl")
    teacher = E5LargePCA(teacher, pca_model)
    return teacher

def get_italian_miniLM(ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    ckpt = torch.load(ckpt_path)

    state_dict = get_model_state_dict(ckpt)

    
    student = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)
    
    student.load_state_dict(state_dict)
    
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
    # Please change this path to your own path where you have saved your checkpoint
    model = get_italian_miniLM('PLEASE CHANGE THIS PATH WITH YOUR .ckpt PATH')
    
    eval.run(model)

    
    
    