# Distilled MiniLM Italian Training and Evaluation

This documentation provides detailed instructions for training and evaluating the `MiniLM-L6-v2` sentence embedder fine-tuned for the Italian language.  
It includes a step-by-step guide to help developers set up the environment, prepare data, train the model, and evaluate it on MTEB tasks specific to Italian.

---

## Overview

The training pipeline consists of four main stages:

1. **Teacher Model Embeddings:** Compute embeddings using the teacher model.
2. **PCA Reduction:** Apply PCA to reduce the dimensionality of the teacher embeddings to match the student model.
3. **Shuffling and Alignment:** Shuffle and align PCA embeddings for better training stability.
4. **Knowledge Distillation Training:** Train the `MiniLM-L6-v2` model using PCA-reduced teacher embeddings as supervision.

---

## IMPORTANT — Step 1: Data Preparation

Before training, make sure that preprocessed data is available. Use the data preparation script provided previously in D2+D3 deliverables:

🔗 [Download Data Preparation Code](https://drive.google.com/file/d/1JAFclnAUFU_kRsKf-VWll8WPFmRPKRGT/view)

If you already have the code and dependencies installed, you don’t need to re-download it.  
However, apply the following configuration changes:

### ✅ Changes in `Config.yaml`

1. **Updated Split:**  
   Change

   ```yaml
   download.split: "train"
   ```

   to

   ```yaml
   download.split: "train[:10%]"
   ```

   (This limits the dataset to 10% for faster testing and development.)

2. **Disable Filtering Steps by Default:**  
   In `general.steps`, set both `keyword_filter` and `zero_shot` to `false`.  
   This ensures only the **download** and **process** steps are active by default.

The pipeline will now download and process text into sentence-level chunks unless other stages are manually enabled.

**Recommended:**  
For convenience, a ready-to-use configuration file + code is provided in the `dataPrep` folder.  
You can use it for data preparation [Please note the only change in data preparation is configuration settings].
Please preprocess data again using dataPrep script. Afterwards, you will have .csv file with name _data_filtered_zero_shot_chunk.csv_. Please use it for training.

---

### Optional: Use Preprocessed Data Directly

If you prefer to skip preprocessing and use preprocessed data directly, download it here:  
🔗 [Preprocessed Data (Google Drive)](https://drive.google.com/file/d/13eds80HqF58w_HCns2Vj3j20o4nbcN7w/view?usp=sharing)

---

## Step 2: Training the Model

Once data is ready, you can begin training. The script automatically performs the four sequential steps mentioned earlier.

Training is done for **10 epochs**.  
On an NVIDIA A6000 GPU, each of the first three steps takes about **4 hours**, and each epoch takes about **2 hours**.

---

## Installation

Make sure you have **Python 3.12.2** installed.

```bash
# Create virtual environment (optional but recommended)
python3 -m venv .venv

# Activate the environment
# Linux / macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

See `requirement.txt` for the complete list of dependencies.

---

## 🧩 Prepare Golden Embeddings for MiniLM-L6-v2

This is the first step in knowledge distillation.

1. Navigate to the embedding preparation directory:

   ```
   cd ./embedding_prep/italian
   ```

2. Open `config.py` and set your dataset path:

   ```python
   INPUT_CSV = "PATH_TO_YOUR_DATASET"
   (Please make sure to use data preprocessed during data preparation script)
   ```

3. Run the embedding preparation script:

   ```
   python3 main.py
   ```

   (This step takes around 4 hours. Progress will be visible in the terminal.)

4. After completion, return to the root directory.

5. Open `organized_training.py` and find the variable `base_path_emb` at the bottom of the file.  
   Replace it with the following path (adjusting to your directory):

   ```python
   base_path_data = "[YOUR_ROOT_DIRECTORY]/embedding_prep/italian/data/aligned_labels/my_data"
   # Please make sure to replace the path with a folder path (my_data) you can find inside aligned labels.
   # It will have embeddings for training and validation.
   ```

6. Start model training:

   ```
   python3 organized_training.py
   ```

7. During training, model weights will be saved automatically in:
   ```
   runs/version0/
   ```

Each epoch takes approximately 2 hours on an A6000 GPU.  
After 10 epochs, the model is ready for evaluation.

---

## Evaluation

To evaluate your model on MTEB tasks:

1. Locate your trained checkpoint (`.ckpt`) file from the `runs/versionx/checkpoints`, where x can be (0,1,2,...) folder.
2. Open `mteb_evaluation.py` and replace the model path as shown:
   ```
   model = get_italian_miniLM("PATH_TO_YOUR_MODEL.ckpt")
   ```

Run the evaluation script to compute scores across MTEB Italian tasks.

---

## Evaluation on Provided Model

We have provided MiniLM-Italian folder compatible with hugging face, it has the trained model. This trained hugging face compatible model can be evaluated using mteb_eval_hf_comp.py file.

Make sure to replace

```
      model = get_italian_miniLM('./MiniLM-Italian')
      # with your model path
```

python3 mteb_eval_hf_comp.py

# Prepare your hugging face compatible model for deployment.

Once you are ready for deployment, you would need to convert your model to hugging face compatible model.
For this please use script _export_hf_comp.py_

Important: Please make sure to change ckpt_path and export_dir, with your paths. You can find these two
variables at the end of export_hf_comp.py script.

Once you have hugging face compatible model we are ready for deployment. Please note you can easily
evaluate hugging face compatible model using script mteb_eval_hf_comp.py

# 🚀 Deployment on Infinity Server

## This section provides step-by-step instructions to deploy a Sentence Transformer or embedding model on the **Infinity Server** using **Docker**.

## 🧩 1. Requirements

Make sure the following are installed on your system:

- **Docker** (version `28.3.3` or newer)
- **NVIDIA GPU drivers** (if using GPU)
- **nvidia-container-toolkit** (for `--gpus all` support)
- Your **model files** (local or remote from Hugging Face)

---

## ⚙️ 2. Install NVIDIA Container Toolkit (GPU setup)

If you plan to use GPU acceleration, install and configure NVIDIA Docker support:

```bash
# Install prerequisites
sudo apt-get update
sudo apt-get install -y curl gnupg2 apt-transport-https ca-certificates

# Add NVIDIA repository key & list
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:11.1.1-base-ubuntu20.04 nvidia-smi
```

✅ If you see GPU information printed by nvidia-smi, the setup is correct.

## Run Infinity with Docker

The following command launches the Infinity server with embedding model:

```

# Variables

port=7997
model1=michaelfeil/bge-small-en-v1.5
model2=mixedbread-ai/mxbai-rerank-xsmall-v1
volume=$PWD/data # cache folder

# Run Infinity

docker run -it --gpus all \
 -v "$volume":/app/.cache \
   -p "$port":"$port" \
   michaelf34/infinity:latest \
   v2 \
   --model-id "$model1" \
 --model-id "$model2" \
   --port "$port"

```

# Deploy a Local Model

To serve a local model, mount your model directory inside the container:

```
# Example: Host model path
HOST_MODEL_DIR=/home/haiderali/Desktop/miniLM-Model/MiniLM-Italian

# Run Infinity with local model
docker run -it --gpus all \
  -v "$HOST_MODEL_DIR":/models/MiniLM-Italian \
  -v "$PWD"/data:/app/.cache \
  -p 8081:8081 \
  michaelf34/infinity:latest \
  v2 --model-id "/models/MiniLM-Italian" --port 8081
```

_Important_

Always provide the container path (/models/...) to --model-id.

Do not use local paths like ./model; Infinity will interpret them as remote Hugging Face IDs. Rather use full path like /home/haider
_Please make sure to replace HOST_MODEL_DIR path with your model path_

# Verify Deployment

Once running, open the following URLs in your browser:

Deployed models: http://0.0.0.0:8081/models

# Smoke test

After running server (Please make sure server keep running), please run smoke_test.py file in other terminal.

## Possible Error and their resolution

1. docker: permission denied while trying to connect to the docker API at unix:///var/run/docker.sock

### To solve this error please run these commands:

```
sudo groupadd docker   # creates the group if it doesn't exist
sudo usermod -aG docker $USER
```

To get $USER you need to use comman

```
whoami
```

For example if it prints ecuser
then use this command

```
sudo usermod -aG docker ecuser
```

## Important Final Note

We intentionally split the full pipeline into multiple stages for easier debugging.
If an error occurs, rerun only the failed stage instead of the entire process.
