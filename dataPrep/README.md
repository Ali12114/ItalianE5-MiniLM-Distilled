# Distilled MiniLM Italian Preprocessing Pipeline

This repository provides a preprocessing pipeline for building an **Italian text dataset** from Hugging Face datasets or load the data from local directory.  
It performs four sequential steps:

1. **Download** — fetch a dataset from Hugging Face (`gsarti/clean_mc4_it`) and save it to CSV. Or use custom/local dataset from local directory.
2. **Keyword Filter** — filter rows by Italian crime-related keywords.
3. **Zero-Shot Classification** — further filter using a transformer-based zero-shot classifier (`facebook/bart-large-mnli`).
4. **Sentence Chunking** — split long texts into sentences with token length constraints using spaCy + HuggingFace tokenizer.

The resulting CSV can then be used for downstream Italian NLP tasks such as classification, embeddings, or fine-tuning.

---

## ⚙️ Installation

Clone the repository and install dependencies:

Please make sure you have Python 3.12.2 version.

```bash
# create virtual environment (optional but recommended)
python3 -m venv .venv
# Activate the environment
# In Linux OS system
source .venv/bin/activate
# In window OS system
.venv\Scripts\activate.bat

# install requirements
pip install -r requirement.txt

# Optional in case you get any error during spacy download

# Upgrade pip, setuptools, wheel
pip install --upgrade pip setuptools wheel

# Remove any corrupted cached wheel
pip cache purge

```

## Main Dependencies

Please refer to requirement.txt file for complete list of dependencies.

## Start Data Preprocessing Dataset

```bash
python data_preprocessing.py --config config.yaml
```

### Output Files

data.csv → raw downloaded dataset (tiny split).

data_filtered.csv → keyword-filtered subset.

data_filtered_zero_shot.csv → zero-shot classification filtered rows.

data_filtered_zero_shot_chunk.csv → final processed chunk-level dataset.

```bash
io:
  base_dir: "_PATH_TO_SAVE_OUTPUT_FILES_"
  download_csv: "data.csv"
  filtered_csv: "data_filtered.csv"
  zero_shot_csv: "data_filtered_zero_shot.csv"
  processed_csv: "data_filtered_zero_shot_chunk.csv"
  text_column: "text"
```

## Configurations

### Preprocessing Steps

You can change which steps to run by editing steps. **Do not flag false **download. Example:

```bash
  steps:
    download: true # Make sure it's always true
    keyword_filter: true
    zero_shot: true
    process: true
```

### Using local data

Either download from Hugging Face or point to a local CSV. The CSV must contain the column specified by io.text_column (default: text). For reference please check template.csv file. Script just need text column other columns are not needed (ref. template.csv)

```bash
download:
  hf_dataset: "gsarti/clean_mc4_it"
  config_name: "tiny"
  split: "train"
  streaming: false
  # Optional: provide an absolute path to a pre-downloaded CSV to skip downloading step
  # The CSV must contain a column named as in `io.text_column` (default: "text")
  local_csv: PATH_TO_THE_CSV
```

### Using mc4-it data

Please note: if you want to download the dataset you should set the value of local_csv to false. In that case mc4 italian data will be downloaded.

```bash
local_csv: false
```

If local_csv is a path, the pipeline uses that CSV and skips huggingface download. If local_csv is false, the pipeline downloads the dataset from Hugging Face.

### Keywords Filtering

Update keywords here. chunk_size controls how many rows are processed per batch.

```bash
keyword_filter:
  chunk_size: 100000
  keywords:
    - "omicidio"
    - "assassinio"
    - "femminicidio"
    - "aggressione"
    - "lesioni personali"
    - "violenza sessuale"
    - "stupro"
    - "omicidio colposo"
    - "tentato omicidio"
    - "furto"
    - "rapina"
    - "scasso"
    - "borseggio"
    - "effrazione"
    - "spaccio di droga"
    - "traffico di stupefacenti"
    - "detenzione di droga"
    - "riciclaggio di denaro"
    - "corruzione"
    - "estorsione"
    - "truffa"
    - "usura"
    - "frode fiscale"
    - "evasione fiscale"
    - "appropriazione indebita"
    - "pedofilia"
    - "abuso su minori"
    - "pornografia minorile"
    - "crimine organizzato"
    - "associazione mafiosa"
    - "mafia"
    - "camorra"
    - "ndrangheta"
    - "terrorismo"
    - "attentato terroristico"
    - "radicalizzazione"
    - "cellula terroristica"
    - "sequestro di persona"
    - "minacce"
    - "intimidazione"
    - "latitante"
    - "indagine penale"
    - "procedimento penale"
    - "arresto"
    - "carcere"
    - "custodia cautelare"
```

### Zero-shot classification

Configure model, device and labels. device can be 0 (GPU 0), -1 (CPU), or similar depending on your environment.

```bash
zero_shot:
  model_name: "facebook/bart-large-mnli"
  device: 0
  chunk_size: 1000
  batch_size: 8
  labels: ["crimine", "non crimine"]
  hypothesis_template: "Questo testo riguarda {}."
```

### Sentence Chunking

We use spaCy for sentence splitting and a Hugging Face tokenizer for token counts.

```bash
process:
  spacy_model: "it_core_news_lg"
  tokenizer_name: "intfloat/multilingual-e5-large"
  chunksize: 100
  max_tokens: 250
  min_tokens: 80
  output_column: "sentence1"
  auto_download_spacy_model: true
  mode: "sentences" # Options: "chunks", "sentences"

  # Please set it to chunks in case you want to process data in chunks
```

Note: auto_download_spacy_model: true can trigger python -m spacy download it_core_news_lg if not present — ensure you have the necessary permissions.

### Continue / redo steps

If the pipeline stops, you can continue from the last successful step (the code will pick up existing outputs). To force re-running steps and overwrite outputs:

```bash
general:
  force: false  # set to true to redo steps even if outputs already exist
```

## Download Preprocessed Data

Preprocessed data (if you wish to use it directly):
https://drive.google.com/file/d/1qw9k9Rm9w20X0KEpc53MUp9oVGDJ8Vfx/view?usp=drive_link
