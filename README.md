# 📦 DA6401 Assignment 3 _ A B Keshav Kumar (AE24S021)


# 🔗 [GitHub Repository] https://github.com/Keshav-iitm/RNN_assign3.git
# 📊 [WandB Project Link] https://wandb.ai/ae24s021-indian-institute-of-technology-madras/RNN-Transliteration/reports/Assignment-3_RNN--VmlldzoxMjcwOTgyNg?accessToken=0a8alnduag53zg9q48od9g6fgcca6hn657g5vpazacrif27yxmxxgr1tpqrwczs6


This repository contains implementations of sequence-to-sequence models using Recurrent Neural Networks (RNNs) for English-native language transliteration. The project is part of an assignment focusing on vanilla and attention-based RNNs.

---

## 📁 Folder Structure & Dataset Placement

```
RNN_assign3-main/
├── RNN_vannila/
│   ├── train_RNN.py              # Training script for vanilla RNN
│   ├── test_RNN.py               # Testing script for vanilla RNN
│   ├── model_RNN.py              # Defines the vanilla RNN architecture
│   ├── dataset_RNN.py            # Loads and prepares dataset for vanilla RNN
│   ├── best_model_*.pt           # Saved best-performing vanilla RNN model
│   └── dhakshina_dataset_v1.0/   # Dataset folder for vanilla RNN
│
├── RNN_attention/
│   ├── train_att_RNN.py          # Training script for attention-based RNN
│   ├── test_att_RNN.py           # Testing script for attention-based RNN
│   ├── model_att_RNN.py          # Defines the attention-based RNN architecture
│   ├── dataset_att_RNN.py        # Loads and prepares dataset for attention RNN
│   ├── connect_att_RNN.py        # Visualizes attention weights (generates gifs)
│   ├── best_att_model_*.pt       # Saved best-performing attention RNN model
│   └── dhakshina_dataset_v1.0/   # Dataset folder for attention RNN
│
├── connectivity_gifs/           # Visualizations of attention weights over training for 3 different samples
│   ├── sample 1 (gru.gif ; lstm.gif)
│   ├── sample 2 (gru.gif ; lstm.gif)
│   └── sample 3 (gru.gif ; lstm.gif)
│
├── predictions_attention/
│   ├── test_predictions.csv              # Basic prediction output
│   ├── test_predictions_full.csv         # Full prediction logs with scores
│   └── test_predictions_submission.csv   # Native-english format
│
├── predictions_vanilla/
│   └── test_predictions.csv              # Predictions from vanilla RNN
│   ├── test_predictions_with_accuracy.csv
├── .gitignore
├── LICENSE
├── README.md
```

### 📌 IMPORTANT:

- The `dhakshina_dataset_v1.0/` and `best_model_file` dataset is placed **inside `RNN_vannila/`** and **inside `RNN_attention/`** (**Recommended**)
- If placed **anywhere else (e.g., in parent folder)**, just use `--data_dir`. (**⚠️ this might not work in  every script**)
- Donot run the scripts from the parent folder. Please use the respective folders.
---

## ⚙️ Environment Setup (Tested with CUDA GPU)

Use the following commands to create a reproducible environment:

```bash
conda create -n torch_gpu python=3.9
conda activate torch_gpu

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn==1.0.2
pip install wandb==0.12.21
pip install numpy==1.21.6
pip install tqdm==4.62.3
pip install thop==0.0.31.post2005241907
pip install matplotlib==3.5.3
pip install numpy==1.21.6
pip install pandas==1.3.5
pip install seaborn==0.11.2
pip install matplotlib==3.5.3
pip install imageio==2.9.0

```

### ➕ Additional Imports

- Python standard: `os`, `argparse`, `sys`, `traceback`, `types`, `getpass`
- All included by default in Python ≥ 3.6

---

## 🚀 How to Run Scripts from Terminal

### ✅ Always run from the respective folders `RNN_assign3-main/RNN_vannila` or `RNN_assign3-main/RNN_attention`  using this format:

```bash
python <script_name.py> [--arguments]
python <script_name.py> [--arguments]
```

---
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
## 🧠 Script Descriptions

### ✅ RNN_vannila/ Scripts  - Vannila RNN
Run scripts inside :
**Folder**: `RNN_vannila`  
**Description**: Vanilla RNN implementation for sequence modeling with training, testing, and dataset handling scripts.


### 1. `model_RNN.py` — RNN model with flexibility

**Run**:
```bash
python model_RNN.py
```

**Argparse options**:
- `----src_vocab_size`
- `--tgt_vocab_size`
- `--embed_size`
- `--hidden_size`
- `--num_encoder_layers`
- `--num_decoder_layers`
- `--dropout`
- `--cell_type`
- `--init_method`

#model and computations for Vannila RNN.

---

### 2. `dataset_RNN.py` — Data Loaders with Stratified Split

**Run**:
```bash
python dataset_RNN.py 
```
**Argparse options**:
- `--data_dir` #should be specified if is not in RNN_vannila folder
- `--lang` 

#Loading the dataset and verifying samples in each set.
---

### 3. `train_RNN.py` — W&B Sweep Training

**⚠️ NOTE**: You **must specify --sweep while running the script** via:
**Run**:
```bash
python train_RNN.py --sweep
```

**Argparse options**:
- `--data_dir` #should be specified if is not in RNN_vannila folder
- `--lang`
- `--sweep`

#best_model_<name>.pt will automatically be saved.
---

### 4. `test_RNN.py` — Evaluation & W&B Prediction Grid
**⚠️ NOTE**: You **must specify --model_path while running the script** if not using the default model which is `RNN_vannila/best_model_*.pt` annd place it in same folder `RNN_vannila/` via:
**Run**:
```bash
python test_RNN.py 
```

**Argparse options**:
- `--model_path` #should be specified if not using the default model
- `--data_dir` #should be specified if is not in RNN_vannila folder
- `--lang`
- `--embed_size`
- `--hidden_size`
- `--num_encoder_layers`
- `--num_decoder_layers`
- `--cell_type`
- `--init_method`
- `--dropout`
- `--batch_size`
#based on users best hyperparamter combinations. Default is set with respect to the obtained best model.
#produces confusion matrix token level and prediction tables in WandB.
#displayes exact and token level accuracy.
---
----------------------------------------------------------------------------------------------------------------------------------------------------------------------
### ✅ RNN_attention/ Scripts  - RNN with attention 
Run scrips inside
**Folder**: ``RNN_attention` `  
**Description**: RNN with attention mechanism for sequence-to-sequence modeling, including training, testing, and dataset processing scripts and connectivity animation.

### 1. `model_att_RNN.py` — RNN model with attention

**Run**:
```bash
python model_att_RNN.py
```

**Argparse options**:
- `----src_vocab_size`
- `--tgt_vocab_size`
- `--embed_size`
- `--hidden_size`
- `--num_encoder_layers`
- `--num_decoder_layers`
- `--dropout`
- `--cell_type`
- `--init_method`
#Model loaded with attention RNN.
---

### 2. `dataset_att_RNN.py` — Data Loaders with Stratified Split

**Run**:
```bash
python dataset_att_RNN.py 
```
**Argparse options**:
- `--data_dir` #should be specified if is not in RNN_attention folder
- `--lang` 

#Loading the dataset and verifying samples in each set.
---

### 3. `train_att_RNN.py` — W&B Sweep Training

**⚠️ NOTE**: You **must specify --sweep while running the script** via:
**Run**:
```bash
python train_att_RNN.py --sweep
```

**Argparse options**:
- `--data_dir` #should be specified if is not in RNN_attention folder
- `--lang`
- `--sweep`

#generates hyperparameter combination sweeps for attention RNN in WandB.
---

### 4. `test_att_RNN.py` — Evaluation & W&B Prediction Grid
**⚠️ NOTE**: You **must specify --model_path while running the script** if not using the default model which is `RNN_vannila/best_model_*.pt` and place it `RNN_attention/` via:
**Run**:
```bash
python test_att_RNN.py 
```

**Argparse options**:
- `--model_path` #should be specified if not using the default model
- `--data_dir` #should be specified if is not in RNN_vannila folder
- `--lang`
- `--embed_size`
- `--hidden_size`
- `--num_encoder_layers`
- `--num_decoder_layers`
- `--cell_type`
- `--init_method`
- `--dropout`
- `--batch_size`
#based on users best hyperparamter combinations. Default is set with respect to the obtained best model.
#produces confusion matrix token level, heatmaps and prediction tables in WandB.
#displayes exact and token level accuracy.
---

### 4. `connect_att_RNN.py` — Gives animation of connectivity for different samples
**⚠️ NOTE**: You **must specify --model_path while running the script** if not using the default model which is `RNN_vannila/best_model_*.pt` and place it `RNN_attention/` via:
**Run**:
```bash
python connect_att_RNN.py 
```

**Argparse options**:
- `--model_path` #should be specified if not using the default model
- `--embed_size`
- `--hidden_size`
- `--num_encoder_layers`
- `--num_decoder_layers`
- `--cell_type`
- `--dropout`
#Produces animation for connectivity and displays the weights at each generating steps.
#Provides animations for 3 different samples. Can be argparsed for different cell_types too.
-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
## 🖼️ WandB Instructions: 

### ⚙️ W\&B Setup Instructions
To use W\&B for logging and hyperparameter sweeps:
1. **Install W\&B**:
```bash
pip install wandb
```
2. **Login to W\&B**:

```bash
wandb login
```
This will prompt you to enter your API key. Get it from: [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. **Run a Sweep**:
```bash
wandb sweep sweep.yaml  # Generates a sweep ID
python <script_name>.py --sweep <sweep_id>
```
### 📌 Note
Please log in to Weights & Biases (`wandb login`) and initialize with your API key to run the program with logging or sweeps.
---

## ✍️ Author

> **DA6401 - Deep Learning**  
>  *A B Keshav Kumar (AE24S021),MS Scholar, IIT Madras* 
> *Assignment 3_RNN
---

## 💬 Need Help?

If any script fails due to import/module issues, check:
- Python version (3.9 recommended)
- CUDA 11.3 required for GPU support
- Dataset path structure (Dataset should be inside respective folders like "RNN_vannila" and "RNN_attention")
- Run scripts only from the repective folders like "RNN_vannila" and "RNN_attention"
- W&B login status (`wandb login`)
