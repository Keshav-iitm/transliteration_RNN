import torch
import argparse
import wandb
import os
import csv
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import warnings
from collections import Counter

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from dataset_att_RNN import DakshinaDataset, get_collate_fn
from model_att_RNN import Seq2Seq

# --------- Suppress Font Warnings ---------
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
try:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Font with better Unicode support
except:
    print("⚠️ Font not found - using default")

# --------- Argument Parsing ---------
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='best_att_model_qurwg2mv.pt',
                    help='Path to the best attention model file')
args = parser.parse_args()

# --------- W&B Init ---------
wandb.init(project="RNN-Transliteration-Attention", name="final_test_eval_attention")

# --------- Config (Match your best sweep parameters) ---------
config = {
    'data_dir': './dakshina_dataset_v1.0',
    'lang': 'ta',
    'embed_size': 256,
    'hidden_size': 256,
    'num_encoder_layers': 2,
    'num_decoder_layers': 2,
    'cell_type': 'gru',
    'init_method': 'xavier',
    'dropout': 0.4,
    'beam_width': 3,
    'batch_size': 16,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
device = torch.device(config['device'])

# --------- Load Data and Vocabs ---------
dataset = DakshinaDataset(config['data_dir'], config['lang'])
src_vocab = dataset.src_vocab
tgt_vocab = dataset.tgt_vocab
inv_src_vocab = {v: k for k, v in src_vocab.items()}
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

test_data = dataset.test_data
collate_fn = get_collate_fn(dataset)
test_loader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

# --------- Load Attention Model ---------
model = Seq2Seq(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    embed_size=config['embed_size'],
    hidden_size=config['hidden_size'],
    num_encoder_layers=config['num_encoder_layers'],
    num_decoder_layers=config['num_decoder_layers'],
    dropout=config['dropout'],
    cell_type=config['cell_type'],
    init_method=config['init_method']
).to(device)
model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()

# --------- Inference & Logging ---------
y_true_all, y_pred_all, pred_table = [], [], []
token_correct, token_total = 0, 0
attention_data = []
predictions_folder = "predictions_attention"
os.makedirs(predictions_folder, exist_ok=True)

with torch.no_grad():
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        # Get outputs and attention weights
        outputs, attn_weights = model(src, tgt_input)
        preds = outputs.argmax(dim=-1)

        for i in range(src.size(0)):
            # Process source and target sequences
            src_seq = "".join([inv_src_vocab[ix.item()] for ix in src[i] if ix.item() != 0])
            tgt_seq = [inv_tgt_vocab[ix.item()] for ix in tgt_output[i] if ix.item() not in [0, 1, 2]]
            pred_seq = [inv_tgt_vocab[ix.item()] for ix in preds[i] if ix.item() not in [0, 1, 2]]
            
            # Get attention weights for this sample (only store for first 10)
            if len(attention_data) < 10:
                src_chars = [inv_src_vocab[ix.item()] for ix in src[i] if ix.item() != 0]
                tgt_chars = pred_seq
                sample_attn = attn_weights[i, :len(tgt_chars), :len(src_chars)].cpu().numpy()
                attention_data.append((src_chars, tgt_chars, sample_attn))

            # Calculate accuracies
            tgt_str = "".join(tgt_seq)
            pred_str = "".join(pred_seq)
            is_exact = (tgt_str == pred_str)

            # Token-level accuracy
            min_len = min(len(tgt_seq), len(pred_seq))
            token_correct_sample = sum([tgt_seq[j] == pred_seq[j] for j in range(min_len)])
            token_total_sample = max(len(tgt_seq), len(pred_seq))
            token_acc_sample = token_correct_sample / token_total_sample if token_total_sample > 0 else 0

            # Update global stats
            token_correct += token_correct_sample
            token_total += token_total_sample

            # Store results
            y_true_all.append(tgt_str)
            y_pred_all.append(pred_str)
            pred_table.append([
                src_seq,
                pred_str,
                tgt_str,
                "✔" if is_exact else "✘",
                f"{token_acc_sample:.2%}"
            ])

# --------- Save Prediction CSVs ---------
# Full predictions with all details
csv_path = os.path.join(predictions_folder, "test_predictions_full.csv")
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Input", "Prediction", "Target", "Exact Match", "Token-Level Accuracy"])
    for row in pred_table:
        writer.writerow(row)

# Simple submission format
submission_csv_path = os.path.join(predictions_folder, "test_predictions_submission.csv")
with open(submission_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Input", "Prediction"])
    for row in pred_table:
        writer.writerow([row[0], row[1]])

print(f"Prediction files saved to {predictions_folder}/")

# --------- Attention Heatmaps Visualization ---------
def plot_attention_grid(attention_data):
    """Plot attention heatmaps in a grid layout (4x3 for up to 10 samples)"""
    fig, axs = plt.subplots(4, 3, figsize=(20, 16))
    axs = axs.flatten()
    
    for idx, (src_chars, tgt_chars, attn_weights) in enumerate(attention_data):
        if idx >= 10:  # Only plot up to 10 samples
            break
            
        ax = axs[idx]
        cax = ax.matshow(attn_weights, cmap='viridis')
        
        # Set axis labels
        ax.set_xticks(range(len(src_chars)))
        ax.set_yticks(range(len(tgt_chars)))
        ax.set_xticklabels(src_chars, rotation=45, fontsize=9)
        ax.set_yticklabels(tgt_chars, fontsize=9)
        
        ax.xaxis.set_ticks_position("bottom")
        ax.set_title(f"Sample {idx+1}")
        fig.colorbar(cax, ax=ax)
    
    # Hide unused subplots
    for i in range(len(attention_data), len(axs)):
        axs[i].set_visible(False)
    
    plt.tight_layout()
    return fig

# --------- Calculate Metrics ---------
exact_matches = sum([1 for t, p in zip(y_true_all, y_pred_all) if t == p])
exact_acc = 100 * exact_matches / len(y_true_all)
token_acc = 100 * token_correct / token_total if token_total > 0 else 0

print(f"Exact Match Test Accuracy: {exact_acc:.2f}%")
print(f"Token-level Test Accuracy: {token_acc:.2f}%")

# --------- Character-Level Confusion Matrix ---------
def plot_char_confusion_matrix():
    all_chars = sorted(set("".join(y_true_all + y_pred_all)))
    char_to_idx = {ch: i for i, ch in enumerate(all_chars)}
    true_chars, pred_chars = [], []

    for t_seq, p_seq in zip(y_true_all, y_pred_all):
        for t, p in zip(t_seq, p_seq):
            true_chars.append(char_to_idx.get(t, -1))
            pred_chars.append(char_to_idx.get(p, -1))

    true_chars = np.array([i for i in true_chars if i >= 0])
    pred_chars = np.array([i for i in pred_chars if i >= 0])

    cm_token = confusion_matrix(true_chars, pred_chars, labels=range(len(all_chars)))
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_token, xticklabels=all_chars, yticklabels=all_chars,
                annot=False, fmt="d", cmap="coolwarm", ax=ax)
    ax.set_title("Character-Level Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    return fig

# --------- Sequence-Level Confusion Matrix ---------
def plot_seq_confusion_matrix():
    # Get top 20 most common sequences
    pair_counts = Counter(zip(y_true_all, y_pred_all))
    most_common_true = [x[0] for x in Counter(y_true_all).most_common(20)]
    most_common_pred = [x[0] for x in Counter(y_pred_all).most_common(20)]

    seq_cm = np.zeros((len(most_common_true), len(most_common_pred)), dtype=int)
    for (t, p), count in pair_counts.items():
        if t in most_common_true and p in most_common_pred:
            i = most_common_true.index(t)
            j = most_common_pred.index(p)
            seq_cm[i, j] = count

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(seq_cm, xticklabels=most_common_pred, yticklabels=most_common_true,
                annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Sequence-Level Confusion Matrix (Top 20)")
    ax.set_xlabel("Predicted Sequence")
    ax.set_ylabel("True Sequence")
    plt.tight_layout()
    return fig

# --------- Log Everything to W&B ---------
# Log metrics
wandb.log({
    "Exact Match Accuracy": exact_acc,
    "Token-level Accuracy": token_acc
})

# Log prediction tables
wandb.log({
    "all_predictions": wandb.Table(
        columns=["Input", "Prediction", "Target", "Exact Match", "Token-Level Accuracy"],
        data=pred_table[:50]
    ),
    "exact_match_predictions": wandb.Table(
        columns=["Input", "Prediction", "Target", "Exact Match", "Token-Level Accuracy"],
        data=[row for row in pred_table if row[3] == "✔"][:50]
    )
})

# Log attention heatmaps
if attention_data:
    attn_fig = plot_attention_grid(attention_data)
    wandb.log({"attention_heatmaps": wandb.Image(attn_fig)})
    plt.close(attn_fig)

# Log confusion matrices
char_cm_fig = plot_char_confusion_matrix()
wandb.log({"confusion_matrix_character": wandb.Image(char_cm_fig)})
plt.close(char_cm_fig)

seq_cm_fig = plot_seq_confusion_matrix()
wandb.log({"confusion_matrix_sequence": wandb.Image(seq_cm_fig)})
plt.close(seq_cm_fig)

print("Evaluation and logging complete.")
