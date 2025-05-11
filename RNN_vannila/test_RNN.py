import torch
import argparse
import wandb
import os
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from dataset_RNN import DakshinaDataset
from model_RNN import Seq2Seq
from train_RNN import get_collate_fn

import argparse
import torch
from torch.utils.data import DataLoader
import wandb

# Argument Parser
parser = argparse.ArgumentParser(description="Evaluate trained Seq2Seq model on test set")
parser.add_argument('--model_path', type=str, default='best_model_9wrhl7f9.pt', help='Path to the best model file')
parser.add_argument('--data_dir', type=str, default='./dakshina_dataset_v1.0', help='Path to the dataset directory')
parser.add_argument('--lang', type=str, default='ta', help='Language code')
parser.add_argument('--embed_size', type=int, default=128, help='Size of the embedding vectors')
parser.add_argument('--hidden_size', type=int, default=256, help='Size of the hidden layers')
parser.add_argument('--num_encoder_layers', type=int, default=2, help='Number of layers in the encoder')
parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of layers in the decoder')
parser.add_argument('--cell_type', type=str, choices=['lstm', 'gru'], default='lstm', help='RNN cell type to use')
parser.add_argument('--init_method', type=str, choices=['xavier', 'kaiming', 'normal'], default='xavier', help='Weight initialization method')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout probability')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
args = parser.parse_args()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize wandb
wandb.init(project="RNN-Transliteration", name="final_test_eval")

# Load data
dataset = DakshinaDataset(args.data_dir, args.lang)
src_vocab = dataset.src_vocab
tgt_vocab = dataset.tgt_vocab
inv_src_vocab = {v: k for k, v in src_vocab.items()}
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

test_data = dataset.test_data
collate_fn = get_collate_fn(src_vocab, tgt_vocab)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# Load model
model = Seq2Seq(
    src_vocab_size=len(src_vocab),
    tgt_vocab_size=len(tgt_vocab),
    embed_size=args.embed_size,
    hidden_size=args.hidden_size,
    num_encoder_layers=args.num_encoder_layers,
    num_decoder_layers=args.num_decoder_layers,
    dropout=args.dropout,
    cell_type=args.cell_type,
    init_method=args.init_method
).to(device)

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()



# Inferences
y_true_all, y_pred_all, pred_table = [], [], []
token_correct, token_total = 0, 0
predictions_folder = "predictions_vanilla"
os.makedirs(predictions_folder, exist_ok=True)
csv_path = os.path.join(predictions_folder, "test_predictions.csv")

with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Input", "Prediction", "Target", "Exact Match", "Token-Level Accuracy"])

    with torch.no_grad():
        for src, tgt in test_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            outputs = model(src, tgt_input)
            preds = outputs.argmax(dim=-1)

            for i in range(src.size(0)):
                src_seq = "".join([inv_src_vocab[ix.item()] for ix in src[i] if ix.item() != tgt_vocab['<pad>']])
                tgt_seq = [inv_tgt_vocab[ix.item()] for ix in tgt_output[i] if ix.item() not in [tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>']]]
                pred_seq = [inv_tgt_vocab[ix.item()] for ix in preds[i] if ix.item() not in [tgt_vocab['<pad>'], tgt_vocab['<sos>'], tgt_vocab['<eos>']]]

                tgt_str = "".join(tgt_seq)
                pred_str = "".join(pred_seq)
                is_exact = (tgt_str == pred_str)

                # Token-level accuracy for this sample
                min_len = min(len(tgt_seq), len(pred_seq))
                token_correct_sample = sum([tgt_seq[j] == pred_seq[j] for j in range(min_len)])
                token_total_sample = max(len(tgt_seq), len(pred_seq))
                token_acc_sample = token_correct_sample / token_total_sample if token_total_sample > 0 else 0

                # For global stats
                token_correct += token_correct_sample
                token_total += token_total_sample

                y_true_all.append(tgt_str)
                y_pred_all.append(pred_str)
                pred_table.append([
                    src_seq,
                    pred_str,
                    tgt_str,
                    "✔" if is_exact else "✘",
                    f"{token_acc_sample:.2%}"
                ])
                writer.writerow([src_seq, pred_str, tgt_str, "✔" if is_exact else "✘", f"{token_acc_sample:.2%}"])
submission_folder = "predictions_vanilla"
os.makedirs(submission_folder, exist_ok=True)
submission_csv_path = os.path.join(submission_folder, "test_predictions_with_accuracy.csv")

with open(submission_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Input", "Prediction", "Token-Level Accuracy"])
    for row in pred_table:
        writer.writerow([row[0], row[1], row[4]])
print(f"Submission file saved to: {submission_csv_path}")

# Prediction tables
wandb.log({
    "all_predictions_with_token_acc": wandb.Table(
        columns=["Input", "Prediction", "Target", "Exact Match", "Token-Level Accuracy"],
        data=pred_table[:50]
    ),
    "exact_match_predictions": wandb.Table(
        columns=["Input", "Prediction", "Target", "Exact Match", "Token-Level Accuracy"],
        data=[row for row in pred_table if row[3] == "✔"][:50]
    )
})

# Accuracies
exact_matches = sum([1 for t, p in zip(y_true_all, y_pred_all) if t == p])
exact_acc = 100 * exact_matches / len(y_true_all)
token_acc = 100 * token_correct / token_total if token_total > 0 else 0
print(f"Exact Match Test Accuracy: {exact_acc:.2f}%")
print(f"Token-level Test Accuracy: {token_acc:.2f}%")
wandb.log({"Exact Match Accuracy": exact_acc, "Token-level Accuracy": token_acc})

# Confusion matrix : Charecter Level
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
fig_token, ax_token = plt.subplots(figsize=(12, 10))
sns.heatmap(cm_token, xticklabels=all_chars, yticklabels=all_chars,
            annot=False, fmt="d", cmap="coolwarm", ax=ax_token)
ax_token.set_title("Character-Level Confusion Matrix (Token-Level)")
ax_token.set_xlabel("Predicted")
ax_token.set_ylabel("Actual")
plt.tight_layout()
wandb.log({"confusion_matrix_token_level": wandb.Image(fig_token)})
plt.close(fig_token)

# Confusion matrix : Sequence level
# This matrix shows how often each true sequence is predicted as each predicted sequence.
# For visualization, I'll use the top 20 most common true/pred sequences for clarity.
from collections import Counter

pair_counts = Counter(zip(y_true_all, y_pred_all))
most_common_true = [x[0] for x in Counter(y_true_all).most_common(20)]
most_common_pred = [x[0] for x in Counter(y_pred_all).most_common(20)]

seq_cm = np.zeros((len(most_common_true), len(most_common_pred)), dtype=int)
for (t, p), count in pair_counts.items():
    if t in most_common_true and p in most_common_pred:
        i = most_common_true.index(t)
        j = most_common_pred.index(p)
        seq_cm[i, j] = count

fig_seq, ax_seq = plt.subplots(figsize=(14, 12))
sns.heatmap(seq_cm, xticklabels=most_common_pred, yticklabels=most_common_true,
            annot=True, fmt="d", cmap="Blues", ax=ax_seq)
ax_seq.set_title("Sequence-Level Confusion Matrix (Top 20)")
ax_seq.set_xlabel("Predicted Sequence")
ax_seq.set_ylabel("True Sequence")
plt.tight_layout()
wandb.log({"confusion_matrix_sequence_level": wandb.Image(fig_seq)})
plt.close(fig_seq)

print("Evaluation and logging complete.")
