import torch
import argparse
import wandb
import os
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from dataset_RNN import DakshinaDataset
from model_RNN import Seq2Seq
from train_RNN import get_collate_fn

# FONT CONFIGURATION
try:
    # Tamil font configuration
    tamil_font_path = '/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf'
    tamil_font = mpl.font_manager.FontProperties(fname=tamil_font_path)
    
    # English font configuration
    english_font = {'fontname': 'DejaVu Sans', 'fontsize': 12}
    
    print(f"✅ Using Noto Sans Tamil from: {tamil_font_path}")
except Exception as e:
    print(f"⚠️ Font error: {str(e)}")
    tamil_font = mpl.font_manager.FontProperties()
    english_font = {'fontname': 'sans-serif', 'fontsize': 12}

# Clear matplotlib cache
mpl.rcParams.update(mpl.rcParamsDefault)


# Argument Parser
parser = argparse.ArgumentParser(description="Evaluate trained Seq2Seq model on test set")
parser.add_argument('--model_path', type=str, default='./best_model_y9bfi7o3.pt', help='Path to trained model checkpoint')
parser.add_argument('--data_dir', type=str, default='./dakshina_dataset_v1.0', help='Path to dataset directory')
parser.add_argument('--lang', type=str, default='ta', help='Language code (ta/hi/bn)')
parser.add_argument('--embed_size', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_encoder_layers', type=int, default=2)
parser.add_argument('--num_decoder_layers', type=int, default=2)
parser.add_argument('--cell_type', type=str, default='gru', choices=['lstm', 'gru'])
parser.add_argument('--init_method', type=str, default='xavier', choices=['xavier', 'he', 'default'])
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--wandb_project', type=str, default='Eng_tamil-Transliteration')
args = parser.parse_args()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize WandB
wandb.init(project=args.wandb_project, config=vars(args))

# Dataset and Model setup
dataset = DakshinaDataset(args.data_dir, args.lang)
src_vocab = dataset.src_vocab
tgt_vocab = dataset.tgt_vocab

# Inverse mappings
inv_src_vocab = {v: k for k, v in src_vocab.items()}
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

# DataLoader
test_loader = DataLoader(
    dataset.test_data,
    batch_size=args.batch_size,
    collate_fn=get_collate_fn(src_vocab, tgt_vocab),
    shuffle=False,
    pin_memory=True
)

# Model initialization
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

# Evaluation loop
predictions = []
with torch.no_grad(), open('test_predictions.csv', 'w', encoding='utf-8-sig') as pred_file:
    writer = csv.writer(pred_file)
    writer.writerow(['Input (Latin)', 'Prediction (Native)', 'Target (Native)', 'Exact Match', 'Token Accuracy'])
    
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        outputs = model(src, tgt[:, :-1])
        preds = outputs.argmax(dim=-1)
        
        for i in range(src.size(0)):
            # Input processing
            input_str = ''.join([inv_src_vocab[idx.item()] for idx in src[i] if idx.item() != 0])
            
            # Target processing
            target_str = ''.join([inv_tgt_vocab[idx.item()] for idx in tgt[i, 1:] if idx.item() not in [0,1,2]])
            
            # Prediction processing
            pred_str = ''.join([inv_tgt_vocab[idx.item()] for idx in preds[i] if idx.item() not in [0,1,2]])
            
            # Metrics calculation
            exact_match = (pred_str == target_str)
            min_len = min(len(target_str), len(pred_str))
            token_acc = sum(t == p for t, p in zip(target_str[:min_len], pred_str[:min_len])) / max(len(target_str), len(pred_str))
            
            # Write to CSV
            writer.writerow([
                input_str,
                pred_str,
                target_str,
                exact_match,
                f"{token_acc*100:.1f}%"
            ])
            predictions.append({
                'input': input_str,
                'prediction': pred_str,
                'target': target_str,
                'exact_match': exact_match,
                'token_accuracy': token_acc
            })

# Calculate metrics
exact_match_acc = sum(p['exact_match'] for p in predictions) / len(predictions)
token_level_acc = sum(p['token_accuracy'] for p in predictions) / len(predictions)

print(f"\n{'Exact Match Accuracy:':<25} {exact_match_acc:.2%}")
print(f"{'Token-level Accuracy:':<25} {token_level_acc:.2%}")

#  CONFUSION MATRIX WITH PERCENTAGES
true_labels = []
pred_labels = []
for p in predictions:
    t_chars = list(p['target'])
    p_chars = list(p['prediction'])
    min_len = min(len(t_chars), len(p_chars))
    if min_len == 0: continue
    true_labels.extend(t_chars[:min_len])
    pred_labels.extend(p_chars[:min_len])

all_chars = sorted(set(true_labels + pred_labels))
char_to_idx = {c: i for i, c in enumerate(all_chars)}

cm = confusion_matrix([char_to_idx[c] for c in true_labels], 
                      [char_to_idx[c] for c in pred_labels],
                      labels=range(len(all_chars)))

# Calculate percentages
cm_perc = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
np.fill_diagonal(cm_perc, 0)  # Clear diagonal for secondary coloring

plt.figure(figsize=(20, 18))
ax = sns.heatmap(
    cm_perc, 
    annot=cm,
    fmt='d',
    cmap='OrRd',
    square=True,
    linewidths=0.5,
    linecolor='white',
    xticklabels=all_chars,
    yticklabels=all_chars,
    annot_kws={'fontsize': 8, 'color': 'black'}
)

# Overlay diagonal with correct predictions
cm_diag = np.zeros_like(cm)
np.fill_diagonal(cm_diag, np.diag(cm))
sns.heatmap(
    cm_diag, 
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    square=True,
    linewidths=0.5,
    linecolor='white',
    xticklabels=all_chars,
    yticklabels=all_chars,
    mask=(cm_diag == 0),
    ax=ax,
    annot_kws={'fontsize': 8, 'color': 'white'}
)

# Font handling
ax.set_xticklabels(ax.get_xticklabels(), fontproperties=tamil_font, rotation=90, fontsize=10)
ax.set_yticklabels(ax.get_yticklabels(), fontproperties=tamil_font, fontsize=10)
plt.title('Character-level Confusion Matrix\nBlue: Correct, Red: Incorrect', **english_font, pad=20)
plt.xlabel('Predicted Characters', **english_font, labelpad=15)
plt.ylabel('Actual Characters', **english_font, labelpad=15)
plt.tight_layout()
wandb.log({'confusion_matrix': wandb.Image(plt)})
plt.close()
# 3x3 Alignment Visualization
samples = predictions[:9]
fig, axs = plt.subplots(3, 3, figsize=(20, 20))

for idx, sample in enumerate(samples):
    ax = axs[idx//3, idx%3]
    input_chars = list(sample['input'])
    pred_chars = list(sample['prediction'])
    
    # Create alignment matrix
    max_len = max(len(input_chars), len(pred_chars))
    matrix = np.zeros((max_len, max_len))
    for i in range(min(len(input_chars), len(pred_chars))):
        matrix[i,i] = 1  # Diagonal pattern
    
    sns.heatmap(
        matrix,
        ax=ax,
        cmap='Greys',
        cbar=False,
        square=True,
        xticklabels=input_chars + ['']*(max_len-len(input_chars)),
        yticklabels=pred_chars + ['']*(max_len-len(pred_chars))
    )
    
    # Font handling
    ax.set_xticklabels(ax.get_xticklabels(), **english_font)
    ax.set_yticklabels(ax.get_yticklabels(), fontproperties=tamil_font)
    ax.set_title(f"Sample {idx+1}", **english_font)
    ax.set_xlabel("Input (Latin)", **english_font)
    ax.set_ylabel("Prediction (Tamil)", **english_font)

plt.tight_layout()
wandb.log({'alignment_visualization': wandb.Image(fig)})
plt.close()

# Final logging
wandb.log({
    'exact_match_accuracy': exact_match_acc,
    'token_level_accuracy': token_level_acc,
    'predictions': wandb.Table(dataframe=pd.DataFrame(predictions))
})

print("Evaluation complete. Results saved to test_predictions.csv")
