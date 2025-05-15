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
from dataset_att_RNN import DakshinaDataset, get_collate_fn
from model_att_RNN import Seq2Seq

# ================== FONT CONFIGURATION ==================
try:
    tamil_font_path = '/usr/share/fonts/truetype/noto/NotoSansTamil-Regular.ttf'
    tamil_font = mpl.font_manager.FontProperties(fname=tamil_font_path)
    english_font = {'fontname': 'DejaVu Sans', 'fontsize': 12}
    print(f"✅ Using Noto Sans Tamil from: {tamil_font_path}")
except Exception as e:
    print(f"⚠️ Font error: {str(e)}")
    tamil_font = mpl.font_manager.FontProperties()
    english_font = {'fontname': 'sans-serif', 'fontsize': 12}

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.family'] = 'DejaVu Sans'
# ========================================================

# Argument Parser
parser = argparse.ArgumentParser(description="Evaluate attention-based Seq2Seq model")
parser.add_argument('--model_path', type=str, default='./best_att_model_wn3dcrxt.pt')
parser.add_argument('--data_dir', type=str, default='./dakshina_dataset_v1.0')
parser.add_argument('--lang', type=str, default='ta')
parser.add_argument('--embed_size', type=int, default=128)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_encoder_layers', type=int, default=1)
parser.add_argument('--num_decoder_layers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.4)
parser.add_argument('--cell_type', type=str, default='gru')
parser.add_argument('--init_method', type=str, default='xavier')
parser.add_argument('--beam_width', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize WandB
wandb.init(project="Eng_Tamil-Transliteration", config=vars(args))

# Dataset and Model setup
dataset = DakshinaDataset(args.data_dir, args.lang)
src_vocab = dataset.src_vocab
tgt_vocab = dataset.tgt_vocab
inv_src_vocab = {v: k for k, v in src_vocab.items()}
inv_tgt_vocab = {v: k for k, v in tgt_vocab.items()}

# DataLoader
test_loader = DataLoader(
    dataset.test_data,
    batch_size=args.batch_size,
    collate_fn=get_collate_fn(dataset),
    shuffle=False
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

# Output directory
predictions_dir = "predictions_attention"
os.makedirs(predictions_dir, exist_ok=True)

# Evaluation variables
predictions = []
attention_samples = []
token_correct = 0
token_total = 0

# Evaluation loop
with torch.no_grad(), open(os.path.join(predictions_dir, 'test_predictions.csv'), 'w', encoding='utf-8-sig') as pred_file:
    writer = csv.writer(pred_file)
    writer.writerow(['Input (Latin)', 'Prediction (Native)', 'Target (Native)', 'Exact Match', 'Token Accuracy'])
    
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        outputs, attn_weights = model(src, tgt[:, :-1])
        preds = outputs.argmax(dim=-1)
        
        for i in range(src.size(0)):
            # Process sequences
            src_str = ''.join([inv_src_vocab[idx.item()] for idx in src[i] if idx.item() != 0])
            tgt_str = ''.join([inv_tgt_vocab[idx.item()] for idx in tgt[i, 1:] if idx.item() not in [0,1,2]])
            pred_str = ''.join([inv_tgt_vocab[idx.item()] for idx in preds[i] if idx.item() not in [0,1,2]])
            
            # Store attention data
            if len(attention_samples) < 12:
                src_chars = [inv_src_vocab[idx.item()] for idx in src[i] if idx.item() != 0]
                pred_chars = [inv_tgt_vocab[idx.item()] for idx in preds[i] if idx.item() not in [0,1,2]]
                sample_attn = attn_weights[i, :len(pred_chars), :len(src_chars)].cpu().numpy()
                attention_samples.append((src_chars, pred_chars, sample_attn))
            
            # Calculate metrics
            exact_match = (pred_str == tgt_str)
            min_len = min(len(tgt_str), len(pred_str))
            token_acc = sum(t == p for t, p in zip(tgt_str[:min_len], pred_str[:min_len])) / max(len(tgt_str), len(pred_str))
            
            # Update stats
            token_correct += sum(t == p for t, p in zip(tgt_str, pred_str))
            token_total += max(len(tgt_str), len(pred_str))
            
            # Write to CSV
            writer.writerow([
                src_str,
                pred_str,
                tgt_str,
                exact_match,
                f"{token_acc*100:.1f}%"
            ])
            
            predictions.append({
                'input': src_str,
                'prediction': pred_str,
                'target': tgt_str,
                'exact_match': exact_match,
                'token_accuracy': token_acc
            })

# Calculate metrics
exact_match_acc = sum(p['exact_match'] for p in predictions) / len(predictions)
token_level_acc = token_correct / token_total

# ================== CONFUSION MATRIX ==================
# ================== CONFUSION MATRIX WITH PERCENTAGES ==================
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
# ================== 4x3 ATTENTION HEATMAPS ==================
def plot_attention_grid():
    fig = plt.figure(figsize=(24, 32))
    axs = fig.subplots(4, 3)
    axs = axs.flatten()
    
    for idx, (src_chars, tgt_chars, attn) in enumerate(attention_samples[:12]):
        ax = axs[idx]
        sns.heatmap(
            attn,
            ax=ax,
            cmap='viridis',
            xticklabels=src_chars,
            yticklabels=tgt_chars,
            annot=True,
            fmt=".2f",
            annot_kws={'fontsize': 8}
        )
        ax.set_title(f"Sample {idx+1}", **english_font)
        ax.set_xlabel("Source (Latin)", **english_font)
        ax.set_ylabel("Target (Tamil)", **english_font)
        ax.set_xticklabels(ax.get_xticklabels(), **english_font)
        ax.set_yticklabels(ax.get_yticklabels(), fontproperties=tamil_font)
    
    for i in range(len(attention_samples), 12):
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig

attention_fig = plot_attention_grid()
wandb.log({'attention_heatmaps_4x3': wandb.Image(attention_fig)})
plt.close(attention_fig)

# ================== WANDB TABLE LOGGING ==================
df = pd.DataFrame(predictions)
df['token_accuracy'] = df['token_accuracy'].apply(lambda x: f"{x*100:.1f}%")
wandb.log({'predictions': wandb.Table(dataframe=df)})

# Finalize WandB
wandb.log({
    'exact_match_accuracy': exact_match_acc,
    'token_level_accuracy': token_level_acc
})
wandb.finish()

# Print results
print(f"\n{'Exact Match Accuracy:':<25} {exact_match_acc:.2%}")
print(f"{'Token-level Accuracy:':<25} {token_level_acc:.2%}")
print(f"Results saved to: {predictions_dir}")
