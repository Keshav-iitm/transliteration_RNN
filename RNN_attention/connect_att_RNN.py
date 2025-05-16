import torch
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
from torch.utils.data import DataLoader, Subset
import argparse

matplotlib.use('Agg')
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.titlepad': 20,
    'axes.labelpad': 15
})

# Args parse
parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='best_att_model_wn3dcrxt.pt',
                    help='Path to the best attention model file')
parser.add_argument('--embed_size', type=int, default=128, help='Size of the embedding vectors.')
parser.add_argument('--hidden_size', type=int, default=256, help='Size of the hidden layers.')
parser.add_argument('--num_encoder_layers', type=int, default=1, help='Number of encoder layers.')
parser.add_argument('--num_decoder_layers', type=int, default=2, help='Number of decoder layers.')
parser.add_argument('--dropout', type=float, default=0.4, help='Dropout probability.')
parser.add_argument('--cell_type', type=str, choices=['lstm', 'gru'], default='gru', help='Type of RNN cell.')
args = parser.parse_args()

def create_visualization(src_chars, tgt_chars, attn_weights, filename):
    """Create attention visualization for a single sample"""
    frames = []
    max_width = max(len(src_chars), 8)
    
    for step in range(len(tgt_chars)):
        fig = plt.figure(figsize=(max_width/2, 3), dpi=150)
        ax = fig.add_subplot(111)
        
        ax.matshow(attn_weights[step].reshape(1, -1), 
                  cmap='Greens',
                  aspect='auto',
                  vmin=0,
                  vmax=1)
        
        ax.set_xlabel('Input Characters', labelpad=20)
        ax.set_ylabel('Attention Weights', labelpad=15)
        ax.xaxis.set_label_position('top')
        
        ax.set_xticks(np.arange(len(src_chars)))
        ax.set_xticklabels(src_chars, rotation=45)
        ax.set_yticks([])
        ax.set_title(f"Generating  (Step {step+1})", pad=25)
        
        for idx, weight in enumerate(attn_weights[step]):
            ax.text(idx, 0, f"{weight:.2f}", 
                   ha='center', va='center',
                   color='white' if weight > 0.5 else 'darkgreen',
                   fontsize=9)
        
        fig.tight_layout(pad=3.0)
        fig.canvas.draw()
        frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)
    
    imageio.mimsave(filename, frames, duration=1000, subrectangles=True)

def create_connectivity_visualization(src_chars, tgt_chars, attn_weights, filename):
    """Create connectivity visualization showing attention alignment between input and output"""
    # Ensuring valid dimensions
    attention_matrix = attn_weights[:len(tgt_chars), :len(src_chars)]
    
    fig = plt.figure(figsize=(max(8, len(src_chars)*0.7), max(6, len(tgt_chars)*0.7)), dpi=150)
    ax = fig.add_subplot(111)
    
    # Plot the heatmap
    im = ax.imshow(attention_matrix, cmap='viridis', interpolation='nearest')
    
    # Add colorbar
    cbar = fig.colorbar(im)
    cbar.set_label('Attention Weight', rotation=90)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(src_chars)))
    ax.set_yticks(np.arange(len(tgt_chars)))
    ax.set_xticklabels(src_chars, rotation=45, ha='right')
    ax.set_yticklabels(tgt_chars)
    
    # Label the axes
    ax.set_xlabel('Input Characters (Latin/English)')
    ax.set_ylabel('Output Characters (Tamil)')
    ax.set_title('Connectivity: Character-level Attention')
    
    # Add attention weight annotations
    for i in range(len(tgt_chars)):
        for j in range(len(src_chars)):
            ax.text(j, i, f"{attention_matrix[i, j]:.2f}",
                   ha="center", va="center", 
                   color="white" if attention_matrix[i, j] > 0.5 else "black",
                   fontsize=8)
    
    fig.tight_layout()
    plt.savefig(filename)
    plt.close(fig)

def main():
    from dataset_att_RNN import DakshinaDataset, get_collate_fn
    from model_att_RNN import Seq2Seq

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DakshinaDataset('./dakshina_dataset_v1.0', 'ta')
    
    # Create inverse vocabulary mappings
    inv_src_vocab = {v: k for k, v in dataset.src_vocab.items()}
    inv_tgt_vocab = {v: k for k, v in dataset.tgt_vocab.items()}
    
    # Load model
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    model = Seq2Seq(
        src_vocab_size=len(dataset.src_vocab),
        tgt_vocab_size=len(dataset.tgt_vocab),
        embed_size=args.embed_size,
        hidden_size=args.hidden_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dropout=args.dropout,
        cell_type=args.cell_type
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_size = len(dataset.test_data)
    sample_indices = [
        0,
        min(999, test_size-1),
        min(1999, test_size-1)
    ]
    
    test_subset = Subset(dataset.test_data, sample_indices)
    test_loader = DataLoader(
        test_subset,
        batch_size=1,
        collate_fn=get_collate_fn(dataset),
        shuffle=False
    )
    
    for batch_idx, (src, tgt) in enumerate(test_loader):
        original_line_number = sample_indices[batch_idx] + 1
        with torch.no_grad():
            src, tgt = src.to(device), tgt.to(device)
            outputs, attn_weights = model(src, tgt[:, :-1])
            attn_weights = attn_weights[0].cpu().numpy()
            
            # Extract source and target characters
            src_chars = [inv_src_vocab[idx.item()] for idx in src[0] if idx.item() != 0]
            pred_chars = [inv_tgt_vocab[idx.item()] for idx in outputs.argmax(-1)[0] if idx.item() not in [0, 1, 2]]

            if len(pred_chars) > 0:
                # Create original frame-by-frame visualization 
                gif_filename = f'attention_line_{original_line_number}.gif'
                create_visualization(src_chars, pred_chars, attn_weights, gif_filename)
                
                # Create new connectivity visualization 
                conn_filename = f'connectivity_line_{original_line_number}.png'
                create_connectivity_visualization(src_chars, pred_chars, attn_weights, conn_filename)
                
                print(f"✅ Saved visualizations for line {original_line_number}")
                print(f"   - Animation: {gif_filename}")
                print(f"   - Connectivity: {conn_filename}")

if __name__ == "__main__":
    main()
