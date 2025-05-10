import torch
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imageio
from torch.utils.data import DataLoader, Subset

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

def create_visualization(src_chars, tgt_chars, attn_weights, filename):
    """Create attention visualization for a single sample"""
    frames = []
    max_width = max(len(src_chars), 8)
    
    for step in range(len(tgt_chars)):
        fig = plt.figure(figsize=(max_width/2, 3), dpi=150)
        ax = fig.add_subplot(111)
        
        # Plot attention matrix
        ax.matshow(attn_weights[step].reshape(1, -1), 
                  cmap='Greens',
                  aspect='auto',
                  vmin=0,
                  vmax=1)
        
        # Axis labels
        ax.set_xlabel('Input Characters', labelpad=20)
        ax.set_ylabel('Attention Weights', labelpad=15)
        ax.xaxis.set_label_position('top')
        
        # Ticks and values
        ax.set_xticks(np.arange(len(src_chars)))
        ax.set_xticklabels(src_chars, rotation=45)
        ax.set_yticks([])
        ax.set_title(f"Generating '{tgt_chars[step]}' (Step {step+1})", pad=25)
        
        # Add attention values
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

def main():
    from dataset_att_RNN import DakshinaDataset, get_collate_fn
    from model_att_RNN import Seq2Seq

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = DakshinaDataset('./dakshina_dataset_v1.0', 'ta')
    
    # Load model
    model_path = 'best_att_model_qurwg2mv.pt'
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    model = Seq2Seq(
        src_vocab_size=len(dataset.src_vocab),
        tgt_vocab_size=len(dataset.tgt_vocab),
        embed_size=256,
        hidden_size=256,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.4,
        cell_type='gru'
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Select 3 distinct samples (lines 1, 1000, 2000 in 1-based numbering)
    test_size = len(dataset.test_data)
    sample_indices = [
        0,   # Line 1 (0-based index)
        min(999, test_size-1),   # Line 1000
        min(1999, test_size-1)   # Line 2000
    ]
    
    test_subset = Subset(dataset.test_data, sample_indices)
    test_loader = DataLoader(
        test_subset,
        batch_size=1,
        collate_fn=get_collate_fn(dataset),
        shuffle=False
    )
    
    for batch_idx, (src, tgt) in enumerate(test_loader):
        original_line_number = sample_indices[batch_idx] + 1  # Convert to 1-based
        with torch.no_grad():
            src, tgt = src.to(device), tgt.to(device)
            
            # Get attention weights
            outputs, attn_weights = model(src, tgt[:, :-1])
            attn_weights = attn_weights[0].cpu().numpy()
            
            # Decode sequences
            src_chars = [
                dataset.src_vocab.get(int(ix), '') 
                for ix in src[0].cpu().numpy() 
                if ix not in [0, 1, 2]
            ]
            pred_chars = [
                dataset.tgt_vocab.get(int(ix), '') 
                for ix in outputs.argmax(-1)[0].cpu().numpy()
                if ix not in [0, 2]
            ]

            # Generate visualization
            if len(pred_chars) > 0:
                filename = f'attention_line_{original_line_number}.gif'
                create_visualization(src_chars, pred_chars, attn_weights, filename)
                print(f"✅ Saved visualization for line {original_line_number}")

if __name__ == "__main__":
    main()
