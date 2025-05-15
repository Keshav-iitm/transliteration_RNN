import argparse
import time
import torch
import wandb
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from model_RNN import Seq2Seq
from dataset_RNN import DakshinaDataset

def print_device_info(device):
    if device.type == 'cuda':
        print(f"🚀 Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"    Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        print(f"    Memory Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
    else:
        print("🖥️ Using CPU for training")

class TransliterationDataset(Dataset):
    def __init__(self, data, src_vocab, tgt_vocab):
        self.data = data
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_collate_fn(src_vocab, tgt_vocab):
    def collate_fn(batch):
        src_seqs, tgt_seqs = zip(*batch)
        max_src_len = max(len(seq) for seq in src_seqs)
        max_tgt_len = max(len(seq) for seq in tgt_seqs)

        src_padded = [
            [src_vocab[ch] for ch in seq] + [0] * (max_src_len - len(seq))
            for seq in src_seqs
        ]
        tgt_padded = [
            [1] + [tgt_vocab[ch] for ch in seq] + [2] + [0] * (max_tgt_len - len(seq))
            for seq in tgt_seqs
        ]

        return torch.tensor(src_padded), torch.tensor(tgt_padded)
    return collate_fn

def train(config=None):
    with wandb.init(config=config, project="RNN-Transliteration", group=config.lang if config else None) as run:
        config = wandb.config 
        
        # Dynamic GPU detection
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print_device_info(device)

        run.name = (
            f"{config.cell_type}_bw{config.beam_width}_{config.lang}_"
            f"emb{config.embed_size}hs{config.hidden_size}_"
            f"en{config.num_encoder_layers}de{config.num_decoder_layers}_"
            f"do{config.dropout}_tf{config.teacher_forcing_ratio}_"
            f"bs{config.batch_size}_lr{config.learning_rate:.1e}"
        )
        print(f"\n=== Starting training run {wandb.run.name} ===")

        dataset = DakshinaDataset(config.data_dir, config.lang)
        src_vocab = dataset.src_vocab
        tgt_vocab = dataset.tgt_vocab

        # Verify data direction
        print("\nSample training pairs (Latin → Native):")
        for i in range(3):
            src, tgt = dataset.train_data[i]
            print(f"{src} → {tgt}")

        train_dataset = TransliterationDataset(dataset.train_data, src_vocab, tgt_vocab)
        val_dataset = TransliterationDataset(dataset.val_data, src_vocab, tgt_vocab)
        collate = get_collate_fn(src_vocab, tgt_vocab)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            pin_memory=(device.type == 'cuda'),
            collate_fn=collate,
            drop_last=True
        )

        model = Seq2Seq(
            src_vocab_size=len(src_vocab),
            tgt_vocab_size=len(tgt_vocab),
            embed_size=config.embed_size,
            hidden_size=config.hidden_size,
            num_encoder_layers=config.num_encoder_layers,
            num_decoder_layers=config.num_decoder_layers,
            dropout=config.dropout,
            cell_type=config.cell_type,
            init_method=config.init_method
        ).to(device)

        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

        best_val_acc = 0
        for epoch in range(config.epochs):
            start_time = time.time()
            model.train()
            train_loss = train_acc = train_total = 0
            
            for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Train]"):
                src, tgt = src.to(device), tgt.to(device)
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]

                optimizer.zero_grad()
                outputs = model(src, tgt_input)
                loss = criterion(outputs.view(-1, len(tgt_vocab)), tgt_output.reshape(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                preds = outputs.argmax(dim=-1)
                train_acc += (preds == tgt_output).sum().item()
                train_total += tgt_output.numel()

            model.eval()
            val_loss = val_acc = val_total = 0
            with torch.no_grad():
                for src, tgt in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.epochs} [Val]"):
                    src, tgt = src.to(device), tgt.to(device)
                    tgt_input = tgt[:, :-1]
                    tgt_output = tgt[:, 1:]

                    outputs = model(src, tgt_input)
                    loss = criterion(outputs.view(-1, len(tgt_vocab)), tgt_output.reshape(-1))
                    
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=-1)
                    val_acc += (preds == tgt_output).sum().item()
                    val_total += tgt_output.numel()

            train_acc = 100 * train_acc / train_total
            val_acc = 100 * val_acc / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = f"best_model_{wandb.run.id}.pt"
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)
                print(f"💾 New best model saved with val_acc: {val_acc:.2f}%")

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss / len(val_loader),
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'beam_width': config.beam_width
            })

            print(f"\n⏱️ Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        print(f"\n🏆 Best validation accuracy: {best_val_acc:.2f}%")

def get_sweep_config():
    return {
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'data_dir': {'value': './dakshina_dataset_v1.0'},
            'lang': {'value': 'ta'},
            'epochs': {'value': 20},
            'device': {'value': 'cuda' if torch.cuda.is_available() else 'cpu'},
            'teacher_forcing_ratio': {'values': [0.5, 0.7, 1.0]},
            'beam_width': {'values': [1, 3, 5]},
            'embed_size': {'values': [128, 256]},
            'hidden_size': {'values': [256, 512]},
            'num_encoder_layers': {'values': [1, 2]},
            'num_decoder_layers': {'values': [1, 2]},
            'cell_type': {'values': ['rnn','lstm', 'gru']},
            'dropout': {'values': [0.2, 0.3]},
            'learning_rate': {'min': 1e-4, 'max': 1e-3},
            'batch_size': {'values': [32, 64]},
            'init_method': {'values': ['xavier', 'he', 'default']}
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dakshina_dataset_v1.0')
    parser.add_argument('--lang', type=str, default='ta')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(get_sweep_config(), project="Eng_tamil-Transliteration")
        wandb.agent(sweep_id, function=train, count=50)
    else:
        config = {
            'data_dir': args.data_dir,
            'lang': args.lang,
            'epochs': 20,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'teacher_forcing_ratio': 0.7,
            'beam_width': 3,
            'embed_size': 256,
            'hidden_size': 512,
            'num_encoder_layers': 2,
            'num_decoder_layers': 2,
            'cell_type': 'lstm',
            'dropout': 0.3,
            'learning_rate': 0.001,
            'batch_size': 64,
            'init_method': 'xavier'
        }
        train(config=config)

if __name__ == '__main__':
    main()
