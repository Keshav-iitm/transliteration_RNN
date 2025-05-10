import argparse
import time
import torch
import wandb
from tqdm import tqdm
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from model_att_RNN import Seq2Seq
from dataset_att_RNN import DakshinaDataset

def print_device_info(device):
    if device == 'cuda':
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

def get_collate_fn(dataset):
    def collate_fn(batch):
        src_seqs, tgt_seqs = zip(*batch)
        max_src_len = max(len(seq) for seq in src_seqs)
        max_tgt_len = max(len(seq) for seq in tgt_seqs) + 2  # +2 for <sos> and <eos>

        src_padded = [
            [dataset.src_vocab[c] for c in src] + [0] * (max_src_len - len(src))
            for src in src_seqs
        ]
        tgt_padded = [
            [1] + [dataset.tgt_vocab[c] for c in tgt] + [2] + [0] * (max_tgt_len - len(tgt) - 2)
            for tgt in tgt_seqs
        ]
        return torch.tensor(src_padded, dtype=torch.long), torch.tensor(tgt_padded, dtype=torch.long)
    return collate_fn

def train(config=None):
    with wandb.init(config=config, project="RNN-Transliteration-Attention", group=config.lang if config else None) as run:
        config = wandb.config 

        run.name = (
            f"{config.cell_type}_{config.lang}_embd{config.embed_size}_hs{config.hidden_size}"
            f"_en{config.num_encoder_layers}_de{config.num_decoder_layers}_do{config.dropout}"
            f"_bw{getattr(config, 'beam_width', 'NA')}_bs{config.batch_size}_lr{config.learning_rate:.1e}"
        )
        print(f"\n=== Starting training run {wandb.run.name} ===")
        device = torch.device(config.device)
        print_device_info(config.device)

        dataset = DakshinaDataset(config.data_dir, config.lang)
        src_vocab = dataset.src_vocab
        tgt_vocab = dataset.tgt_vocab

        train_dataset = TransliterationDataset(dataset.train_data, src_vocab, tgt_vocab)
        val_dataset = TransliterationDataset(dataset.val_data, src_vocab, tgt_vocab)
        collate = get_collate_fn(dataset)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            pin_memory=(config.device == 'cuda'),
            collate_fn=collate,
            drop_last=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            pin_memory=(config.device == 'cuda'),
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
                outputs, _ = model(src, tgt_input)

                # --- FIX: Truncate target to match output sequence length ---
                outputs_len = outputs.size(1)
                tgt_output = tgt_output[:, :outputs_len]
                assert outputs.shape[1] == tgt_output.shape[1], f"Decoder outputs {outputs.shape[1]}, targets {tgt_output.shape[1]}"
                # ----------------------------------------------------------

                loss = criterion(outputs.view(-1, len(tgt_vocab)), tgt_output.reshape(-1))
                loss.backward()
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

                    outputs, _ = model(src, tgt_input)

                    
                    outputs_len = outputs.size(1)
                    tgt_output = tgt_output[:, :outputs_len]
                    assert outputs.shape[1] == tgt_output.shape[1], f"Decoder outputs {outputs.shape[1]}, targets {tgt_output.shape[1]}"
                    

                    loss = criterion(outputs.view(-1, len(tgt_vocab)), tgt_output.reshape(-1))
                    val_loss += loss.item()
                    preds = outputs.argmax(dim=-1)
                    val_acc += (preds == tgt_output).sum().item()
                    val_total += tgt_output.numel()

            train_acc = 100 * train_acc / train_total
            val_acc = 100 * val_acc / val_total

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_path = f"best_att_model_{wandb.run.id}.pt"
                torch.save(model.state_dict(), model_path)
                wandb.save(model_path)
                print(f"💾 New best model saved with val_acc: {val_acc:.2f}%")

            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss / len(val_loader),
                'val_acc': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            })

            print(f"\n⏱️ Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

        print(f"\n🏆 Best validation accuracy: {best_val_acc:.2f}%")

def get_sweep_config():
    sweep_params = {
        'embed_size': {'values': [64, 128, 256]},
        'hidden_size': {'values': [128, 256]},
        'num_encoder_layers': {'values': [1, 2]},
        'num_decoder_layers': {'values': [1, 2]},
        'cell_type': {'values': ['lstm', 'gru', 'rnn']},
        'init_method': {'values': ['xavier', 'he', 'default']},
        'dropout': {'values': [0.0, 0.2, 0.4]},
        'learning_rate': {'min': 1e-5, 'max': 1e-3, 'distribution': 'log_uniform_values'},
        'batch_size': {'values': [16, 32, 64]},
        'beam_width': {'values': [1, 3, 5]},
    }

    sweep_name = f"translit-attn-sweep-{wandb.util.generate_id()}"
    return {
        'name': sweep_name,
        'method': 'bayes',
        'metric': {'name': 'val_acc', 'goal': 'maximize'},
        'parameters': {
            'data_dir': {'value': './dakshina_dataset_v1.0'},
            'lang': {'value': 'ta'},
            'epochs': {'value': 20},
            'device': {'value': 'cuda' if torch.cuda.is_available() else 'cpu'},
            **sweep_params
        }
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dakshina_dataset_v1.0')
    parser.add_argument('--lang', type=str, default='ta')
    parser.add_argument('--sweep', action='store_true')
    args = parser.parse_args()

    if args.sweep:
        sweep_id = wandb.sweep(get_sweep_config(), project="RNN-Transliteration-Attention")
        wandb.agent(sweep_id, function=train, count=80)
    else:
        sweep_config = get_sweep_config()
        sweep_defaults = sweep_config['parameters']
        config = {
            k: (v['value'] if 'value' in v else v['values'][0])
            for k, v in sweep_defaults.items()
        }   
        config.update({'data_dir': args.data_dir, 'lang': args.lang})
        train(config=config)

if __name__ == '__main__':
    main()
