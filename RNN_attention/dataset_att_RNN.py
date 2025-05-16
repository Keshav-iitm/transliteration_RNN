import argparse
import os
from collections import Counter
import torch

class DakshinaDataset:
    def __init__(self, data_dir='dakshina_dataset_v1.0', lang='ta'):
        self.data_dir = data_dir
        self.lang = lang
        self.train_data = self.load_data('train')
        self.val_data = self.load_data('dev')
        self.test_data = self.load_data('test')
        
        # Source=Latin, Target=Native
        self.src_vocab = self.build_vocab([x[0] for x in self.train_data])  # Latin
        self.tgt_vocab = self.build_vocab([x[1] for x in self.train_data])  # Native

    def load_data(self, split):
        data = []
        file_path = os.path.join(
            self.data_dir,
            self.lang,
            'lexicons',
            f'{self.lang}.translit.sampled.{split}.tsv'
        )
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        roman = parts[1].strip()  # Latin input (source)
                        native = parts[0].strip()  # Native output (target)
                        data.append((roman, native))  # (source, target)
        return data

    def build_vocab(self, texts):
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        counter = Counter()
        for text in texts:
            counter.update(text)
        for char, _ in counter.most_common():
            if char not in vocab:
                vocab[char] = len(vocab)
        return vocab

def get_collate_fn(dataset):
    def collate_fn(batch):
        src_seqs, tgt_seqs = zip(*batch)
        max_src_len = max(len(seq) for seq in src_seqs)
        max_tgt_len = max(len(seq) for seq in tgt_seqs) + 2  # +2 for <sos> and <eos>

        src_padded = []
        tgt_padded = []

        for src, tgt in zip(src_seqs, tgt_seqs):
            # Source=Latin, Target=Native
            src_idx = [dataset.src_vocab[c] for c in src]  # Latin indices
            tgt_idx = [1] + [dataset.tgt_vocab[c] for c in tgt] + [2]  # Native indices

            src_padded.append(src_idx + [0] * (max_src_len - len(src_idx)))
            tgt_padded.append(tgt_idx + [0] * (max_tgt_len - len(tgt_idx)))

        return (
            torch.tensor(src_padded, dtype=torch.long),
            torch.tensor(tgt_padded, dtype=torch.long)
        )
    return collate_fn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, 
                       default='dakshina_dataset_v1.0',
                       help='Path to dataset root directory')
    parser.add_argument('--lang', type=str, default='ta',
                       choices=['ta', 'hi', 'bn', 'gu', 'kn', 'ml',
                               'mr', 'pa', 'sd', 'si', 'te', 'ur'],
                       help='ISO 639-1 language code')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    dataset = DakshinaDataset(args.data_dir, args.lang)
    
    print(f"\nLanguage: {args.lang}")
    print(f"Train samples: {len(dataset.train_data)}")
    print(f"Validation samples: {len(dataset.val_data)}")
    print(f"Test samples: {len(dataset.test_data)}")
    print(f"Source (Latin) vocab size: {len(dataset.src_vocab)}")
    print(f"Target (Native) vocab size: {len(dataset.tgt_vocab)}")
    
    # Verify data direction with samples
    print("\nFirst 5 training pairs (Latin → Native):")
    for i in range(5):
        src, tgt = dataset.train_data[i]
        print(f"[{i+1}] {src} → {tgt}")
