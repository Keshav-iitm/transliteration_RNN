import argparse
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,  # src=Latin, tgt=Devanagari
                 embed_size=128, hidden_size=256,
                 num_encoder_layers=1, num_decoder_layers=1, dropout=0.3,
                 cell_type='lstm', init_method='xavier'):
        super(Seq2Seq, self).__init__()
        
        # ENCODER: Processes Latin characters (English input)
        self.encoder = Encoder(
            src_vocab_size,  # Must be Latin vocabulary size
            embed_size, 
            hidden_size, 
            num_encoder_layers, 
            dropout, 
            cell_type, 
            init_method
        )
        
        # DECODER: Generates Devanagari characters (Native output)
        self.decoder = Decoder(
            tgt_vocab_size,  # Must be Devanagari vocabulary size
            embed_size,
            hidden_size,
            num_decoder_layers,
            dropout, 
            cell_type, 
            init_method
        )

    def forward(self, src, tgt):
        # src: Latin sequences, tgt: Devanagari sequences
        encoder_outputs, hidden = self.encoder(src)
        hidden = self._adapt_hidden(hidden)
        output, _ = self.decoder(tgt, hidden)
        return output

    def _adapt_hidden(self, hidden):
        def pad_or_truncate(h, target_layers):
            current_layers = h.size(0)
            if current_layers == target_layers:
                return h
            elif current_layers > target_layers:
                return h[:target_layers]
            else:
                pad_shape = (target_layers - current_layers, *h.shape[1:])
                padding = h.new_zeros(pad_shape)
                return torch.cat([h, padding], dim=0)

        if isinstance(hidden, tuple):
            h, c = hidden
            h = pad_or_truncate(h, self.decoder.rnn.num_layers)
            c = pad_or_truncate(c, self.decoder.rnn.num_layers)
            return (h, c)
        else:
            return pad_or_truncate(hidden, self.decoder.rnn.num_layers)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_encoder_layers, dropout, cell_type, init_method):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Latin embeddings
        self.rnn = self._get_rnn_cell(embed_size, hidden_size,
                                    num_encoder_layers, dropout, cell_type)
        self.init_weights(init_method)

    def _get_rnn_cell(self, embed_size, hidden_size, 
                    num_encoder_layers, dropout, cell_type):
        if cell_type.lower() == 'lstm':
            return nn.LSTM(
                embed_size, hidden_size, num_encoder_layers,
                dropout=dropout if num_encoder_layers > 1 else 0, 
                batch_first=True
            )
        elif cell_type.lower() == 'gru':
            return nn.GRU(
                embed_size, hidden_size, num_encoder_layers,
                dropout=dropout if num_encoder_layers > 1 else 0, 
                batch_first=True
            )
        else:
            return nn.RNN(
                embed_size, hidden_size, num_encoder_layers,
                nonlinearity='tanh', 
                dropout=dropout if num_encoder_layers > 1 else 0, 
                batch_first=True
            )

    def init_weights(self, method):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(param.data, gain=gain)
                elif method == 'he':
                    nn.init.kaiming_normal_(param.data, nonlinearity='tanh')

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_decoder_layers, dropout, cell_type, init_method):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)  # Devanagari embeddings
        self.rnn = self._get_rnn_cell(embed_size, hidden_size,
                                    num_decoder_layers, dropout, cell_type)
        self.fc = nn.Linear(hidden_size, vocab_size)  # Devanagari output
        
        self.init_weights(init_method)

    def _get_rnn_cell(self, embed_size, hidden_size, 
                    num_decoder_layers, dropout, cell_type):
        if cell_type.lower() == 'lstm':
            return nn.LSTM(
                embed_size, hidden_size, num_decoder_layers,
                dropout=dropout if num_decoder_layers > 1 else 0, 
                batch_first=True
            )
        elif cell_type.lower() == 'gru':
            return nn.GRU(
                embed_size, hidden_size, num_decoder_layers,
                dropout=dropout if num_decoder_layers > 1 else 0, 
                batch_first=True
            )
        else:
            return nn.RNN(
                embed_size, hidden_size, num_decoder_layers,
                nonlinearity='tanh', 
                dropout=dropout if num_decoder_layers > 1 else 0, 
                batch_first=True
            )

    def init_weights(self, method):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.embedding.weight)
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(param.data, gain=gain)
                elif method == 'he':
                    nn.init.kaiming_normal_(param.data, nonlinearity='tanh')
        nn.init.xavier_normal_(self.fc.weight.data)
        nn.init.constant_(self.fc.bias.data, 0)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.fc(output)
        return output, hidden
    
    def beam_search_decode(self, hidden, max_len=30, beam_width=3, sos_idx=1, eos_idx=2):
        device = next(self.parameters()).device
        sequences = [[[], 0.0, hidden]]  # (sequence, score, hidden_state)

        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden in sequences:
                input_token = torch.tensor([[seq[-1]]] if seq else [[sos_idx]], device=device)
                embedded = self.embedding(input_token)
                output, hidden_next = self.rnn(embedded, hidden)
                logits = self.fc(output[:, -1])
                probs = torch.log_softmax(logits, dim=1)

                topk_probs, topk_indices = torch.topk(probs, beam_width)

                for i in range(beam_width):
                    char_idx = topk_indices[0][i].item()
                    char_prob = topk_probs[0][i].item()
                    candidate = (seq + [char_idx], score + char_prob, hidden_next)
                    all_candidates.append(candidate)

            sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(seq and seq[-1] == eos_idx for seq, _, _ in sequences):
                break

        best_seq = sequences[0][0]
        return best_seq

def get_args():
    parser = argparse.ArgumentParser()
    # Critical parameters - must match data vocabulary sizes
    parser.add_argument('--src_vocab_size', type=int, default=10000,  # Latin
                        help='Must be size of Latin/English vocabulary')
    parser.add_argument('--tgt_vocab_size', type=int, default=10000,  # Devanagari 
                        help='Must be size of native script vocabulary')
    parser.add_argument('--embed_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--num_encoder_layers', type=int, default=1)
    parser.add_argument('--num_decoder_layers', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--cell_type', type=str, default='lstm',
                        choices=['rnn', 'gru', 'lstm'])
    parser.add_argument('--init_method', type=str, default='xavier',
                        choices=['xavier', 'he', 'default'])
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # When initializing: 
    # - src_vocab_size = len(latin_tokenizer)
    # - tgt_vocab_size = len(devanagari_tokenizer)
    model = Seq2Seq(
        args.src_vocab_size, 
        args.tgt_vocab_size,
        args.embed_size, 
        args.hidden_size,
        args.num_encoder_layers,
        args.num_decoder_layers, 
        args.dropout,
        args.cell_type, 
        args.init_method
    )
    print(model)
