import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch, hidden_size] (decoder last hidden)
        # encoder_outputs: [batch, src_len, hidden_size]
        batch_size = encoder_outputs.size(0)
        src_len = encoder_outputs.size(1)

        # Repeat decoder hidden state src_len times
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch, src_len, hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch, src_len]
        return F.softmax(attention, dim=1)  # [batch, src_len]

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 embed_size=128, hidden_size=256,
                 num_encoder_layers=1, num_decoder_layers=1, dropout=0.3,
                 cell_type='lstm', init_method='xavier'):
        super(Seq2Seq, self).__init__()
        
        self.encoder = Encoder(src_vocab_size, embed_size, 
                               hidden_size, num_encoder_layers, 
                               dropout, cell_type, init_method)
        
        self.decoder = AttnDecoder(tgt_vocab_size, embed_size,
                                   hidden_size, num_decoder_layers,
                                   dropout, cell_type, init_method)

    def forward(self, src, tgt):
        encoder_outputs, hidden = self.encoder(src)
        hidden = self._adapt_hidden(hidden)
        output, attn_weights = self.decoder(tgt, hidden, encoder_outputs)
        return output, attn_weights

    def _adapt_hidden(self, hidden):
        def pad_or_truncate(h, target_layers):
            current_layers = h.size(0)
            if current_layers == target_layers:
                return h
            elif current_layers > target_layers:
                return h[:target_layers]
            else:  # pad with zeros
                pad_shape = (target_layers - current_layers, *h.shape[1:])
                padding = h.new_zeros(pad_shape)
                return torch.cat([h, padding], dim=0)

        if isinstance(hidden, tuple):  # LSTM
            h, c = hidden
            h = pad_or_truncate(h, self.decoder.rnn.num_layers)
            c = pad_or_truncate(c, self.decoder.rnn.num_layers)
            return (h, c)
        else:  # GRU or RNN
            return pad_or_truncate(hidden, self.decoder.rnn.num_layers)

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_encoder_layers, dropout, cell_type, init_method):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = self._get_rnn_cell(embed_size, hidden_size,
                                      num_encoder_layers, dropout, cell_type)
        self.init_weights(init_method)

    def _get_rnn_cell(self, embed_size, hidden_size, 
                      num_encoder_layers, dropout, cell_type):
        if cell_type.lower() == 'lstm':
            return nn.LSTM(embed_size, hidden_size, num_encoder_layers,
                           dropout=dropout if num_encoder_layers > 1 else 0, batch_first=True)
        elif cell_type.lower() == 'gru':
            return nn.GRU(embed_size, hidden_size, num_encoder_layers,
                          dropout=dropout if num_encoder_layers > 1 else 0, batch_first=True)
        else:
            return nn.RNN(embed_size, hidden_size, num_encoder_layers,
                          nonlinearity='tanh', dropout=dropout if num_encoder_layers > 1 else 0, batch_first=True)

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
        return outputs, hidden  # outputs: [batch, src_len, hidden_size]

class AttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size,
                 num_decoder_layers, dropout, cell_type, init_method):
        super(AttnDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = self._get_rnn_cell(embed_size + hidden_size, hidden_size,
                                      num_decoder_layers, dropout, cell_type)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.attention = Attention(hidden_size)
        self.cell_type = cell_type.lower()
        self.init_weights(init_method)

    def _get_rnn_cell(self, input_size, hidden_size, 
                      num_decoder_layers, dropout, cell_type):
        if cell_type.lower() == 'lstm':
            return nn.LSTM(input_size, hidden_size, num_decoder_layers,
                           dropout=dropout if num_decoder_layers > 1 else 0, batch_first=True)
        elif cell_type.lower() == 'gru':
            return nn.GRU(input_size, hidden_size, num_decoder_layers,
                          dropout=dropout if num_decoder_layers > 1 else 0, batch_first=True)
        else:
            return nn.RNN(input_size, hidden_size, num_decoder_layers,
                          nonlinearity='tanh', dropout=dropout if num_decoder_layers > 1 else 0, batch_first=True)

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

    def forward(self, x, hidden, encoder_outputs):
        # x: [batch, tgt_len]
        # hidden: (num_layers, batch, hidden_size) or tuple for LSTM
        batch_size, tgt_len = x.size()
        outputs = []
        attn_weights_list = []
        input_token = x[:, 0]  # <sos> for each sequence

        # Prepare initial hidden state for decoder
        if self.cell_type == 'lstm':
            h, c = hidden
        else:
            h = hidden

        for t in range(1, tgt_len):
            embedded = self.embedding(input_token).unsqueeze(1)  # [batch, 1, embed_size]
            # Attention: scores over encoder outputs
            attn_weights = self.attention(h[-1], encoder_outputs)  # [batch, src_len]
            attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden_size]
            rnn_input = torch.cat((embedded, attn_applied), dim=2)  # [batch, 1, embed+hidden]
            if self.cell_type == 'lstm':
                output, (h, c) = self.rnn(rnn_input, (h, c))
            else:
                output, h = self.rnn(rnn_input, h)
            output = self.fc(output.squeeze(1))  # [batch, vocab_size]
            outputs.append(output)
            attn_weights_list.append(attn_weights)
            input_token = x[:, t]  # next input token

        outputs = torch.stack(outputs, dim=1)  # [batch, tgt_len-1, vocab_size]
        attn_weights = torch.stack(attn_weights_list, dim=1)  # [batch, tgt_len-1, src_len]
        return outputs, attn_weights

    def beam_search_decode(self, hidden, encoder_outputs, max_len=30, beam_width=3, sos_idx=1, eos_idx=2):
        device = next(self.parameters()).device
        sequences = [[[], 0.0, hidden]]
        for _ in range(max_len):
            all_candidates = []
            for seq, score, hidden in sequences:
                input_token = torch.tensor([[seq[-1]]] if seq else [[sos_idx]], device=device)
                embedded = self.embedding(input_token)
                # Attention
                if self.cell_type == 'lstm':
                    h, c = hidden
                    attn_weights = self.attention(h[-1], encoder_outputs)
                    attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                    rnn_input = torch.cat((embedded, attn_applied), dim=2)
                    output, (h_next, c_next) = self.rnn(rnn_input, (h, c))
                    hidden_next = (h_next, c_next)
                else:
                    h = hidden
                    attn_weights = self.attention(h[-1], encoder_outputs)
                    attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
                    rnn_input = torch.cat((embedded, attn_applied), dim=2)
                    output, h_next = self.rnn(rnn_input, h)
                    hidden_next = h_next
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
    parser.add_argument('--src_vocab_size', type=int, default=10000)
    parser.add_argument('--tgt_vocab_size', type=int, default=10000)
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
    model = Seq2Seq(args.src_vocab_size, args.tgt_vocab_size,
                    args.embed_size, args.hidden_size,
                    args.num_encoder_layers, args.num_decoder_layers, args.dropout,
                    args.cell_type, args.init_method)
    print(model)
