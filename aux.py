import torch
import torch.nn as nn
import math
import math


class LSTMLanguageModel(nn.Module):
    """
    LSTM Language Model

    Args:
        vocab_size: int, size of the vocabulary
        emb_dim: int, size of the word embeddings
        hid_dim: int, size of the hidden state
        num_layers: int, number of layers in the LSTM
        dropout_rate: float, dropout rate
    """

    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim = hid_dim
        self.emb_dim = emb_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hid_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim, self.hid_dim).uniform_(-init_range_other, init_range_other)  # We
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim, self.hid_dim).uniform_(-init_range_other, init_range_other)  # Wh

    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, src, hidden):
        embedding = self.dropout(self.embedding(src))
        output, hidden = self.lstm(embedding, hidden)
        output = self.dropout(output)
        prediction = self.fc(output)
        return prediction, hidden


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    """
    Generate text given a prompt

    Args:
        prompt: str, the input prompt
        max_seq_len: int, maximum sequence length
        temperature: float, temperature for sampling
        model: nn.Module, the language model
        tokenizer: torchtext.data.utils.get_tokenizer, the tokenizer
        vocab: torchtext.vocab.Vocab, the vocabulary
        device: torch.device, the device to run the model
        seed: int, random seed
    """
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)

            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)
            prediction = torch.multinomial(probs, num_samples=1).item()

            while prediction == vocab["<unk>"]:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab["<eos>"]:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens
