import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from collections import Counter
import urllib.request
import numpy as np

class TextDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx][0], self.sequences[idx][1]

def load_corpus(batch_size=32, seq_len=30):
    # Скачиваем небольшой корпус "Alice in Wonderland" (первые 2000 слов)
    try:
        with urllib.request.urlopen("https://www.gutenberg.org/files/11/11-0.txt") as f:
            raw = f.read().decode('utf-8', errors='ignore')
        lines = [l.strip() for l in raw.split('\n') if len(l.strip()) > 30 and not l.startswith('[')]
        text = ' '.join(lines[:200])
    except:
        text = "Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. Once or twice she had peeped into the book her sister was reading. So she was considering in her own mind whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies. Suddenly a White Rabbit with pink eyes ran close by her."

    # Токенизация
    words = text.split()
    word_counts = Counter(words)
    vocab = {word: i+1 for i, (word, _) in enumerate(word_counts.most_common(2000))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = len(vocab)
    inv_vocab = {i: w for w, i in vocab.items()}
    def encode(sentence): return [vocab.get(w, vocab['<UNK>']) for w in sentence.split()]

    # Создание последовательностей
    encoded = encode(text)
    sequences = []
    for i in range(0, len(encoded)-seq_len-1, 1):
        seq = encoded[i:i+seq_len+1]
        sequences.append((torch.tensor(seq[:-1]), torch.tensor(seq[1:])))
    dataset = TextDataset(sequences)
    def collate(batch):
        inputs = pad_sequence([item[0] for item in batch], batch_first=True, padding_value=0)
        targets = pad_sequence([item[1] for item in batch], batch_first=True, padding_value=0)
        return inputs, targets
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate)
    return dataloader, len(vocab), inv_vocab, vocab
