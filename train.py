#!/usr/bin/env python3
"""
Train UAF-LM on a small text corpus.
Usage: python train.py [--lambda_uaf 0.1] [--epochs 20]
"""

import argparse
import torch
import matplotlib.pyplot as plt
from uaf_lm.data_utils import load_corpus, create_dataloader
from uaf_lm.model import UAF_LSTM
from uaf_lm.train_utils import train_model, generate_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda_uaf', type=float, default=0.1, help='UAF regularization strength')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--no_cuda', action='store_true', help='Disable GPU')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # Загрузка данных (по умолчанию – Alice in Wonderland)
    dataloader, vocab_size, inv_vocab, vocab = load_corpus(batch_size=args.batch_size, seq_len=args.seq_len)
    print(f"Vocabulary size: {vocab_size}, sequences: {len(dataloader.dataset)}")

    # Базовая модель
    model_base = UAF_LSTM(vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers)
    print("\nTraining baseline model (λ=0)...")
    _, _ = train_model(model_base, dataloader, epochs=args.epochs, lambda_uaf=0.0, device=device, verbose=True)

    # UAF-модель
    model_uaf = UAF_LSTM(vocab_size, args.embedding_dim, args.hidden_dim, args.num_layers)
    print("\nTraining UAF-regularized model (λ={})...".format(args.lambda_uaf))
    _, _ = train_model(model_uaf, dataloader, epochs=args.epochs, lambda_uaf=args.lambda_uaf, device=device, verbose=True)

    # Генерация от одной и той же seed
    seed = "Alice was"
    print("\n=== GENERATED TEXT ===")
    print("Baseline:")
    print(generate_text(model_base, seed, length=50, temperature=args.temperature, vocab=vocab, inv_vocab=inv_vocab, device=device))
    print("\nUAF model:")
    print(generate_text(model_uaf, seed, length=50, temperature=args.temperature, vocab=vocab, inv_vocab=inv_vocab, device=device))

if __name__ == '__main__':
    main()
