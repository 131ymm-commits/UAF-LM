import torch
import torch.optim as optim
import numpy as np

def train_model(model, dataloader, epochs=20, lr=0.001, lambda_uaf=0.1, device='cpu', verbose=True):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    losses = []
    gammas = []
    for epoch in range(epochs):
        total_loss = 0.0
        epoch_gammas = []
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits, _, gamma = model(inputs, compute_gamma=True)
            loss_ce = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss_uaf = lambda_uaf * (1 - gamma)
            loss = loss_ce + loss_uaf
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_ce.item()
            epoch_gammas.append(gamma.item())
        avg_loss = total_loss / len(dataloader)
        avg_gamma = np.mean(epoch_gammas)
        losses.append(avg_loss)
        gammas.append(avg_gamma)
        if verbose and (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Gamma: {avg_gamma:.4f}")
    return losses, gammas

def generate_text(model, start_words, length=50, temperature=0.8, vocab=None, inv_vocab=None, device='cpu'):
    model.eval()
    words = start_words.split()
    tokens = [vocab.get(w, vocab['<UNK>']) for w in words]
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    hidden = None
    generated = words[:]
    with torch.no_grad():
        for _ in range(length):
            logits, hidden, _ = model(input_ids, hidden)
            logits = logits[0, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            generated.append(inv_vocab[next_token])
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(device)], dim=1)
            if input_ids.size(1) > 30:
                input_ids = input_ids[:, -30:]
    return ' '.join(generated)
