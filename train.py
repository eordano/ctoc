"""
Train the token count predictor model.
"""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random

from model import TokenCounterCNN, text_to_bytes, count_parameters


class TokenCountDataset(Dataset):
    """Dataset of (text, count) pairs."""

    def __init__(self, data_file: str, max_len: int = 1024):
        self.max_len = max_len
        self.samples = []

        with open(data_file) as f:
            for line in f:
                data = json.loads(line)
                text = data["text"]
                count = data["count"]
                # Filter out very long samples
                if len(text.encode('utf-8')) <= max_len:
                    self.samples.append((text, count))

        print(f"Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, count = self.samples[idx]
        x = text_to_bytes(text, max_len=self.max_len)
        y = torch.tensor(count, dtype=torch.float)
        return x, y


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    total_samples = 0

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(x)
        loss = nn.L1Loss()(pred, y)  # L1 for robustness
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(x)
        total_samples += len(x)

    return total_loss / total_samples


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    exact_matches = 0

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = nn.L1Loss()(pred, y)

            total_loss += loss.item() * len(x)
            total_samples += len(x)

            # Count exact matches (rounded prediction == true count)
            rounded_pred = torch.round(pred)
            exact_matches += (rounded_pred == y).sum().item()

    avg_loss = total_loss / total_samples
    accuracy = exact_matches / total_samples

    return avg_loss, accuracy


def train(
    data_file: str = "training_data.jsonl",
    output_file: str = "model.pt",
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    val_split: float = 0.1,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    dataset = TokenCountDataset(data_file)

    # Split train/val
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    model = TokenCounterCNN(
        embed_dim=64,
        hidden_dim=128,
        num_layers=4,
        max_len=1024,
    ).to(device)

    print(f"Model parameters: {count_parameters(model):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_accuracy = 0
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_accuracy:.2%}")

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), output_file)
            print(f"  -> Saved best model (accuracy: {val_accuracy:.2%})")

    print(f"\nBest validation accuracy: {best_accuracy:.2%}")
    print(f"Model saved to {output_file}")


if __name__ == "__main__":
    train()
