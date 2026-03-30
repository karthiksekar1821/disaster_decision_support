"""
TextCNN Classifier for disaster tweet classification (Addition 5).

Implements a CNN text classifier with filter sizes [2, 3, 4] and 128 filters each.
Uses a simple word embedding layer initialised randomly with embedding_dim=128.
Saves predictions in the same npz format as transformer models.

Usage (from Kaggle/Colab):
    from cnn_classifier import train_cnn, predict_cnn
    model, vocab = train_cnn(train_texts, train_labels, val_texts, val_labels,
                             test_texts, test_labels, output_dir, class_weights)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter


SEED = 42


def set_all_seeds(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Vocabulary ───────────────────────────────────────────────────────────────

class Vocabulary:
    """Simple word-level vocabulary."""

    def __init__(self, max_vocab_size=30000, min_freq=2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        """Build vocabulary from list of texts."""
        counter = Counter()
        for text in texts:
            counter.update(text.lower().split())

        # Filter by min frequency and take top max_vocab_size
        sorted_words = sorted(
            [(w, c) for w, c in counter.items() if c >= self.min_freq],
            key=lambda x: -x[1]
        )[:self.max_vocab_size - 2]  # -2 for PAD and UNK

        for word, _ in sorted_words:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"  Vocabulary size: {len(self.word2idx)}")
        return self

    def encode(self, text, max_length=128):
        """Encode text to list of indices."""
        tokens = text.lower().split()[:max_length]
        indices = [self.word2idx.get(w, 1) for w in tokens]  # 1 = UNK
        # Pad to max_length
        indices = indices + [0] * (max_length - len(indices))
        return indices

    def __len__(self):
        return len(self.word2idx)


# ── Dataset ──────────────────────────────────────────────────────────────────

class TextDataset(Dataset):
    """Simple text classification dataset."""

    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        indices = self.vocab.encode(self.texts[idx], self.max_length)
        return (
            torch.tensor(indices, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


# ── TextCNN Model ────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    TextCNN with multiple filter sizes.

    Architecture:
        Embedding(vocab_size, 128)
        → Conv1d with filter sizes [2, 3, 4] × 128 filters each
        → MaxPool over sequence
        → Dropout(0.5)
        → Linear(384, num_classes)
    """

    def __init__(self, vocab_size, embedding_dim=128, num_classes=5,
                 filter_sizes=(2, 3, 4), num_filters=128, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch, embed_dim, seq_len)

        conv_outputs = []
        for conv in self.convs:
            c = F.relu(conv(embedded))  # (batch, num_filters, seq_len - fs + 1)
            c = F.max_pool1d(c, c.size(2)).squeeze(2)  # (batch, num_filters)
            conv_outputs.append(c)

        cat = torch.cat(conv_outputs, dim=1)  # (batch, num_filters * len(filter_sizes))
        cat = self.dropout(cat)
        logits = self.fc(cat)  # (batch, num_classes)
        return logits


# ── Training ─────────────────────────────────────────────────────────────────

def train_cnn(
    train_texts, train_labels,
    val_texts, val_labels,
    test_texts, test_labels,
    output_dir,
    class_weights=None,
    num_epochs=10,
    batch_size=64,
    learning_rate=1e-3,
    device=None,
):
    """
    Train a TextCNN model and save predictions.

    Args:
        train_texts: list of training texts
        train_labels: array of training labels
        val_texts: list of validation texts
        val_labels: array of validation labels
        test_texts: list of test texts
        test_labels: array of test labels
        output_dir: directory to save model and predictions
        class_weights: list of class weights (optional)
        num_epochs: training epochs
        batch_size: batch size
        learning_rate: learning rate
        device: 'cuda' or 'cpu'

    Returns:
        model, vocab, results dict
    """
    set_all_seeds(SEED)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_labels = np.array(train_labels)
    val_labels = np.array(val_labels)
    test_labels = np.array(test_labels)
    num_classes = len(np.unique(train_labels))

    # Build vocabulary
    print("\nTraining TextCNN...")
    vocab = Vocabulary(max_vocab_size=30000, min_freq=2)
    vocab.build(train_texts)

    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels, vocab)
    val_dataset = TextDataset(val_texts, val_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Model
    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=128,
        num_classes=num_classes,
        filter_sizes=[2, 3, 4],
        num_filters=128,
    ).to(device)

    # Loss with class weights
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5
    )

    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(num_epochs):
        # Training
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        val_preds, val_probs_list = _evaluate_model(model, val_loader, device)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = accuracy_score(val_labels, val_preds)

        scheduler.step(val_f1)

        print(f"  Epoch {epoch+1}/{num_epochs}: loss={avg_loss:.4f}, "
              f"val_f1={val_f1:.4f}, val_acc={val_acc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final predictions
    val_preds, val_probs = _evaluate_model(model, val_loader, device)
    test_preds, test_probs = _evaluate_model(model, test_loader, device)

    test_f1 = f1_score(test_labels, test_preds, average="macro")
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"\n  TextCNN Final — Test F1: {test_f1:.4f}, Test Acc: {test_acc:.4f}")

    # Save predictions
    pred_dir = os.path.join(output_dir, "cnn", "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    np.savez(
        os.path.join(pred_dir, "val_predictions.npz"),
        preds=val_preds, labels=val_labels, probs=val_probs,
    )
    np.savez(
        os.path.join(pred_dir, "test_predictions.npz"),
        preds=test_preds, labels=test_labels, probs=test_probs,
    )

    # Save model
    model_dir = os.path.join(output_dir, "cnn", "best_model")
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    torch.save(vocab, os.path.join(model_dir, "vocab.pt"))

    print(f"  CNN predictions saved to {pred_dir}")

    return model, vocab, {
        "val_f1": float(best_val_f1),
        "test_f1": float(test_f1),
        "test_acc": float(test_acc),
    }


def _evaluate_model(model, loader, device):
    """Evaluate model and return predictions and probabilities."""
    model.eval()
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_preds), np.concatenate(all_probs)


def predict_cnn(model, vocab, texts, device=None, batch_size=64):
    """Run CNN inference on new texts."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    # Encode texts
    encoded = [vocab.encode(t) for t in texts]
    all_probs = []

    for i in range(0, len(encoded), batch_size):
        batch = torch.tensor(encoded[i:i+batch_size], dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu().numpy())

    probs = np.concatenate(all_probs)
    preds = np.argmax(probs, axis=-1)
    return preds, probs
