import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, f1_score, accuracy_score


KERNEL_GRID = [3, 5, 7]
FILTER_GRID = [16, 32]
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 10


class SpectralCNN(nn.Module):
    # Matches the architecture described in Hu et al. (2015), "Deep
    # Convolutional Neural Networks for Hyperspectral Image Classification":
    # one convolutional layer, one max-pooling layer, one fully-connected
    # (hidden) layer, one output layer.
    def __init__(self, n_classes, n_bands, kernel_size, filters, hidden_dim=64):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(1, filters, kernel_size, padding=pad)
        self.pool = nn.MaxPool1d(2)
        pooled_len = n_bands // 2
        self.fc1 = nn.Linear(filters * pooled_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)              # (batch, 1, n_bands)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def train_cnn(X_train, y_train, X_val, y_val, n_classes, kernel_size, filters,
              max_epochs=MAX_EPOCHS, patience=PATIENCE, random_state=42):
    torch.manual_seed(random_state)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
    )

    model = SpectralCNN(n_classes, X_train.shape[1], kernel_size, filters)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_preds = val_logits.argmax(dim=1).numpy()
        val_f1 = f1_score(y_val, val_preds, average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        if epoch % 10 == 0:
            print(f"    epoch {epoch:3d}  val macro F1={val_f1:.4f}  best={best_val_f1:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_f1, epoch + 1


def evaluate_cnn(model, X, y, split_name: str) -> dict:
    X_t = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds = model(X_t).argmax(dim=1).numpy()

    acc = accuracy_score(y, preds)
    macro_f1 = f1_score(y, preds, average="macro")
    weighted_f1 = f1_score(y, preds, average="weighted")

    print(f"--- {split_name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print()
    print(classification_report(y, preds, digits=3))

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1}


if __name__ == "__main__":
    data = np.load("data/salinas_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    y_train_0 = y_train - 1
    y_val_0 = y_val - 1
    y_test_0 = y_test - 1
    n_classes = int(y_train_0.max()) + 1

    print("--- Architecture search (kernel_size x filters) ---")
    results = []
    for kernel_size in KERNEL_GRID:
        for filters in FILTER_GRID:
            _, val_f1, n_epochs = train_cnn(
                X_train, y_train_0, X_val, y_val_0, n_classes,
                kernel_size=kernel_size, filters=filters,
            )
            print(f"kernel_size={kernel_size}  filters={filters:3d}  "
                  f"val macro F1={val_f1:.4f}  stopped at epoch {n_epochs}")
            results.append((kernel_size, filters, val_f1))

    best_kernel, best_filters, _ = max(results, key=lambda r: r[2])
    print(f"\nChosen: kernel_size={best_kernel}, filters={best_filters}")

    with open("data/cnn_architecture.json", "w") as f:
        json.dump({"kernel_size": best_kernel, "filters": best_filters}, f)

    print("\n--- Final evaluation with chosen architecture ---")
    model, val_f1, n_epochs = train_cnn(
        X_train, y_train_0, X_val, y_val_0, n_classes,
        kernel_size=best_kernel, filters=best_filters,
    )
    evaluate_cnn(model, X_val, y_val_0, "validation")
    evaluate_cnn(model, X_test, y_test_0, "test")

    print("Saved data/cnn_architecture.json")
