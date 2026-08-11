import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from captum.module import GaussianStochasticGates
from sklearn.metrics import f1_score


ARCH_GRID = [32, 64, 128]       # hidden_dim candidates, searched below
LEARNING_RATE = 1e-3            # Adam paper default (Kingma & Ba, 2015), not swept
BATCH_SIZE = 256                # conventional default, minor effect on final accuracy at this scale, not swept
MAX_EPOCHS = 200
PATIENCE = 10                   # early stopping patience on validation macro F1
REG_WEIGHT_GRID = [0.003, 0.004, 0.005, 0.006, 0.008, 0.01]
TARGET_K = 30


class STGClassifier(nn.Module):
    def __init__(self, n_features, n_classes, hidden_dim, reg_weight):
        super().__init__()
        self.gates = GaussianStochasticGates(n_features, reg_weight=reg_weight, std=0.5)
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        gated_x, reg = self.gates(x)
        logits = self.net(gated_x)
        return logits, reg


def train_stg(X_train, y_train, X_val, y_val, n_classes, hidden_dim, reg_weight,
              max_epochs=MAX_EPOCHS, patience=PATIENCE, random_state=42):
    torch.manual_seed(random_state)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=BATCH_SIZE, shuffle=True
    )

    model = STGClassifier(X_train.shape[1], n_classes, hidden_dim, reg_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    best_val_f1 = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits, reg = model(xb)
            loss = criterion(logits, yb) + reg
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_logits, _ = model(X_val_t)
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

    model.load_state_dict(best_state)
    gate_values = model.gates.get_gate_values(clamp=True).detach().numpy()
    n_active = int((gate_values > 0.5).sum())

    return model, best_val_f1, gate_values, n_active, epoch + 1


if __name__ == "__main__":
    data = np.load("data/salinas_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    y_train_0 = y_train - 1
    y_val_0 = y_val - 1
    n_classes = int(y_train_0.max()) + 1

    print("--- Stage A: architecture search (reg_weight=0, no sparsity pressure) ---")
    arch_results = []
    for hidden_dim in ARCH_GRID:
        _, val_f1, _, _, n_epochs = train_stg(
            X_train, y_train_0, X_val, y_val_0, n_classes,
            hidden_dim=hidden_dim, reg_weight=0.0,
        )
        print(f"hidden_dim={hidden_dim:4d}  val macro F1={val_f1:.4f}  stopped at epoch {n_epochs}")
        arch_results.append((hidden_dim, val_f1))

    best_hidden_dim = max(arch_results, key=lambda r: r[1])[0]
    print(f"\nChosen hidden_dim: {best_hidden_dim}")
    with open("data/stg_architecture.json", "w") as f:
        json.dump({"hidden_dim": best_hidden_dim}, f)

    print("\n--- Stage B: lambda (reg_weight) sweep, fixed architecture ---")
    reg_weights, n_actives, val_f1s = [], [], []
    all_gate_values = []
    for reg_weight in REG_WEIGHT_GRID:
        _, val_f1, gate_values, n_active, n_epochs = train_stg(
            X_train, y_train_0, X_val, y_val_0, n_classes,
            hidden_dim=best_hidden_dim, reg_weight=reg_weight,
        )
        print(f"reg_weight={reg_weight:<8} active_gates={n_active:3d}  "
              f"val macro F1={val_f1:.4f}  stopped at epoch {n_epochs}")
        reg_weights.append(reg_weight)
        n_actives.append(n_active)
        val_f1s.append(val_f1)
        all_gate_values.append(gate_values)

    np.savez(
        "data/salinas_stg_sweep.npz",
        reg_weights=reg_weights,
        n_active=n_actives,
        val_f1=val_f1s,
        gate_values=np.array(all_gate_values),
    )
    print("\nSaved data/salinas_stg_sweep.npz")

    closest_idx = int(np.argmin(np.abs(np.array(n_actives) - TARGET_K)))
    chosen_reg_weight = reg_weights[closest_idx]
    chosen_gate_values = all_gate_values[closest_idx]
    stg_bands = np.argsort(chosen_gate_values)[::-1][:TARGET_K]

    print(f"\nClosest to target k={TARGET_K}: reg_weight={chosen_reg_weight} "
          f"(n_active={n_actives[closest_idx]}, val macro F1={val_f1s[closest_idx]:.4f})")
    print(f"STG selected bands: {sorted(stg_bands.tolist())}")

    np.savez("data/salinas_stg_selection.npz", stg_bands=stg_bands, reg_weight=chosen_reg_weight)
    print("Saved data/salinas_stg_selection.npz")
