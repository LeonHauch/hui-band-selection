import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score


N_BOOTSTRAP = 10000
RANDOM_STATE = 42


class SpectralCNN(nn.Module):
    # Matches cnn_model.py's architecture: Hu et al. (2015) style --
    # one conv layer, one max-pool layer, one FC hidden layer, output layer.
    def __init__(self, n_classes, n_bands, kernel_size, filters, hidden_dim=64):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(1, filters, kernel_size, padding=pad)
        self.pool = nn.MaxPool1d(2)
        pooled_len = n_bands // 2
        self.fc1 = nn.Linear(filters * pooled_len, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def paired_bootstrap_test(y_true, pred_a, pred_b, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f1_a = f1_score(y_true[idx], pred_a[idx], average="macro")
        f1_b = f1_score(y_true[idx], pred_b[idx], average="macro")
        diffs[i] = f1_a - f1_b
    observed_diff = diffs.mean()
    p_value = min(2 * min((diffs > 0).mean(), (diffs < 0).mean()), 1.0)
    ci = np.percentile(diffs, [2.5, 97.5])
    return observed_diff, p_value, ci


if __name__ == "__main__":
    data = np.load("data/salinas_split.npz")
    X_test = data["X_test"]
    y_test_1indexed = data["y_test"]
    y_test = y_test_1indexed - 1  # CNN was trained on 0-indexed labels

    with open("data/cnn_architecture.json") as f:
        arch = json.load(f)

    n_classes = int(y_test.max()) + 1
    model = SpectralCNN(n_classes, X_test.shape[1], arch["kernel_size"], arch["filters"])

    weights_path = "data/cnn_model_all_bands.pt"
    if os.path.exists(weights_path):
        print(f"Loading saved CNN weights from {weights_path}")
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        raise FileNotFoundError(
            f"{weights_path} not found. This script does not retrain the CNN "
            "automatically -- if you don't have saved weights, rerun cnn_model.py "
            "once (final architecture only, not the search) to produce them, "
            "then rerun this script."
        )

    model.eval()
    with torch.no_grad():
        cnn_preds_0indexed = model(torch.tensor(X_test, dtype=torch.float32)).argmax(dim=1).numpy()
    cnn_preds = cnn_preds_0indexed + 1  # back to original 1..16 label space

    tree_preds = np.load("data/test_predictions_all_204_bands.npy")
    tree_labels = np.load("data/test_labels.npy")
    assert np.array_equal(tree_labels, y_test_1indexed), "test set mismatch between tree and CNN runs"

    cnn_f1 = f1_score(y_test_1indexed, cnn_preds, average="macro")
    tree_f1 = f1_score(y_test_1indexed, tree_preds, average="macro")
    print(f"Tree (all bands) test macro F1: {tree_f1:.4f}")
    print(f"CNN (all bands) test macro F1: {cnn_f1:.4f}")

    print(f"\nRunning paired bootstrap test ({N_BOOTSTRAP} resamples)...")
    diff, p, ci = paired_bootstrap_test(y_test_1indexed, tree_preds, cnn_preds)
    sig_marker = "*" if p < 0.05 else " "
    print(f"Tree vs CNN  diff={diff:+.4f}  p={p:.4f} {sig_marker}  95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")
    print("(* = significant at alpha=0.05)")

    with open("data/cnn_significance_results.json", "w") as f:
        json.dump({"tree_f1": tree_f1, "cnn_f1": cnn_f1, "diff": diff, "p_value": p,
                   "ci_low": ci[0], "ci_high": ci[1]}, f, indent=2)
    print("\nSaved data/cnn_significance_results.json")
