import json

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from selection import lasso_rank, mrmr_select


K_VALUES = [5, 10, 20, 30, 50, 75, 100, 150]


if __name__ == "__main__":
    data = np.load("data/salinas_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    with open("data/baseline_best_params.json") as f:
        tree_params = json.load(f)
    with open("data/lasso_alpha.json") as f:
        lasso_alpha = json.load(f)["alpha"]

    tuned_model = HistGradientBoostingClassifier(**tree_params)

    max_k = max(K_VALUES)
    print("Computing full LASSO ranking (fixed alpha, no re-tuning)...")
    lasso_order = lasso_rank(X_train, y_train, alpha=lasso_alpha, k=max_k)
    print("Computing full mRMR ranking...")
    mrmr_order = mrmr_select(X_train, y_train, k=max_k)

    print("\n--- Sweep results ---")
    rows = []
    for k in K_VALUES:
        lasso_bands = lasso_order[:k]
        mrmr_bands = mrmr_order[:k]

        lasso_model = clone(tuned_model)
        lasso_model.fit(X_train[:, lasso_bands], y_train)
        lasso_f1 = f1_score(
            y_val, lasso_model.predict(X_val[:, lasso_bands]), average="macro"
        )

        mrmr_model = clone(tuned_model)
        mrmr_model.fit(X_train[:, mrmr_bands], y_train)
        mrmr_f1 = f1_score(
            y_val, mrmr_model.predict(X_val[:, mrmr_bands]), average="macro"
        )

        print(f"k={k:3d}  LASSO macro F1={lasso_f1:.4f}  mRMR macro F1={mrmr_f1:.4f}")
        rows.append((k, lasso_f1, mrmr_f1))

    np.savez(
        "data/salinas_k_sweep.npz",
        k=[r[0] for r in rows],
        lasso_f1=[r[1] for r in rows],
        mrmr_f1=[r[2] for r in rows],
        lasso_order=lasso_order,
        mrmr_order=mrmr_order,
    )
    print("\nSaved data/salinas_k_sweep.npz")
