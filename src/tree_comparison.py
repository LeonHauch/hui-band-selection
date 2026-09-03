import json
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score


if __name__ == "__main__":
    data = np.load("data/salinas_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    with open("data/baseline_best_params.json") as f:
        tree_params = json.load(f)

    k_sweep_data = np.load("data/salinas_k_sweep.npz")
    lasso_bands = k_sweep_data["lasso_order"][:30]
    mrmr_bands = k_sweep_data["mrmr_order"][:30]

    stg_data = np.load("data/salinas_stg_selection.npz")
    stg_bands = stg_data["stg_bands"]

    band_sets = {
        "all_204_bands": None,  # None means use all bands
        "lasso_30": lasso_bands,
        "mrmr_30": mrmr_bands,
        "stg_30": stg_bands,
    }

    print("--- Tree comparison across band selection methods ---")
    results = {}
    for name, bands in band_sets.items():
        Xtr = X_train if bands is None else X_train[:, bands]
        Xva = X_val if bands is None else X_val[:, bands]
        Xte = X_test if bands is None else X_test[:, bands]

        model = HistGradientBoostingClassifier(**tree_params)
        model.fit(Xtr, y_train)

        val_preds = model.predict(Xva)
        test_preds = model.predict(Xte)

        val_f1 = f1_score(y_val, val_preds, average="macro")
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average="macro")

        print(f"{name:15s}  val macro F1={val_f1:.4f}  "
              f"test acc={test_acc:.4f}  test macro F1={test_f1:.4f}")

        results[name] = {
            "val_macro_f1": val_f1,
            "test_accuracy": test_acc,
            "test_macro_f1": test_f1,
        }

        with open(f"data/tree_model_{name}.pkl", "wb") as f:
            pickle.dump(model, f)

    with open("data/tree_comparison_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved data/tree_comparison_results.json")
    print("Saved data/tree_model_<name>.pkl for each band set")
