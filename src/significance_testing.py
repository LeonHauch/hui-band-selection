import pickle
from itertools import combinations

import numpy as np
from sklearn.metrics import f1_score


N_BOOTSTRAP = 10000
RANDOM_STATE = 42

BAND_SETS = {
    "all_204_bands": None,
    "lasso_30": np.array([3,4,5,6,7,8,17,18,19,30,31,32,33,34,35,36,37,38,39,80,81,82,105,145,146,148,149,150,151,152]),
    "mrmr_30": np.array([4,5,6,7,8,9,10,11,12,13,14,15,26,38,43,46,48,49,50,51,52,53,54,55,56,57,59,150,151,152]),
    "stg_30": np.array([1,2,3,4,5,6,7,8,9,10,16,17,18,37,38,39,40,41,42,43,73,98,100,101,102,103,104,105,106,148]),
}


def bootstrap_standard_error(y_true, y_pred, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    scores = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores[i] = f1_score(y_true[idx], y_pred[idx], average="macro")
    return scores.mean(), scores.std(ddof=1), np.percentile(scores, [2.5, 97.5])


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
    # two-sided p-value: fraction of bootstrap diffs on the opposite side of zero
    p_value = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
    p_value = min(p_value, 1.0)
    ci = np.percentile(diffs, [2.5, 97.5])
    return observed_diff, p_value, ci


if __name__ == "__main__":
    data = np.load("data/salinas_split.npz")
    X_test, y_test = data["X_test"], data["y_test"]

    predictions = {}
    print("--- Loading models and generating predictions ---")
    for name, bands in BAND_SETS.items():
        with open(f"data/tree_model_{name}.pkl", "rb") as f:
            model = pickle.load(f)
        Xte = X_test if bands is None else X_test[:, bands]
        predictions[name] = model.predict(Xte)
        print(f"  {name}: loaded, predictions generated")

    print(f"\n--- Bootstrap standard error per model ({N_BOOTSTRAP} resamples) ---")
    se_results = {}
    for name, pred in predictions.items():
        mean_f1, se, ci = bootstrap_standard_error(y_test, pred)
        se_results[name] = {"mean_f1": mean_f1, "se": se, "ci_low": ci[0], "ci_high": ci[1]}
        print(f"{name:15s}  macro F1 = {mean_f1:.4f}  SE = {se:.4f}  "
              f"95% CI = [{ci[0]:.4f}, {ci[1]:.4f}]")

    print(f"\n--- Paired bootstrap significance tests (all pairs) ---")
    sig_results = {}
    for name_a, name_b in combinations(predictions.keys(), 2):
        diff, p, ci = paired_bootstrap_test(y_test, predictions[name_a], predictions[name_b])
        key = f"{name_a}_vs_{name_b}"
        sig_results[key] = {"diff": diff, "p_value": p, "ci_low": ci[0], "ci_high": ci[1]}
        sig_marker = "*" if p < 0.05 else " "
        print(f"{name_a:15s} vs {name_b:15s}  diff={diff:+.4f}  "
              f"p={p:.4f} {sig_marker}  95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")

    print("\n(* = significant at alpha=0.05)")

    import json
    with open("data/significance_results.json", "w") as f:
        json.dump({"standard_errors": se_results, "pairwise_tests": sig_results}, f, indent=2)
    print("\nSaved data/significance_results.json")

    for name, pred in predictions.items():
        np.save(f"data/test_predictions_{name}.npy", pred)
    np.save("data/test_labels.npy", y_test)
    print("Saved raw predictions for reuse (e.g. by the CNN significance script)")
