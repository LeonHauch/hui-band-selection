import pickle

import numpy as np
import shap


N_SAMPLE = 500
TOP_K = 30


if __name__ == "__main__":
    data = np.load("data/salinas_split.npz")
    X_test = data["X_test"]

    with open("data/tree_model_all_204_bands.pkl", "rb") as f:
        model = pickle.load(f)

    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=N_SAMPLE, replace=False)
    X_sample = X_test[sample_idx]

    print(f"Running SHAP on a sample of {N_SAMPLE} test rows...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values shape for multiclass: (n_samples, n_features, n_classes)
    # average absolute importance across samples AND classes
    mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))

    shap_top_bands = np.argsort(mean_abs_shap)[::-1][:TOP_K]
    print(f"\nTop {TOP_K} bands by mean |SHAP value|: {sorted(shap_top_bands.tolist())}")

    k_sweep_data = np.load("data/salinas_k_sweep.npz")
    lasso_bands = k_sweep_data["lasso_order"][:30]
    mrmr_bands = k_sweep_data["mrmr_order"][:30]
    stg_data = np.load("data/salinas_stg_selection.npz")
    stg_bands = stg_data["stg_bands"]

    def jaccard(a, b):
        a, b = set(a.tolist()), set(b.tolist())
        return len(a & b) / len(a | b)

    print(f"\nJaccard(SHAP, LASSO) = {jaccard(shap_top_bands, lasso_bands):.3f}")
    print(f"Jaccard(SHAP, mRMR)  = {jaccard(shap_top_bands, mrmr_bands):.3f}")
    print(f"Jaccard(SHAP, STG)   = {jaccard(shap_top_bands, stg_bands):.3f}")

    np.savez(
        "data/salinas_shap.npz",
        mean_abs_shap=mean_abs_shap,
        shap_top_bands=shap_top_bands,
    )
    print("\nSaved data/salinas_shap.npz")
