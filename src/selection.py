import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV


LASSO_ALPHA_GRID = {"alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]}


def lasso_tune_alpha(X_train, y_train, random_state: int = 42) -> float:
    base = SGDClassifier(
        loss="log_loss",
        penalty="l1",
        early_stopping=True,
        n_iter_no_change=5,
        max_iter=1000,
        random_state=random_state,
    )
    search = GridSearchCV(base, LASSO_ALPHA_GRID, scoring="f1_macro", cv=3, n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"LASSO best alpha: {search.best_params_['alpha']}")
    print(f"LASSO best CV macro F1: {search.best_score_:.4f}")
    return search.best_params_["alpha"]


def lasso_rank(X_train, y_train, alpha: float, k: int, random_state: int = 42) -> np.ndarray:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l1",
        alpha=alpha,
        early_stopping=True,
        n_iter_no_change=5,
        max_iter=1000,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    importance = np.linalg.norm(model.coef_, axis=0)
    return np.argsort(importance)[::-1][:k]


def mrmr_select(X_train, y_train, k: int = 30) -> np.ndarray:
    relevance = mutual_info_classif(X_train, y_train, random_state=42)
    corr = np.abs(np.corrcoef(X_train, rowvar=False))
    np.fill_diagonal(corr, 0.0)

    selected = [int(np.argmax(relevance))]
    remaining = set(range(X_train.shape[1])) - set(selected)

    while len(selected) < k:
        best_score, best_feat = -np.inf, None
        for feat in remaining:
            redundancy = corr[feat, selected].mean()
            score = relevance[feat] - redundancy
            if score > best_score:
                best_score, best_feat = score, feat
        selected.append(best_feat)
        remaining.remove(best_feat)

    return np.array(selected)


def jaccard(set_a: np.ndarray, set_b: np.ndarray) -> float:
    a, b = set(set_a.tolist()), set(set_b.tolist())
    return len(a & b) / len(a | b)


if __name__ == "__main__":
    import json

    data = np.load("data/salinas_split.npz")
    X_train, y_train = data["X_train"], data["y_train"]

    print("Tuning LASSO alpha...")
    alpha = lasso_tune_alpha(X_train, y_train)
    lasso_bands = lasso_rank(X_train, y_train, alpha=alpha, k=30)
    print(f"LASSO selected bands: {sorted(lasso_bands.tolist())}")

    print("\nRunning mRMR selection...")
    mrmr_bands = mrmr_select(X_train, y_train, k=30)
    print(f"mRMR selected bands: {sorted(mrmr_bands.tolist())}")

    overlap = jaccard(lasso_bands, mrmr_bands)
    print(f"\nJaccard overlap LASSO vs mRMR: {overlap:.3f}")

    np.savez("data/salinas_selection.npz", lasso_bands=lasso_bands, mrmr_bands=mrmr_bands)
    with open("data/lasso_alpha.json", "w") as f:
        json.dump({"alpha": alpha}, f)
    print("Saved data/salinas_selection.npz and data/lasso_alpha.json")
