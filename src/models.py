import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV


PARAM_GRID = {
    "max_iter": [400],
    "learning_rate": [0.05, 0.1],
    "max_depth": [4, 6],
}


def train_baseline(X_train, y_train, random_state: int = 42) -> HistGradientBoostingClassifier:
    base = HistGradientBoostingClassifier(early_stopping=True, random_state=random_state)
    search = GridSearchCV(base, PARAM_GRID, scoring="f1_macro", cv=3, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_}")
    print(f"Best CV macro F1: {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate(model, X, y, split_name: str) -> dict:
    preds = model.predict(X)
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

    model = train_baseline(X_train, y_train)

    evaluate(model, X_val, y_val, "validation")
    evaluate(model, X_test, y_test, "test")

    with open("data/baseline_best_params.json", "w") as f:
        json.dump(model.get_params(), f, indent=2)
    print("Saved tuned hyperparameters to data/baseline_best_params.json")
