import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
):
    feature_cols = [c for c in df.columns if c.startswith("band_")]
    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size, stratify=y_trainval,
        random_state=random_state,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "feature_cols": feature_cols,
    }


def summarize_split(split: dict) -> None:
    print(f"Train: {split['X_train'].shape[0]} rows, "
          f"Val: {split['X_val'].shape[0]} rows, "
          f"Test: {split['X_test'].shape[0]} rows")
    print(f"Features: {split['X_train'].shape[1]}")

    train_counts = pd.Series(split["y_train"]).value_counts().sort_index()
    print("\nSmallest train classes:")
    print(train_counts.sort_values().head(5))

    print(f"\nTrain mean: {split['X_train'].mean():.4f}")
    print(f"Train std: {split['X_train'].std():.4f}")
    print(f"Test mean: {split['X_test'].mean():.4f}")


if __name__ == "__main__":
    df = pd.read_parquet("data/salinas_tabular.parquet")
    split = split_and_scale(df)
    summarize_split(split)

    np.savez(
        "data/salinas_split.npz",
        X_train=split["X_train"], y_train=split["y_train"],
        X_val=split["X_val"], y_val=split["y_val"],
        X_test=split["X_test"], y_test=split["y_test"],
    )
    print("\nSaved data/salinas_split.npz")
