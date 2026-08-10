"""
Step 1: load a hyperspectral scene (.mat cube + .mat ground-truth label
map) and tabularize it into one row per labeled pixel.

Verified against the real file (not assumed shapes):
    Salinas: X (512, 217, 204) uint16, y (512, 217) uint8
Label 0 means "unlabeled background" -- these pixels have no ground-truth
class and must be dropped before training/evaluation.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.io as sio


# .mat variable names differ per scene -- confirmed by inspecting the
# actual files, not assumed.
# NOTE: Pavia University is intentionally not configured yet. It's reserved
# for the cross-scene generalization test at the end of the project and
# should not be touched before then.
SCENE_CONFIG = {
    "salinas": {
        "data_file": "Salinas_corrected.mat",
        "data_key": "salinas_corrected",
        "gt_file": "Salinas_gt.mat",
        "gt_key": "salinas_gt",
    },
}


@dataclass
class SceneData:
    name: str
    cube: np.ndarray          # (H, W, n_bands) raw radiance values
    label_map: np.ndarray      # (H, W) integer labels, 0 = unlabeled
    n_bands: int
    class_ids: np.ndarray        # sorted array of real class ids (excludes 0)


def load_scene(name: str, data_dir: str = "data") -> SceneData:
    """Load one scene's cube + label map from the configured .mat files."""
    if name not in SCENE_CONFIG:
        raise ValueError(f"Unknown scene '{name}'. Known scenes: {list(SCENE_CONFIG)}")
    cfg = SCENE_CONFIG[name]

    cube = sio.loadmat(f"{data_dir}/{cfg['data_file']}")[cfg["data_key"]]
    label_map = sio.loadmat(f"{data_dir}/{cfg['gt_file']}")[cfg["gt_key"]]

    if cube.shape[:2] != label_map.shape:
        raise ValueError(
            f"Cube spatial shape {cube.shape[:2]} does not match label map "
            f"shape {label_map.shape} for scene '{name}' -- check the files."
        )

    class_ids = np.unique(label_map)
    class_ids = class_ids[class_ids != 0]

    return SceneData(
        name=name,
        cube=cube,
        label_map=label_map,
        n_bands=cube.shape[2],
        class_ids=class_ids,
    )


def tabularize(scene: SceneData) -> pd.DataFrame:
    """
    Turn a (H, W, n_bands) cube + (H, W) label map into a flat DataFrame:
    one row per labeled pixel (background/class-0 pixels dropped), columns
    band_0 .. band_{n_bands-1} plus a 'label' column.
    """
    H, W, n_bands = scene.cube.shape
    flat_cube = scene.cube.reshape(H * W, n_bands)
    flat_labels = scene.label_map.reshape(H * W)

    mask = flat_labels != 0
    X = flat_cube[mask]
    y = flat_labels[mask]

    df = pd.DataFrame(X, columns=[f"band_{i}" for i in range(n_bands)])
    df["label"] = y
    return df


def summarize(df: pd.DataFrame, name: str) -> None:
    n_bands = df.shape[1] - 1
    counts = df["label"].value_counts().sort_index()
    print(f"--- {name} ---")
    print(f"Rows (labeled pixels): {len(df)}")
    print(f"Feature columns (bands): {n_bands}")
    print(f"Classes: {df['label'].nunique()}")
    print(f"Samples-per-feature ratio: {len(df) / n_bands:.1f}:1")
    print(f"Smallest class: {counts.min()}, largest class: {counts.max()}")
    print()


if __name__ == "__main__":
    scene_name = "salinas"
    scene = load_scene(scene_name, data_dir="data")
    df = tabularize(scene)
    summarize(df, scene_name)
    df.to_parquet(f"data/{scene_name}_tabular.parquet")
    print(f"Saved data/{scene_name}_tabular.parquet")
