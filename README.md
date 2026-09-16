# Hyperspectral Band Selection for Land-Cover Classification

Comparing classical (LASSO, mRMR) and learnable (Stochastic Gates) feature
selection methods for per-pixel land-cover classification on the Salinas
hyperspectral scene. Full write-up: see `paper.pdf`.

## Task

Predict the land-cover class of each pixel (16 classes) from its 204-band
spectral signature, and determine how many and which bands are actually
necessary, rather than assuming all 204 are needed.

## Dataset

**Salinas scene** (AVIRIS sensor, agricultural valley, California), 204
spectral bands, 16 land-cover classes, 54,129 labeled pixels. Public,
no registration required:
https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes

Download `Salinas_corrected.mat` and `Salinas_gt.mat` into `data/` (not
included in this repo, see `.gitignore`).

## Setup

This project uses two separate virtual environments, due to a NumPy
version conflict between `shap`/`scipy` (need NumPy ≥2) and `torch==2.2.2`
on Intel macOS (needs NumPy <2, and is the last version with Intel Mac
wheels published).

```bash
# Main environment: data pipeline, tree models, LASSO/mRMR, SHAP
python3 -m venv .venv312
source .venv312/bin/activate
pip install -r requirements.txt

# Separate environment: STG and the spectral CNN (needs torch)
python3 -m venv .venv-stg
source .venv-stg/bin/activate
pip install "numpy<2" scikit-learn torch==2.2.2 captum
```

## Pipeline

Run in this order:

| Step | Script | Environment |
|---|---|---|
| 1 | `src/data_loading.py` | `.venv312` |
| 2 | `src/preprocessing.py` | `.venv312` |
| 3 | `src/models.py` (baseline tree, tuned) | `.venv312` |
| 4 | `src/selection.py` (LASSO + mRMR) | `.venv312` |
| 5 | `src/k_sweep.py` (justify k=30) | `.venv312` |
| 6 | `src/stg_selection.py` (Stochastic Gates) | `.venv-stg` |
| 7 | `src/tree_comparison.py` (4-way comparison) | `.venv312` |
| 8 | `src/cnn_model.py` (spectral 1D-CNN) | `.venv-stg` |
| 9 | `src/cnn_save_weights.py` | `.venv-stg` |
| 10 | `src/shap_analysis.py` | `.venv312` |
| 11 | `src/significance_testing.py` (tree comparisons) | `.venv312` |
| 12 | `src/cnn_significance.py` (tree vs. CNN) | `.venv-stg` |

## Key results

At k=30 selected bands (14.7% of the original 204), the tuned tree
classifier retains:

| Band set | Test macro F1 |
|---|---|
| All 204 bands | 0.980 |
| LASSO-30 | 0.975 |
| STG-30 | 0.974 |
| mRMR-30 | 0.964 |

LASSO-30 and STG-30 are not statistically different (paired bootstrap
test, p=0.298); mRMR is significantly worse than both (p<0.001). The
spectral 1D-CNN (all bands) is statistically indistinguishable from the
tree (p=0.266).

## Report

Full methodology, citations, and discussion: `paper.pdf` (LaTeX source
in `paper.tex`, ACL format).

## Limitations

Spectral-only, per-pixel framing (no spatial context); single-dataset
(Salinas only, Pavia University generalization test not completed); CNN
evaluated on full band set only. See the report's Limitations section
for details.
