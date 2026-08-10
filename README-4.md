# Hyperspectral Band Selection for Land-Cover Classification

Predict per-pixel land-cover class from spectral signatures, and study
which spectral bands are actually necessary using classical (LASSO/mRMR)
vs. learnable (stochastic gate) feature selection.

Primary dataset: Salinas. Generalization test: Pavia University.

## Setup

```bash
pip install -r requirements.txt
```

Download data (no registration required) into `data/`:
- Salinas_corrected.mat / Salinas_gt.mat
- PaviaU.mat / PaviaU_gt.mat

Source: https://www.ehu.eus/ccwintco/index.php/Hyperspectral_Remote_Sensing_Scenes
(mirrors also exist, e.g. https://github.com/GiorgioMorales/HSI-BandSelection)

## Pipeline

1. `src/data_loading.py` -- load .mat cube + label map, tabularize to
   one row per labeled pixel (drops unlabeled background, class 0).
2. `src/preprocessing.py` -- normalization, stratified train/test split.
3. `src/models.py` -- baseline classifier on all bands.
4. `src/selection.py` -- LASSO/mRMR + learnable stochastic gate.
5. `src/attribution.py` -- SHAP on the trained model.
6. `src/evaluate.py` -- metrics, cross-scene generalization test.

## Status

- [x] Step 1: data loading / tabularization
- [ ] Step 2: preprocessing + split strategy
- [ ] Step 3: baseline model
- [ ] Step 4: feature selection
- [ ] Step 5: selected-feature model comparison
- [ ] Step 6: attribution
- [ ] Step 7: cross-scene generalization (Pavia University)
