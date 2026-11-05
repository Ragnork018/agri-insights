# Agri Insights

A reproducible workbench for agricultural field-performance classification, with a separate district-level crop-yield research path. It is designed around data provenance, leakage control, transparent validation and repeatable model artefacts—not an arbitrary accuracy claim.

## Start here

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m agri_insights.train --csv /path/to/your_data.csv --target performance_class --out artifacts
pytest -q
```

A CSV must contain the target column and at least two target classes. Review the dataset card before training. The original resume does not specify the original field dataset, target definition, label-generation process, geography or measurement units, so this rebuild does not silently claim to reproduce those labels.

## What is included

- Numeric and categorical preprocessing fitted only on training data.
- Stratified holdout evaluation and reproducible random seeds.
- Random Forest, Extra Trees and Logistic Regression baselines.
- Optional XGBoost classifier when the extra is installed.
- Accuracy, balanced accuracy, macro-F1, per-class report and confusion matrix.
- Saved model pipeline, metrics and a human-readable model card.

## District crop-yield research

The separate `yield_pipeline.py` is a district/crop/season/year research workflow for official Area, Production and Yield (APY) records. Its observations are administrative aggregates, not plot-level measurements. It must not be presented as a field-sensor model or as agronomic advice. See `docs/DATASETS.md`.

## Reproducibility

Every evaluation records the input row count, held-out row count, feature list, target classes, random seed and metrics. The benchmark is valid only for the supplied dataset and its sampling process. It does not prove performance in a new district, season or farm.
