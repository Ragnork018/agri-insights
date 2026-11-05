# Agri Insights — field performance modelling workbench

A reproducible classification pipeline for agricultural field-performance datasets. It validates tabular data, handles numeric/categorical missing values inside the training pipeline, uses a stratified holdout, supports balanced RandomForest, ExtraTrees, Logistic Regression and optional XGBoost, and exports metrics and a fitted model.

## Dataset caveat

The resume does not name the original dataset, its target column, class meanings, location/time coverage, or label-generation process. This repository does not claim to predict crop yield or recommend agronomic actions. Supply the actual dataset and data dictionary before interpreting model outputs; see [DOCS/DATASETS.md](docs/DATASETS.md).

## Run

```bash
pip install -e '.[dev,dashboard]'
python -m agri-insights.train --csv /path/to/fields.csv --target performance_class --out artifacts
streamlit run dashboard.py
pytest
```

The pipeline loops preprocessing only on training folds and uses a stratified holdout. Do not use columns generated after the outcome or derived from the label.
