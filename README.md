# Agri Insights — field performance modelling workbench

A reproducible classification pipeline for agricultural field-performance datasets. Complete source is in [agri-insights-source.zip](./agri-insights-source.zip).

## Run

```bash
pip install -e '.[dev,dashboard]'
python -m agri_insights.train --csv /path/to/fields.csv --target performance_class --out artifacts
streamlit run dashboard.py
pytest
```

## Dataset requirement
The resume does not identify the original dataset, target column, class meanings, geography/time coverage, or label-generation process. This repository does not claim to predict crop yield or issue agronomic advice. Supply the original dataset and data dictionary before interpreting outputs. See docs/DATASETS.md in the source archive.

The pipeline validates tabular data, preprocesses numeric/categorical columns within training folds, uses stratified holdout evaluation, and compares supported classifiers. Metrics are meaningful only for the supplied dataset and split.
