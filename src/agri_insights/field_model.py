"""Leakage-aware tabular classification for documented field-performance labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# These fields commonly encode the outcome or become known only after the event.
# The list is intentionally conservative; inspect every project dataset before training.
_POST_OUTCOME_TERMS = (
    "final_yield", "harvested_yield", "actual_production", "final_grade",
    "outcome", "label_source", "post_harvest", "after_harvest",
    "inspection_result", "decision_after",
)


@dataclass
class FieldModelResult:
    metrics: dict[str, Any]
    report: dict[str, Any]
    confusion: list[list[int]]
    labels: list[str]
    feature_columns: list[str]
    dropped_columns: list[str]
    model: Pipeline


def _validate_frame(frame: pd.DataFrame, target: str) -> None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Upload a non-empty pandas DataFrame.")
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is missing.")
    if frame[target].isna().any():
        raise ValueError("Target contains missing labels. Resolve these before training.")
    if frame[target].nunique() < 2:
        raise ValueError("The target must contain at least two classes.")
    if frame[target].value_counts().min() < 2:
        raise ValueError("Every target class needs at least two rows for a stratified holdout.")


def _select_features(frame: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str]]:
    features = frame.drop(columns=[target]).copy()
    dropped: list[str] = []
    for column in list(features.columns):
        normalized = str(column).strip().lower().replace(" ", "_")
        if any(term in normalized for term in _POST_OUTCOME_TERMS):
            features = features.drop(columns=[column])
            dropped.append(str(column))
        elif features[column].nunique(dropna=False) <= 1:
            features = features.drop(columns=[column])
            dropped.append(str(column))
    if features.shape[1] == 0:
        raise ValueError("No usable features remain after removing constant/post-outcome fields.")
    return features, dropped


def build_classifier(features: pd.DataFrame, algorithm: str = "extra_trees", seed: int = 42) -> Pipeline:
    numeric = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    transformers = []
    if numeric:
        numeric_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric))
    if categorical:
        categorical_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")

    if algorithm == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=seed, n_jobs=-1,
        )
    elif algorithm == "extra_trees":
        estimator = ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        )
    elif algorithm == "logistic":
        estimator = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed,
        )
    elif algorithm == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ValueError('Install the optional dependency with: pip install -e ".[xgboost]"') from exc
        estimator = XGBClassifier(
            n_estimators=450, max_depth=5, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=2,
            eval_metric="mlogloss", random_state=seed, n_jobs=-1,
        )
    else:
        raise ValueError("algorithm must be random_forest, extra_trees, logistic or xgboost")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def train_field_model(
    frame: pd.DataFrame,
    target: str,
    algorithm: str = "extra_trees",
    test_size: float = 0.2,
    seed: int = 42,
) -> FieldModelResult:
    _validate_frame(frame, target)
    if not 0.1 <= test_size <= 0.4:
        raise ValueError("test_size must be between 0.1 and 0.4.")

    features, dropped = _select_features(frame, target)
    labels = frame[target].astype(str)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels,
    )
    model = build_classifier(x_train, algorithm=algorithm, seed=seed)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    classes = sorted(labels.unique().tolist())
    metrics = {
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "feature_count": int(features.shape[1]),
        "algorithm": algorithm,
        "seed": int(seed),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "macro_f1": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "warning": "Holdout metrics describe only this uploaded dataset and split.",
    }
    report = classification_report(y_test, prediction, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, prediction, labels=classes)
    return FieldModelResult(
        metrics=metrics, report=report, confusion=matrix.tolist(), labels=classes,
        feature_columns=features.columns.tolist(), dropped_columns=dropped, model=model,
    )
