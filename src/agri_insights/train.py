"""Command-line trainer for the generic field-performance dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .field_model import train_field_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the documented CSV dataset.")
    parser.add_argument("--target", required=True, help="Name of the label column.")
    parser.add_argument("--algorithm", choices=["random_forest", "extra_trees", "logistic", "xgboost"], default="extra_trees")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    frame = pd.read_csv(args.csv)
    result = train_field_model(frame, args.target, args.algorithm, seed=args.seed)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(result.model, output / "field_performance_pipeline.joblib")
    (output / "metrics.json").write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
    (output / "classification_report.json").write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    (output / "model_card.json").write_text(json.dumps({
        "target": args.target,
        "features": result.feature_columns,
        "dropped_columns": result.dropped_columns,
        "metrics": result.metrics,
        "interpretation": "Research baseline only; validate on future seasons and held-out locations before operational use.",
    }, indent=2), encoding="utf-8")
    print(json.dumps(result.metrics, indent=2))


if __name__ == "__main__":
    main()
