import pandas as pd
import pytest

from agri_insights.field_model import train_field_model


def sample_frame():
    rows = []
    for i in range(180):
        rows.append({
            "soil_ph": 5.5 + (i % 7) * 0.2,
            "rain_mm": 100 + (i % 13) * 9,
            "crop": ["rice", "maize", "wheat"][i % 3],
            "season": ["kharif", "rabi"][i % 2],
            "performance_class": ["low", "medium", "high"][i % 3],
            "final_yield": 900 + i,
            "constant_column": 1,
        })
    return pd.DataFrame(rows)


def test_training_reports_balanced_metrics_and_drops_leakage():
    result = train_field_model(sample_frame(), "performance_class")
    assert 0 <= result.metrics["balanced_accuracy"] <= 1
    assert 0 <= result.metrics["macro_f1"] <= 1
    assert "final_yield" in result.dropped_columns
    assert "constant_column" in result.dropped_columns
    assert "performance_class" not in result.feature_columns


def test_target_with_missing_labels_is_rejected():
    frame = sample_frame()
    frame.loc[0, "performance_class"] = None
    with pytest.raises(ValueError, match="missing"):
        train_field_model(frame, "performance_class")


def test_missing_target_is_rejected():
    with pytest.raises(ValueError, match="missing"):
        train_field_model(sample_frame(), "not_a_label")
