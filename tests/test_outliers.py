"""Testes de autoeda.analysis.outliers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.analysis.outliers import analyze_outliers, detect_outliers_iqr, detect_outliers_zscore
from autoeda.config import AutoEDAConfig


class TestDetectOutliersIqr:
    def test_known_outliers_detected(self):
        rng = np.random.default_rng(42)
        normal_data = rng.normal(50, 5, 100)
        series = pd.Series(list(normal_data) + [200, -100, 210])
        result = detect_outliers_iqr(series, multiplier=1.5)
        assert result["count"] >= 3

    def test_zero_iqr_returns_no_outliers_with_note(self):
        series = pd.Series([5, 5, 5, 5, 5, 5, 5, 6, 5, 5])
        result = detect_outliers_iqr(series, multiplier=1.5)
        assert result["count"] == 0
        assert result["note"] is not None

    def test_insufficient_observations(self):
        series = pd.Series([1.0, 2.0, None, None])
        result = detect_outliers_iqr(series, multiplier=1.5)
        assert result["count"] == 0
        assert "insuficientes" in result["note"]


class TestDetectOutliersZscore:
    def test_known_outliers_detected(self):
        rng = np.random.default_rng(42)
        normal_data = rng.normal(50, 5, 100)
        series = pd.Series(list(normal_data) + [200, -100, 210])
        result = detect_outliers_zscore(series, threshold=3.0)
        assert result["count"] >= 3

    def test_zero_std_returns_no_outliers_with_note(self):
        series = pd.Series([3.0] * 20)
        result = detect_outliers_zscore(series, threshold=3.0)
        assert result["count"] == 0
        assert result["note"] is not None


class TestAnalyzeOutliers:
    def test_only_numeric_columns_analyzed(self):
        df = pd.DataFrame(
            {
                "numerica": np.random.default_rng(0).normal(0, 1, 100),
                "categorica_numerica": ([1, 2, 3, 4, 5] * 20),  # baixa cardinalidade -> nao numeric
                "id": range(100),
                "texto": (["a", "b"] * 50),
            }
        )
        result = analyze_outliers(df, AutoEDAConfig())
        assert "numerica" in result["columns"]
        assert "id" not in result["columns"]
        assert "categorica_numerica" not in result["columns"]

    def test_method_switch_to_zscore(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200)})
        result = analyze_outliers(df, AutoEDAConfig(outlier_method="zscore"))
        assert result["method"] == "zscore"
        assert result["columns"]["x"]["method"] == "zscore"
