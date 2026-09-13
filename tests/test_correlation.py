"""Testes de autoeda.analysis.correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.analysis.correlation import (
    analyze_correlation,
    compute_vif,
    detect_scale_disparity,
    find_high_correlations,
)
from autoeda.config import AutoEDAConfig


class TestAnalyzeCorrelation:
    def test_high_pearson_correlation_detected(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 200)
        df = pd.DataFrame({"x": x, "y": x * 2 + rng.normal(0, 0.01, 200)})
        result = analyze_correlation(df, AutoEDAConfig())
        assert len(result["high_correlations"]) >= 1

    def test_fewer_than_two_numeric_columns_returns_empty_with_note(self):
        df = pd.DataFrame({"x": range(50), "categoria": (["a", "b"] * 25)})
        result = analyze_correlation(df, AutoEDAConfig())
        assert result["note"] is not None
        assert result["pearson"] == {}

    def test_nonlinear_monotonic_relationship_favors_spearman(self):
        rng = np.random.default_rng(1)
        x = rng.normal(0, 1, 200)
        df = pd.DataFrame({"x": x, "cubo": x**3})
        result = analyze_correlation(df, AutoEDAConfig())
        pearson_xy = result["pearson"]["x"]["cubo"]
        spearman_xy = result["spearman"]["x"]["cubo"]
        assert spearman_xy > pearson_xy


class TestComputeVif:
    def test_multivariate_multicollinearity_detected(self):
        rng = np.random.default_rng(50)
        n = 300
        b = rng.normal(0, 1, n)
        c = rng.normal(0, 1, n)
        a = 2 * b + 3 * c + rng.normal(0, 0.01, n)
        df = pd.DataFrame({"a": a, "b": b, "c": c, "independente": rng.normal(0, 1, n)})

        result = compute_vif(df, ["a", "b", "c", "independente"])
        assert result["a"]["vif"] > 100
        assert result["independente"]["vif"] < 5

    def test_single_numeric_column_returns_empty(self):
        df = pd.DataFrame({"x": range(50)})
        assert compute_vif(df, ["x"]) == {}

    def test_perfect_collinearity_returns_infinite_vif(self):
        b = pd.Series(range(100), dtype=float)
        df = pd.DataFrame({"a": b * 2, "b": b})
        result = compute_vif(df, ["a", "b"])
        assert result["a"]["vif"] == float("inf")


class TestDetectScaleDisparity:
    def test_large_disparity_detected(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"renda": rng.normal(3000, 500, 200), "idade": rng.normal(35, 5, 200)}
        )
        result = detect_scale_disparity(df, ["renda", "idade"], threshold=10.0)
        assert result is not None
        assert result["ratio"] > 10

    def test_similar_scales_returns_none(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(0, 1.2, 200)})
        result = detect_scale_disparity(df, ["x", "y"], threshold=10.0)
        assert result is None


class TestFindHighCorrelations:
    def test_no_duplicate_pairs(self):
        corr_matrix = pd.DataFrame(
            {"a": [1.0, 0.9, 0.1], "b": [0.9, 1.0, 0.1], "c": [0.1, 0.1, 1.0]},
            index=["a", "b", "c"],
        )
        pairs = find_high_correlations(corr_matrix, threshold=0.8, method="pearson")
        assert len(pairs) == 1
