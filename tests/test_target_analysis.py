"""Testes de autoeda.analysis.target_analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.analysis.target_analysis import (
    analyze_target,
    compute_chi_square_and_cramers_v,
    compute_point_biserial,
)
from autoeda.config import AutoEDAConfig
from autoeda.exceptions import AnalysisError


class TestAnalyzeTargetTechniqueSelection:
    def test_numeric_predictor_uses_point_biserial(self, simple_binary_df, config):
        result = analyze_target(simple_binary_df, "target", config)
        predictor = next(p for p in result["predictors"] if p["predictor"] == "preditor_forte")
        assert predictor["relationship"] == "point_biserial_mutual_info"
        assert predictor["metrics"]["correlation"] is not None
        assert predictor["metrics"]["mutual_information"] is not None

    def test_categorical_nominal_uses_chi_square(self, simple_binary_df, config):
        result = analyze_target(simple_binary_df, "target", config)
        predictor = next(p for p in result["predictors"] if p["predictor"] == "categoria_fraca")
        assert predictor["relationship"] == "chi_square_cramers_v"

    def test_ordinal_like_categorical_uses_spearman(self, config):
        rng = np.random.default_rng(0)
        n = 200
        target = rng.choice(["a", "b"], n)
        df = pd.DataFrame({"target": target, "nota": rng.choice([1, 2, 3, 4, 5], n)})
        result = analyze_target(df, "target", config)
        predictor = next(p for p in result["predictors"] if p["predictor"] == "nota")
        assert predictor["relationship"] == "spearman"

    def test_id_text_datetime_excluded(self, config):
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame(
            {
                "target": rng.choice(["a", "b"], n),
                "id": range(n),
                "data": pd.date_range("2023-01-01", periods=n),
            }
        )
        result = analyze_target(df, "target", config)
        excluded_names = {item["predictor"] for item in result["excluded_predictors"]}
        assert "id" in excluded_names
        assert "data" in excluded_names


class TestPossibleLeakage:
    def test_near_perfect_association_flagged_as_leakage(self, config):
        rng = np.random.default_rng(0)
        n = 300
        target = rng.choice(["a", "b"], n)
        df = pd.DataFrame({"target": target, "proxy": target})  # copia exata do target
        result = analyze_target(df, "target", config)
        leaking = {item["predictor"] for item in result["possible_leakage"]}
        assert "proxy" in leaking


class TestMultipleComparisonsWarning:
    def test_warning_triggered_above_threshold(self, config):
        rng = np.random.default_rng(4)
        n = 300
        data = {"target": rng.choice(["a", "b"], n)}
        for i in range(15):
            data[f"var_{i}"] = rng.normal(0, 1, n)
        df = pd.DataFrame(data)
        result = analyze_target(df, "target", config)
        assert result["multiple_comparisons_warning"] is not None
        assert result["n_predictors_tested"] == 15

    def test_no_warning_below_threshold(self, simple_binary_df, config):
        result = analyze_target(simple_binary_df, "target", config)
        assert result["multiple_comparisons_warning"] is None


class TestAnalyzeTargetGuards:
    def test_raises_for_multiclass_target(self, config):
        df = pd.DataFrame({"target": ["a", "b", "c"] * 10, "x": range(30)})
        with pytest.raises(AnalysisError):
            analyze_target(df, "target", config)

    def test_raises_for_missing_target_column(self, simple_binary_df, config):
        with pytest.raises(AnalysisError):
            analyze_target(simple_binary_df, "nao_existe", config)


class TestComputePointBiserial:
    def test_insufficient_data_returns_none(self):
        result = compute_point_biserial(pd.Series([1.0, 2.0]), pd.Series([0.0, 1.0]))
        assert result["correlation"] is None


class TestComputeChiSquareAndCramersV:
    def test_perfect_association_returns_cramers_v_near_one(self):
        a = pd.Series(["x", "x", "y", "y"] * 20)
        b = pd.Series(["p", "p", "q", "q"] * 20)
        result = compute_chi_square_and_cramers_v(a, b)
        assert result["cramers_v"] == pytest.approx(1.0, abs=0.05)

    def test_degenerate_table_returns_none(self):
        a = pd.Series(["x"] * 20)
        b = pd.Series(["p", "q"] * 10)
        result = compute_chi_square_and_cramers_v(a, b)
        assert result["cramers_v"] is None
