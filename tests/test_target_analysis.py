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


class TestLeakageViaSingleFeatureAuc:
    def test_threshold_target_flagged_by_auc_despite_moderate_correlation(self, config):
        rng = np.random.default_rng(0)
        x = rng.lognormal(mean=0, sigma=1.5, size=2000) - 1.0
        df = pd.DataFrame({"x": x, "ruido": rng.normal(size=2000), "y": (x > 0).astype(int)})
        result = analyze_target(df, "y", config)
        flagged = {item["predictor"]: item for item in result["possible_leakage"]}
        assert "x" in flagged
        assert flagged["x"]["metric"] == "auc"
        assert "ruido" not in flagged

    def test_weak_predictor_not_flagged(self, config):
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"a": rng.normal(size=500), "y": rng.integers(0, 2, size=500)})
        assert analyze_target(df, "y", config)["possible_leakage"] == []

    def test_auc_criterion_requires_minimum_observations(self, config):
        # Com poucas observações, uma AUC perfeita pode ser acaso: não deve sinalizar.
        df = pd.DataFrame({"x": np.arange(12, dtype=float), "y": [0] * 6 + [1] * 6})
        assert analyze_target(df, "y", config)["possible_leakage"] == []

    def test_non_monotonic_threshold_is_not_detected_by_auc(self, config):
        # Limitação documentada: relações não monotônicas ficam fora do critério de AUC.
        rng = np.random.default_rng(0)
        x = rng.normal(size=3000)
        df = pd.DataFrame({"x": x, "y": (np.abs(x) > 1).astype(int)})
        result = analyze_target(df, "y", config)
        assert all(item["predictor"] != "x" for item in result["possible_leakage"])


class TestCramersVWithoutContinuityCorrection:
    def test_v_matches_pearson_chi_square_for_2x2(self):
        rng = np.random.default_rng(0)
        a = pd.Series(rng.integers(0, 2, 400))
        b = pd.Series(np.where(rng.random(400) < 0.8, a, 1 - a))
        result = compute_chi_square_and_cramers_v(a, b)
        n = len(a)
        phi = np.sqrt(result["chi2"] / n)  # chi2 do teste padrão (Yates) é <= Pearson
        # V (com correção de Bergsma) deve ser ~ phi calculado do qui-quadrado de Pearson
        from scipy import stats

        pearson = stats.chi2_contingency(pd.crosstab(a, b), correction=False)[0]
        assert result["cramers_v"] == pytest.approx(np.sqrt(max(0, pearson / n - 1 / (n - 1)) / (1 - 1 / (n - 1))), rel=1e-6)
        assert phi <= np.sqrt(pearson / n) + 1e-12
