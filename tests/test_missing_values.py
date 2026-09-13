"""Testes de autoeda.analysis.missing_values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.analysis.missing_values import (
    analyze_missing_values,
    classify_missing_severity,
    find_missing_correlations,
    find_missing_target_association,
)


class TestClassifyMissingSeverity:
    @pytest.mark.parametrize(
        "pct,expected",
        [(0.0, "none"), (0.02, "low"), (0.10, "moderate"), (0.70, "high")],
    )
    def test_severity_thresholds(self, pct, expected, config):
        assert classify_missing_severity(pct, config) == expected


class TestFindMissingCorrelations:
    def test_correlated_missingness_detected(self):
        rng = np.random.default_rng(2)
        n = 300
        df = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)})
        idx = df.sample(frac=0.2, random_state=1).index
        df.loc[idx, "a"] = None
        df.loc[idx, "b"] = None  # ausencia sempre junta

        pairs = find_missing_correlations(df)
        assert len(pairs) == 1
        assert pairs[0]["correlation"] == pytest.approx(1.0, abs=0.01)

    def test_no_pairs_with_single_missing_column(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        assert find_missing_correlations(df) == []

    def test_no_pairs_when_nothing_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert find_missing_correlations(df) == []


class TestFindMissingTargetAssociation:
    def test_detects_rate_difference_across_classes(self):
        rng = np.random.default_rng(15)
        n = 400
        target = rng.choice(["inadimplente", "adimplente"], size=n, p=[0.3, 0.7])
        df = pd.DataFrame({"target": target, "renda": rng.normal(3000, 500, n)})
        mask = df["target"] == "inadimplente"
        df.loc[df[mask].sample(frac=0.6, random_state=1).index, "renda"] = None

        result = find_missing_target_association(df, "target", threshold=0.10)
        assert len(result) == 1
        assert result[0]["column"] == "renda"
        assert result[0]["diff"] > 0.10

    def test_no_association_below_threshold(self):
        rng = np.random.default_rng(0)
        n = 200
        target = rng.choice(["a", "b"], n)
        df = pd.DataFrame({"target": target, "x": rng.normal(0, 1, n)})
        df.loc[df.sample(frac=0.1, random_state=1).index, "x"] = None
        result = find_missing_target_association(df, "target", threshold=0.10)
        assert result == []


class TestAnalyzeMissingValues:
    def test_no_missing_values(self, config):
        df = pd.DataFrame({"target": ["a", "b"] * 10, "x": range(20)})
        result = analyze_missing_values(df, "target", config)
        assert result["has_missing_values"] is False
        assert result["columns"] == {}

    def test_mechanism_hint_mar_when_correlated(self, config):
        rng = np.random.default_rng(2)
        n = 300
        df = pd.DataFrame(
            {
                "target": rng.choice(["a", "b"], n),
                "endereco": ["Rua X"] * n,
                "cep": ["12345"] * n,
            }
        )
        idx = df.sample(frac=0.2, random_state=1).index
        df.loc[idx, "endereco"] = None
        df.loc[idx, "cep"] = None

        result = analyze_missing_values(df, "target", config)
        assert result["columns"]["endereco"]["mechanism_hint"]["hint"] == "MAR"
        assert len(result["columns"]["endereco"]["mechanism_hint"]["evidence"]) > 0

    def test_mechanism_hint_indeterminate_without_evidence(self, config):
        rng = np.random.default_rng(9)
        n = 200
        target = rng.choice(["a", "b"], n)
        df = pd.DataFrame({"target": target, "x": rng.normal(0, 1, n)})
        df.loc[df.sample(frac=0.1, random_state=1).index, "x"] = None

        result = analyze_missing_values(df, "target", config)
        assert result["columns"]["x"]["mechanism_hint"]["hint"] == "indeterminado"

    def test_messy_df_has_expected_missing_column(self, messy_df, config):
        result = analyze_missing_values(messy_df, "target", config)
        assert "renda" in result["columns"]
        assert len(result["missing_target_association"]) >= 1
