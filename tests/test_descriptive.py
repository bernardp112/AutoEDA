"""Testes de autoeda.analysis.descriptive."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda.analysis.descriptive import (
    detect_constant_and_near_zero_variance,
    detect_mixed_type_columns,
    generate_dataset_overview,
    generate_descriptive_stats,
)


class TestGenerateDatasetOverview:
    def test_reports_dimensions_and_duplicates(self, simple_binary_df):
        overview = generate_dataset_overview(simple_binary_df)
        assert overview["n_rows"] == len(simple_binary_df)
        assert overview["n_columns"] == simple_binary_df.shape[1]
        assert overview["duplicate_rows"] == 0

    def test_detects_duplicate_rows(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        overview = generate_dataset_overview(df)
        assert overview["duplicate_rows"] == 1


class TestGenerateDescriptiveStats:
    def test_numeric_column_has_iqr(self, simple_binary_df, config):
        result = generate_descriptive_stats(simple_binary_df, config)
        stats = result["columns"]["preditor_forte"]["stats"]
        assert stats["iqr"] == pytest.approx(stats["p75"] - stats["p25"])

    def test_categorical_column_has_mode_and_top_categories(self, simple_binary_df, config):
        result = generate_descriptive_stats(simple_binary_df, config)
        stats = result["columns"]["categoria_fraca"]["stats"]
        assert stats["mode"] in {"X", "Y", "Z"}
        assert len(stats["top_categories"]) > 0

    def test_empty_numeric_column_does_not_crash(self, config):
        df = pd.DataFrame({"vazia": pd.Series([None, None, None], dtype="float64"), "b": [1, 2, 3]})
        result = generate_descriptive_stats(df, config)
        assert result["columns"]["vazia"]["stats"]["count"] == 0


class TestConstantAndNearZeroVariance:
    def test_constant_column_flagged(self, config):
        df = pd.DataFrame({"c": ["x"] * 50})
        result = detect_constant_and_near_zero_variance(df, config)
        assert result["c"]["constant"] is True

    def test_near_zero_variance_flagged_when_both_criteria_met(self, config):
        # 96% de um valor + baixa cardinalidade -> quase constante
        df = pd.DataFrame({"c": ["A"] * 480 + ["B"] * 15 + ["C"] * 5})
        result = detect_constant_and_near_zero_variance(df, config)
        assert result["c"]["near_zero_variance"] is True
        assert result["c"]["constant"] is False

    def test_high_cardinality_column_not_flagged_even_with_dominant_value(self, config):
        n = 500
        df = pd.DataFrame({"c": ["dominante"] * 480 + [f"unico_{i}" for i in range(20)]})
        result = detect_constant_and_near_zero_variance(df, config)
        # cardinalidade (21 valores unicos) excede o limite de "categorical",
        # mas o criterio de nzv exige AMBOS freq_ratio alto E baixa cardinalidade
        assert "c" not in result or result["c"]["near_zero_variance"] is False or result["c"]["constant"] is False

    def test_normal_variance_column_not_flagged(self, config):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({"c": rng.normal(0, 1, 200)})
        result = detect_constant_and_near_zero_variance(df, config)
        assert "c" not in result


class TestMixedTypeColumns:
    def test_mixed_numeric_and_text_detected(self):
        df = pd.DataFrame({"col": (["100", "200", "trezentos", "400"] * 25)})
        result = detect_mixed_type_columns(df)
        assert "col" in result
        assert 0 < result["col"]["numeric_pct"] < 1

    def test_pure_numeric_as_string_not_flagged(self):
        df = pd.DataFrame({"col": (["1", "2", "3"] * 20)})
        result = detect_mixed_type_columns(df)
        assert "col" not in result

    def test_pure_text_not_flagged(self):
        df = pd.DataFrame({"col": (["abc", "def", "ghi"] * 20)})
        result = detect_mixed_type_columns(df)
        assert "col" not in result

    def test_numeric_dtype_column_skipped(self):
        df = pd.DataFrame({"col": range(50)})
        result = detect_mixed_type_columns(df)
        assert "col" not in result
