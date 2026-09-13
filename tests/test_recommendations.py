"""Testes de autoeda.recommendations."""

from __future__ import annotations

import pytest

from autoeda.analysis.correlation import analyze_correlation
from autoeda.analysis.descriptive import generate_descriptive_stats
from autoeda.analysis.missing_values import analyze_missing_values
from autoeda.analysis.outliers import analyze_outliers
from autoeda.analysis.target_analysis import analyze_target
from autoeda.recommendations import generate_recommendations


def _full_pipeline(df, target, config):
    desc = generate_descriptive_stats(df, config)
    miss = analyze_missing_values(df, target, config)
    out = analyze_outliers(df, config)
    corr = analyze_correlation(df, config)
    target_res = analyze_target(df, target, config)
    return generate_recommendations(desc, miss, out, corr, target_res, config)


class TestSchema:
    def test_output_has_schema_version_and_recommendations_list(self, simple_binary_df, config):
        result = _full_pipeline(simple_binary_df, "target", config)
        assert result["schema_version"] == "1.0"
        assert isinstance(result["recommendations"], list)

    def test_every_recommendation_has_required_fields(self, messy_df, config):
        result = _full_pipeline(messy_df, "target", config)
        required = {"feature", "problem_type", "metric", "metric_value", "severity", "recommendation", "explanation"}
        for rec in result["recommendations"]:
            assert required.issubset(rec.keys())
            assert rec["severity"] in {"high", "medium", "low"}

    def test_recommendations_sorted_by_severity(self, messy_df, config):
        result = _full_pipeline(messy_df, "target", config)
        severities = [r["severity"] for r in result["recommendations"]]
        order = {"high": 0, "medium": 1, "low": 2}
        assert severities == sorted(severities, key=lambda s: order[s])


class TestDataLeakageAlwaysPresent:
    def test_data_leakage_recommendation_always_included(self, simple_binary_df, config):
        result = _full_pipeline(simple_binary_df, "target", config)
        problem_types = {r["problem_type"] for r in result["recommendations"]}
        assert "data_leakage" in problem_types


class TestMessyDatasetTriggersExpectedRules:
    def test_constant_column_flagged_high(self, messy_df, config):
        result = _full_pipeline(messy_df, "target", config)
        rec = next(r for r in result["recommendations"] if r["feature"] == "constante")
        assert rec["severity"] == "high"

    def test_id_column_flagged(self, messy_df, config):
        result = _full_pipeline(messy_df, "target", config)
        assert any(r["feature"] == "id" for r in result["recommendations"])

    def test_skewed_column_flagged(self, messy_df, config):
        result = _full_pipeline(messy_df, "target", config)
        assert any(r["feature"] == "assimetrica" and r["metric"] == "skewness" for r in result["recommendations"])

    def test_correlated_pair_flagged(self, messy_df, config):
        result = _full_pipeline(messy_df, "target", config)
        assert any(r["problem_type"] == "correlation" for r in result["recommendations"])

    def test_missing_target_association_flagged(self, messy_df, config):
        result = _full_pipeline(messy_df, "target", config)
        assert any(r["metric"] == "missing_rate_diff_by_target" for r in result["recommendations"])


class TestLanguageOutput:
    def test_recommendations_translated_to_english(self, simple_binary_df, config_en):
        result = _full_pipeline(simple_binary_df, "target", config_en)
        leakage_rec = next(r for r in result["recommendations"] if r["problem_type"] == "data_leakage")
        assert "training set" in leakage_rec["recommendation"] or "train" in leakage_rec["recommendation"].lower()
