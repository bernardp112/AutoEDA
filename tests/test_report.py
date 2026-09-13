"""Testes de autoeda.report (charts, json_export, builder)."""

from __future__ import annotations

import json

import pytest

from autoeda.analysis.correlation import analyze_correlation
from autoeda.analysis.descriptive import generate_descriptive_stats
from autoeda.analysis.missing_values import analyze_missing_values
from autoeda.analysis.outliers import analyze_outliers
from autoeda.analysis.target_analysis import analyze_target
from autoeda.recommendations import generate_recommendations
from autoeda.report.builder import export_markdown_report
from autoeda.report.charts import generate_all_charts
from autoeda.report.json_export import export_recommendations_json, recommendations_to_json_string


def _run_pipeline(df, target, config):
    desc = generate_descriptive_stats(df, config)
    miss = analyze_missing_values(df, target, config)
    out = analyze_outliers(df, config)
    corr = analyze_correlation(df, config)
    target_res = analyze_target(df, target, config)
    recs = generate_recommendations(desc, miss, out, corr, target_res, config)
    return desc, miss, out, corr, target_res, recs


class TestJsonExport:
    def test_serializes_and_reloads(self, simple_binary_df, config, tmp_path):
        *_, recs = _run_pipeline(simple_binary_df, "target", config)
        path = export_recommendations_json(recs, tmp_path / "out" / "recomendacoes.json")
        assert path.exists()
        with open(path, encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded["schema_version"] == recs["schema_version"]

    def test_infinite_vif_is_json_safe(self, config, tmp_path):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(60)
        n = 200
        b = rng.normal(0, 1, n)
        target = rng.choice(["a", "b"], n)
        df = pd.DataFrame({"target": target, "a": b * 2, "b": b})
        *_, recs = _run_pipeline(df, "target", config)
        # nao deve levantar excecao (Infinity nao seria JSON valido)
        recommendations_to_json_string(recs)


class TestCharts:
    def test_generates_expected_chart_types(self, messy_df, config, tmp_path):
        desc, miss, out, corr, target_res, _ = _run_pipeline(messy_df, "target", config)
        charts = generate_all_charts(messy_df, config, desc, miss, corr, target_res, tmp_path / "charts")
        assert charts["target_distribution"] is not None
        assert charts["target_distribution"].exists()
        assert charts["missing_values"] is not None

    def test_target_column_not_duplicated_in_categorical_charts(self, simple_binary_df, config, tmp_path):
        desc, miss, out, corr, target_res, _ = _run_pipeline(simple_binary_df, "target", config)
        charts = generate_all_charts(simple_binary_df, config, desc, miss, corr, target_res, tmp_path / "charts")
        assert "target" not in charts["categorical"]

    def test_no_missing_values_returns_none(self, simple_binary_df, config, tmp_path):
        desc, miss, out, corr, target_res, _ = _run_pipeline(simple_binary_df, "target", config)
        charts = generate_all_charts(simple_binary_df, config, desc, miss, corr, target_res, tmp_path / "charts")
        assert charts["missing_values"] is None


class TestMarkdownReport:
    def test_report_written_and_contains_target_name(self, messy_df, config, tmp_path):
        desc, miss, out, corr, target_res, recs = _run_pipeline(messy_df, "target", config)
        charts = generate_all_charts(messy_df, config, desc, miss, corr, target_res, tmp_path / "charts")
        report_path = export_markdown_report(
            messy_df.shape, desc, miss, out, corr, target_res, recs, charts, tmp_path / "report.md", config
        )
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "target" in content
        assert "Relatório AutoEDA" in content

    def test_english_report_has_english_headers(self, messy_df, config_en, tmp_path):
        desc, miss, out, corr, target_res, recs = _run_pipeline(messy_df, "target", config_en)
        charts = generate_all_charts(messy_df, config_en, desc, miss, corr, target_res, tmp_path / "charts")
        report_path = export_markdown_report(
            messy_df.shape, desc, miss, out, corr, target_res, recs, charts, tmp_path / "report.md", config_en
        )
        content = report_path.read_text(encoding="utf-8")
        assert "AutoEDA Report" in content
        assert "Dataset overview" in content

    def test_minimal_dataset_no_missing_no_outliers(self, config, tmp_path):
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(3)
        n = 100
        df = pd.DataFrame({"target": rng.choice(["a", "b"], n), "x": rng.uniform(0, 1, n)})
        desc, miss, out, corr, target_res, recs = _run_pipeline(df, "target", config)
        charts = generate_all_charts(df, config, desc, miss, corr, target_res, tmp_path / "charts")
        report_path = export_markdown_report(
            df.shape, desc, miss, out, corr, target_res, recs, charts, tmp_path / "report.md", config
        )
        assert report_path.exists()
