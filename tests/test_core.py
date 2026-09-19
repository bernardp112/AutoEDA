"""Testes de autoeda.core (função pública autoeda())."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoeda import AutoEDAResult, autoeda
from autoeda.config import AutoEDAConfig
from autoeda.exceptions import InvalidDataFrameError, InvalidTargetError, UnsupportedLanguageError


class TestAutoedaHappyPath:
    def test_returns_result_with_expected_fields(self, messy_df, tmp_path):
        result = autoeda(messy_df, target="target", output_dir=tmp_path / "saida")

        assert isinstance(result, AutoEDAResult)
        assert result.report_path.exists()
        assert result.recommendations_json_path.exists()
        assert len(result.recommendations["recommendations"]) > 0
        assert result.target_analysis["target"] == "target"

    def test_english_output(self, simple_binary_df, tmp_path):
        result = autoeda(simple_binary_df, target="target", lang="en-us", output_dir=tmp_path / "saida")
        assert result.config.language == "en-us"
        assert "AutoEDA Report" in result.report_path.read_text(encoding="utf-8")

    def test_explicit_config_overrides_lang(self, simple_binary_df, tmp_path):
        custom_config = AutoEDAConfig(language="en-us", outlier_method="zscore")
        result = autoeda(
            simple_binary_df, target="target", lang="pt-br", output_dir=tmp_path / "saida", config=custom_config
        )
        assert result.config.language == "en-us"
        assert result.outliers["method"] == "zscore"


class TestAutoedaValidationErrors:
    def test_none_target_raises(self, simple_binary_df, tmp_path):
        with pytest.raises(InvalidTargetError):
            autoeda(simple_binary_df, target=None, output_dir=tmp_path / "saida")

    def test_missing_target_column_raises(self, simple_binary_df, tmp_path):
        with pytest.raises(InvalidTargetError):
            autoeda(simple_binary_df, target="nao_existe", output_dir=tmp_path / "saida")

    def test_multiclass_target_raises(self, tmp_path):
        df = pd.DataFrame({"target": ["a", "b", "c"] * 10, "x": range(30)})
        with pytest.raises(InvalidTargetError):
            autoeda(df, target="target", output_dir=tmp_path / "saida")

    def test_invalid_language_raises(self, simple_binary_df, tmp_path):
        with pytest.raises(UnsupportedLanguageError):
            autoeda(simple_binary_df, target="target", lang="fr-fr", output_dir=tmp_path / "saida")

    def test_invalid_dataframe_raises(self, tmp_path):
        with pytest.raises(InvalidDataFrameError):
            autoeda([1, 2, 3], target="target", output_dir=tmp_path / "saida")

    def test_duplicate_column_names_raise(self, tmp_path):
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        with pytest.raises(InvalidDataFrameError):
            autoeda(df, target="a", output_dir=tmp_path / "saida")
