"""Testes de autoeda.utils: validação de entrada e inferência de tipo de coluna."""

from __future__ import annotations

import pandas as pd
import pytest

from autoeda.exceptions import InvalidDataFrameError, InvalidTargetError, UnsupportedLanguageError
from autoeda.utils import (
    get_categorical_columns,
    get_numeric_columns,
    infer_column_types,
    validate_dataframe,
    validate_language,
    validate_target,
)


class TestValidateDataframe:
    def test_valid_dataframe_is_returned_unchanged(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        assert validate_dataframe(df) is df

    def test_rejects_non_dataframe(self):
        with pytest.raises(InvalidDataFrameError):
            validate_dataframe([1, 2, 3])

    def test_rejects_no_columns(self):
        with pytest.raises(InvalidDataFrameError):
            validate_dataframe(pd.DataFrame())

    def test_rejects_no_rows(self):
        with pytest.raises(InvalidDataFrameError):
            validate_dataframe(pd.DataFrame({"a": pd.Series([], dtype="float64")}))

    def test_rejects_duplicate_column_names(self):
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        with pytest.raises(InvalidDataFrameError):
            validate_dataframe(df)

    def test_rejects_empty_column_name(self):
        df = pd.DataFrame({"a": [1, 2], "": [3, 4]})
        with pytest.raises(InvalidDataFrameError):
            validate_dataframe(df)

    def test_rejects_unnamed_column(self):
        df = pd.DataFrame({"a": [1, 2], "Unnamed: 0": [0, 1]})
        with pytest.raises(InvalidDataFrameError):
            validate_dataframe(df)


class TestValidateTarget:
    def test_binary_categorical_target_is_valid(self):
        df = pd.DataFrame({"target": ["sim", "nao", "sim", "nao"]})
        assert validate_target(df, "target") == "target"

    def test_binary_numeric_target_is_valid(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1]})
        assert validate_target(df, "target") == "target"

    def test_none_target_is_rejected(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(InvalidTargetError):
            validate_target(df, None)

    def test_missing_column_is_rejected(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(InvalidTargetError):
            validate_target(df, "nao_existe")

    def test_multiclass_target_is_rejected(self):
        df = pd.DataFrame({"target": ["a", "b", "c", "a"]})
        with pytest.raises(InvalidTargetError):
            validate_target(df, "target")

    def test_constant_target_is_rejected(self):
        df = pd.DataFrame({"target": [1, 1, 1, 1]})
        with pytest.raises(InvalidTargetError):
            validate_target(df, "target")

    def test_fully_null_target_is_rejected(self):
        df = pd.DataFrame({"target": [None, None, None]})
        with pytest.raises(InvalidTargetError):
            validate_target(df, "target")


class TestValidateLanguage:
    def test_accepts_supported_languages(self):
        assert validate_language("pt-br") == "pt-br"
        assert validate_language("en-us") == "en-us"

    def test_normalizes_case_and_whitespace(self):
        assert validate_language(" PT-BR ") == "pt-br"

    def test_rejects_unsupported_language(self):
        with pytest.raises(UnsupportedLanguageError):
            validate_language("fr-fr")


class TestInferColumnTypes:
    def test_id_detected_by_full_uniqueness(self):
        df = pd.DataFrame({"id": range(100)})
        types = infer_column_types(df, categorical_max_cardinality=20, id_cardinality_ratio_threshold=0.95)
        assert types["id"] == "id"

    def test_id_detected_with_small_ratio_of_duplicates(self):
        df = pd.DataFrame({"id": list(range(1, 98)) + [1, 2, 3]})  # 97/100 unique
        types = infer_column_types(df, categorical_max_cardinality=20, id_cardinality_ratio_threshold=0.95)
        assert types["id"] == "id"

    def test_low_uniqueness_ratio_is_not_id(self):
        df = pd.DataFrame({"col": list(range(1, 81)) + list(range(1, 21))})  # 80/100 unique
        types = infer_column_types(df, categorical_max_cardinality=20, id_cardinality_ratio_threshold=0.95)
        assert types["col"] != "id"

    def test_datetime_takes_precedence_over_id(self):
        df = pd.DataFrame({"data": pd.date_range("2024-01-01", periods=50)})
        types = infer_column_types(df, categorical_max_cardinality=20, id_cardinality_ratio_threshold=0.95)
        assert types["data"] == "datetime"

    def test_boolean_column(self):
        df = pd.DataFrame({"flag": [True, False, True, False] * 10})
        types = infer_column_types(df, categorical_max_cardinality=20, id_cardinality_ratio_threshold=0.95)
        assert types["flag"] == "boolean"

    def test_high_missing_ratio_prevents_id_classification(self):
        df = pd.DataFrame({"col": [1, 2, 3] + [None] * 97})
        types = infer_column_types(df, categorical_max_cardinality=20, id_cardinality_ratio_threshold=0.95)
        assert types["col"] != "id"

    def test_get_numeric_and_categorical_columns(self):
        df = pd.DataFrame(
            {
                "numerica": range(50),
                "categorica": (["A", "B"] * 25),
            }
        )
        assert "numerica" in get_numeric_columns(df)
        assert "categorica" in get_categorical_columns(df, categorical_max_cardinality=20)
