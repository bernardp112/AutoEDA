"""
Funções utilitárias de validação e inferência usadas por todo o
pipeline do AutoEDA.

Ficam aqui apenas funções "de apoio" reutilizadas por mais de um
módulo (validação de entrada, inferência de tipo de coluna). Lógica
de análise estatística em si pertence a analysis/.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from autoeda.config import SUPPORTED_LANGUAGES
from autoeda.exceptions import (
    InvalidDataFrameError,
    InvalidTargetError,
    InvalidTemporalColumnError,
    UnsupportedLanguageError,
)


def validate_dataframe(df: Any) -> pd.DataFrame:
    """Garante que `df` é um pandas.DataFrame não vazio e com colunas.

    Levanta InvalidDataFrameError caso contrário. Retorna o próprio
    DataFrame para permitir uso em cadeia (df = validate_dataframe(df)).
    """
    raise NotImplementedError


def validate_target(df: pd.DataFrame, target: str | None) -> str | None:
    """Valida a coluna alvo (target), se informada.

    Regras a implementar:
    - Se target is None, retorna None (análise não supervisionada).
    - Se a coluna não existir em df, levanta InvalidTargetError.
    - Se a coluna for inteiramente nula, levanta InvalidTargetError.
    - Se a coluna tiver um único valor distinto (variância zero),
      levanta InvalidTargetError.
    """
    raise NotImplementedError


def validate_temporal_column(df: pd.DataFrame, temporal_column: str | None) -> str | None:
    """Valida a coluna temporal, se informada.

    Regras a implementar:
    - Se temporal_column is None, retorna None.
    - Se a coluna não existir em df, levanta InvalidTemporalColumnError.
    - Se a coluna não puder ser convertida para datetime
      (pd.to_datetime com errors='raise' em uma amostra), levanta
      InvalidTemporalColumnError.
    """
    raise NotImplementedError


def validate_language(language: str) -> str:
    """Normaliza (lowercase/strip) e valida o idioma de saída.

    Levanta UnsupportedLanguageError se `language` não estiver em
    autoeda.config.SUPPORTED_LANGUAGES.
    """
    raise NotImplementedError


def infer_column_types(
    df: pd.DataFrame,
    categorical_max_cardinality: int,
) -> dict[str, str]:
    """Classifica cada coluna de `df` em um tipo lógico de análise.

    Retorna um dict {nome_coluna: tipo}, onde tipo é um de:
    "numeric", "categorical", "datetime", "boolean", "text", "id".

    Notas de implementação futura:
    - Colunas numéricas com baixa cardinalidade (<=
      categorical_max_cardinality) devem ser reclassificadas como
      "categorical".
    - Colunas com valores 100% únicos (e tipo objeto/int) são fortes
      candidatas a "id" e devem ser sinalizadas para possível exclusão
      da análise de correlação/target.
    """
    raise NotImplementedError


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Retorna a lista de colunas numéricas de `df` (int/float),
    excluindo booleanas.
    """
    raise NotImplementedError


def get_categorical_columns(df: pd.DataFrame, categorical_max_cardinality: int) -> list[str]:
    """Retorna a lista de colunas categóricas de `df`, incluindo
    colunas de tipo objeto/category e colunas numéricas de baixa
    cardinalidade (ver infer_column_types).
    """
    raise NotImplementedError
