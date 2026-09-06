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
    UnsupportedLanguageError,
)


def validate_dataframe(df: Any) -> pd.DataFrame:
    """Garante que `df` é um pandas.DataFrame utilizável: não vazio,
    com colunas, sem nomes de coluna duplicados e sem nomes vazios ou
    do padrão "Unnamed: N" (comum em CSV exportado com uma coluna de
    índice sem cabeçalho).

    Levanta InvalidDataFrameError em qualquer uma dessas condições.
    Validamos isso cedo porque um nome de coluna duplicado quebra
    silenciosamente vários módulos a jusante (ex.: df[coluna] passa a
    retornar um DataFrame em vez de uma Series), e um nome vazio/
    "Unnamed" quase sempre indica erro de exportação, não uma coluna
    de dado legítima.

    Retorna o próprio DataFrame para permitir uso em cadeia
    (df = validate_dataframe(df)).
    """
    if not isinstance(df, pd.DataFrame):
        raise InvalidDataFrameError(
            f"Esperado um pandas.DataFrame, recebido {type(df).__name__}."
        )

    if df.shape[1] == 0:
        raise InvalidDataFrameError("O DataFrame não possui nenhuma coluna.")

    if df.shape[0] == 0:
        raise InvalidDataFrameError("O DataFrame não possui nenhuma linha.")

    duplicated_columns = df.columns[df.columns.duplicated()].unique().tolist()
    if duplicated_columns:
        raise InvalidDataFrameError(
            f"O DataFrame possui nome(s) de coluna duplicado(s): {duplicated_columns}. "
            "Renomeie as colunas antes de usar o AutoEDA."
        )

    suspicious_columns = [
        str(column)
        for column in df.columns
        if str(column).strip() == "" or str(column).startswith("Unnamed:")
    ]
    if suspicious_columns:
        raise InvalidDataFrameError(
            f"O DataFrame possui coluna(s) com nome vazio ou no padrão "
            f"'Unnamed: N': {suspicious_columns}. Isso costuma indicar um "
            "índice exportado por engano (ex.: `index=True` ao salvar um CSV) "
            "— revise a exportação ou remova/renomeie essas colunas antes de "
            "usar o AutoEDA."
        )

    return df


def validate_target(df: pd.DataFrame, target: str | None) -> str:
    """Valida a coluna alvo (target) para classificação binária.

    O AutoEDA está restrito a problemas de classificação binária
    (ver escopo do projeto): o target não é mais opcional, e precisa
    ter exatamente 2 classes distintas.

    Regras:
    - Se target is None, levanta InvalidTargetError — toda execução
      do AutoEDA exige uma coluna alvo.
    - Se a coluna não existir em df, levanta InvalidTargetError.
    - Se a coluna for inteiramente nula, levanta InvalidTargetError.
    - Se a coluna tiver um número de classes distintas diferente de 2
      (0, 1 ou 3+), levanta InvalidTargetError — inclui tanto o caso
      degenerado (sem variância) quanto multiclasse (fora de escopo).
    """
    if target is None:
        raise InvalidTargetError(
            "Uma coluna alvo (target) é obrigatória: o AutoEDA está restrito "
            "a problemas de classificação binária e não realiza análise "
            "não supervisionada."
        )

    if target not in df.columns:
        raise InvalidTargetError(
            f"A coluna alvo '{target}' não existe no DataFrame. "
            f"Colunas disponíveis: {list(df.columns)}."
        )

    target_series = df[target]

    if target_series.isna().all():
        raise InvalidTargetError(
            f"A coluna alvo '{target}' é inteiramente nula."
        )

    n_classes = target_series.nunique(dropna=True)

    if n_classes != 2:
        raise InvalidTargetError(
            f"A coluna alvo '{target}' possui {n_classes} classe(s) distinta(s). "
            "O AutoEDA está restrito a classificação binária: o target precisa "
            "ter exatamente 2 classes."
        )

    return target


def validate_language(language: str) -> str:
    """Normaliza (lowercase/strip) e valida o idioma de saída.

    Levanta UnsupportedLanguageError se `language` não estiver em
    autoeda.config.SUPPORTED_LANGUAGES.
    """
    if not isinstance(language, str):
        raise UnsupportedLanguageError(
            f"O idioma deve ser uma string, recebido {type(language).__name__}."
        )

    normalized = language.strip().lower()

    if normalized not in SUPPORTED_LANGUAGES:
        raise UnsupportedLanguageError(
            f"Idioma '{language}' não suportado. "
            f"Idiomas disponíveis: {SUPPORTED_LANGUAGES}."
        )

    return normalized


def infer_column_types(
    df: pd.DataFrame,
    categorical_max_cardinality: int,
    id_cardinality_ratio_threshold: float = 0.95,
) -> dict[str, str]:
    """Classifica cada coluna de `df` em um tipo lógico de análise.

    Retorna um dict {nome_coluna: tipo}, onde tipo é um de:
    "numeric", "categorical", "datetime", "boolean", "text", "id".

    Regras aplicadas, em ordem de precedência:
    1. dtype booleano -> "boolean".
    2. dtype datetime -> "datetime".
    3. Razão (valores únicos / total de linhas) >= id_cardinality_ratio_threshold,
       e não numérica de ponto flutuante e não datetime/booleana -> "id"
       (candidata a exclusão de correlação/target). Não exigimos 100%
       exato: uma coluna de identificador com algumas duplicatas
       legítimas (reenvio de formulário, erro pontual de digitação)
       ainda deve ser reconhecida como id. A razão usa o total de
       linhas (não só as não nulas) como denominador, então uma
       coluna com muitos valores ausentes é penalizada — ausência alta
       reduz a confiança de que a coluna é, de fato, um identificador.
    4. dtype numérico (int/float):
       - baixa cardinalidade (<= categorical_max_cardinality) ->
         "categorical" (ex.: nota de 1 a 5, código de categoria);
       - caso contrário -> "numeric".
    5. dtype objeto/category/string -> "categorical" se a cardinalidade
       for <= categorical_max_cardinality, senão "text" (texto livre,
       ex.: descrições, comentários).

    Nota: booleano e datetime são checados antes de "id" porque uma
    coluna de timestamps únicos é semanticamente uma coluna de data,
    não um identificador, mesmo que cada valor seja único.
    """
    column_types: dict[str, str] = {}
    n_rows = len(df)

    for column in df.columns:
        series = df[column]

        if pd.api.types.is_bool_dtype(series):
            column_types[column] = "boolean"
            continue

        if pd.api.types.is_datetime64_any_dtype(series):
            column_types[column] = "datetime"
            continue

        non_null = series.dropna()
        n_unique = non_null.nunique()
        id_ratio = n_unique / n_rows if n_rows > 0 else 0.0

        is_id_candidate = id_ratio >= id_cardinality_ratio_threshold

        if is_id_candidate and not pd.api.types.is_float_dtype(series):
            column_types[column] = "id"
            continue

        if pd.api.types.is_numeric_dtype(series):
            if n_unique <= categorical_max_cardinality:
                column_types[column] = "categorical"
            else:
                column_types[column] = "numeric"
            continue

        # objeto / category / string
        if n_unique <= categorical_max_cardinality:
            column_types[column] = "categorical"
        else:
            column_types[column] = "text"

    return column_types


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Retorna a lista de colunas numéricas de `df` (int/float),
    excluindo booleanas.
    """
    return [
        column
        for column in df.columns
        if pd.api.types.is_numeric_dtype(df[column])
        and not pd.api.types.is_bool_dtype(df[column])
    ]


def get_categorical_columns(df: pd.DataFrame, categorical_max_cardinality: int) -> list[str]:
    """Retorna a lista de colunas categóricas de `df`, incluindo
    colunas de tipo objeto/category e colunas numéricas de baixa
    cardinalidade (ver infer_column_types).
    """
    column_types = infer_column_types(df, categorical_max_cardinality)
    return [
        column
        for column, col_type in column_types.items()
        if col_type == "categorical"
    ]
