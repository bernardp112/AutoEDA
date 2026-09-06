"""
Estatísticas descritivas do AutoEDA.

Responsabilidade deste módulo: dado um DataFrame já validado, gerar
um resumo estatístico por coluna (adaptado ao tipo lógico da coluna —
numérica, categórica, datetime, booleana, texto ou id) e um resumo
geral do dataset.

Este módulo não decide "o que fazer" com os resultados (isso é
trabalho de recommendations.py); apenas descreve os dados.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from autoeda.config import AutoEDAConfig
from autoeda.utils import infer_column_types

# Número de categorias mais frequentes reportadas na distribuição de
# colunas categóricas (evita relatórios gigantes em colunas com muitas
# categorias).
_TOP_CATEGORIES_LIMIT = 10


def describe_numeric_column(series: pd.Series) -> dict[str, Any]:
    """Calcula estatísticas descritivas de uma coluna numérica.

    Retorna contagens, medidas de tendência central, dispersão,
    quartis e forma da distribuição (assimetria/curtose), úteis para
    o módulo de recomendações identificar necessidade de
    normalização/transformação (ex.: log-transform em dados
    fortemente assimétricos).
    """
    non_null = series.dropna()
    n_missing = int(series.isna().sum())

    if non_null.empty:
        return {
            "count": 0,
            "missing": n_missing,
            "missing_pct": 1.0 if len(series) else 0.0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "iqr": None,
            "max": None,
            "skewness": None,
            "kurtosis": None,
            "zeros_count": 0,
            "negative_count": 0,
        }

    p25 = float(non_null.quantile(0.25))
    p75 = float(non_null.quantile(0.75))

    return {
        "count": int(non_null.shape[0]),
        "missing": n_missing,
        "missing_pct": n_missing / len(series) if len(series) else 0.0,
        "mean": float(non_null.mean()),
        "std": float(non_null.std()) if non_null.shape[0] > 1 else 0.0,
        "min": float(non_null.min()),
        "p25": p25,
        "median": float(non_null.median()),
        "p75": p75,
        "iqr": p75 - p25,
        "max": float(non_null.max()),
        "skewness": float(non_null.skew()) if non_null.shape[0] > 2 else None,
        "kurtosis": float(non_null.kurt()) if non_null.shape[0] > 3 else None,
        "zeros_count": int((non_null == 0).sum()),
        "negative_count": int((non_null < 0).sum()),
    }


def describe_categorical_column(series: pd.Series) -> dict[str, Any]:
    """Calcula estatísticas descritivas de uma coluna categórica.

    Retorna cardinalidade, moda e a distribuição das categorias mais
    frequentes (top N), em contagem e proporção — base para o
    recommendations.py sugerir, por exemplo, agrupar categorias raras
    ("outros") ou aplicar one-hot / target encoding.
    """
    non_null = series.dropna()
    n_missing = int(series.isna().sum())

    if non_null.empty:
        return {
            "count": 0,
            "missing": n_missing,
            "missing_pct": 1.0 if len(series) else 0.0,
            "unique": 0,
            "mode": None,
            "mode_freq": None,
            "mode_pct": None,
            "top_categories": [],
        }

    value_counts = non_null.value_counts()
    top = value_counts.head(_TOP_CATEGORIES_LIMIT)

    return {
        "count": int(non_null.shape[0]),
        "missing": n_missing,
        "missing_pct": n_missing / len(series) if len(series) else 0.0,
        "unique": int(non_null.nunique()),
        "mode": value_counts.index[0],
        "mode_freq": int(value_counts.iloc[0]),
        "mode_pct": float(value_counts.iloc[0] / non_null.shape[0]),
        "top_categories": [
            {
                "value": str(value),
                "count": int(count),
                "pct": float(count / non_null.shape[0]),
            }
            for value, count in top.items()
        ],
    }


def describe_datetime_column(series: pd.Series) -> dict[str, Any]:
    """Calcula estatísticas descritivas de uma coluna temporal.

    Retorna o intervalo coberto (min/max/amplitude em dias) e a
    granularidade aproximada (menor diferença não nula entre datas
    consecutivas ordenadas). Colunas de data ainda são descritas
    estatisticamente mesmo fora do escopo de série temporal — útil,
    por exemplo, para reportar a janela de coleta dos dados.
    """
    non_null = pd.to_datetime(series.dropna())
    n_missing = int(series.isna().sum())

    if non_null.empty:
        return {
            "count": 0,
            "missing": n_missing,
            "missing_pct": 1.0 if len(series) else 0.0,
            "min": None,
            "max": None,
            "range_days": None,
            "min_step": None,
        }

    sorted_values = non_null.sort_values()
    diffs = sorted_values.diff().dropna()
    positive_diffs = diffs[diffs > pd.Timedelta(0)]
    min_step = positive_diffs.min() if not positive_diffs.empty else None

    return {
        "count": int(non_null.shape[0]),
        "missing": n_missing,
        "missing_pct": n_missing / len(series) if len(series) else 0.0,
        "min": non_null.min().isoformat(),
        "max": non_null.max().isoformat(),
        "range_days": (non_null.max() - non_null.min()).days,
        "min_step": str(min_step) if min_step is not None else None,
    }


def describe_boolean_column(series: pd.Series) -> dict[str, Any]:
    """Calcula estatísticas descritivas de uma coluna booleana.

    Retorna a proporção de True/False, útil para detectar
    desbalanceamento quando a coluna booleana é (ou está relacionada
    a) a variável alvo.
    """
    non_null = series.dropna()
    n_missing = int(series.isna().sum())

    if non_null.empty:
        return {
            "count": 0,
            "missing": n_missing,
            "missing_pct": 1.0 if len(series) else 0.0,
            "true_count": 0,
            "false_count": 0,
            "true_pct": None,
        }

    true_count = int(non_null.sum())
    false_count = int(non_null.shape[0] - true_count)

    return {
        "count": int(non_null.shape[0]),
        "missing": n_missing,
        "missing_pct": n_missing / len(series) if len(series) else 0.0,
        "true_count": true_count,
        "false_count": false_count,
        "true_pct": float(true_count / non_null.shape[0]),
    }


def describe_id_column(series: pd.Series) -> dict[str, Any]:
    """Estatísticas mínimas para colunas classificadas como "id".

    Não calculamos estatísticas de distribuição para colunas de
    identificador (não fazem sentido semântico); apenas confirmamos
    contagem e unicidade, para o relatório justificar por que a
    coluna foi excluída das demais análises.
    """
    non_null = series.dropna()
    return {
        "count": int(non_null.shape[0]),
        "missing": int(series.isna().sum()),
        "unique": int(non_null.nunique()),
        "is_unique": bool(non_null.nunique() == non_null.shape[0]),
    }


# Dispatch table: tipo lógico (ver utils.infer_column_types) -> função
# de descrição correspondente. Colunas "text" reutilizam a descrição
# categórica (cardinalidade alta, mas ainda reportamos moda/top
# valores como aproximação útil).
_DESCRIBE_DISPATCH = {
    "numeric": describe_numeric_column,
    "categorical": describe_categorical_column,
    "text": describe_categorical_column,
    "datetime": describe_datetime_column,
    "boolean": describe_boolean_column,
    "id": describe_id_column,
}


def detect_constant_and_near_zero_variance(
    df: pd.DataFrame,
    config: AutoEDAConfig,
) -> dict[str, dict[str, Any]]:
    """Detecta colunas constantes ou quase-constantes (near-zero variance).

    Aplica-se a qualquer tipo de coluna (numérica ou categórica) — a
    heurística é baseada em frequência, não em variância propriamente
    dita, o que a torna válida também para categóricas:

    - "constant": a coluna tem 0 ou 1 valor distinto (ignorando
      nulos). Candidata forte a remoção — não carrega informação
      alguma.
    - "near_zero_variance": não é constante, mas um único valor domina
      a distribuição (freq_ratio = frequência do valor mais comum /
      frequência do 2º mais comum >= near_zero_variance_freq_ratio_threshold)
      E a cardinalidade é baixa em relação ao total de linhas
      (percent_unique <= near_zero_variance_unique_pct_threshold).
      Exigir os dois critérios evita marcar colunas de alta
      cardinalidade que só por coincidência têm uma categoria um
      pouco mais frequente que as demais.

    Colunas sem nenhum valor não nulo são ignoradas aqui (já cobertas
    por missing_values.py com severidade "high").

    Retorna apenas as colunas sinalizadas (constant=True ou
    near_zero_variance=True), no formato:
    {"<coluna>": {"constant": bool, "near_zero_variance": bool,
                  "top_value_pct": float, "freq_ratio": float | None,
                  "n_unique": int}}
    """
    n_rows = len(df)
    results: dict[str, dict[str, Any]] = {}

    if n_rows == 0:
        return results

    for column in df.columns:
        value_counts = df[column].dropna().value_counts()

        if value_counts.empty:
            continue

        n_unique = int(value_counts.shape[0])
        top_value_pct = float(value_counts.iloc[0] / value_counts.sum())

        if n_unique == 1:
            results[column] = {
                "constant": True,
                "near_zero_variance": False,
                "top_value_pct": top_value_pct,
                "freq_ratio": None,
                "n_unique": n_unique,
            }
            continue

        freq_ratio = float(value_counts.iloc[0] / value_counts.iloc[1])
        percent_unique = n_unique / n_rows

        is_near_zero = (
            freq_ratio >= config.near_zero_variance_freq_ratio_threshold
            and percent_unique <= config.near_zero_variance_unique_pct_threshold
        )

        if is_near_zero:
            results[column] = {
                "constant": False,
                "near_zero_variance": True,
                "top_value_pct": top_value_pct,
                "freq_ratio": freq_ratio,
                "n_unique": n_unique,
            }

    return results


def detect_mixed_type_columns(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Detecta colunas de dtype "object" com mistura de valores
    numéricos e não numéricos (ex.: planilha exportada com erro de
    digitação, "1500" e "mil e quinhentos" na mesma coluna).

    Uma coluna assim é classificada por utils.infer_column_types como
    "categorical" ou "text" sem aviso — o que esconde um problema de
    qualidade de dado que provavelmente precisa de limpeza manual
    antes de qualquer análise fazer sentido.

    Só verifica colunas de texto (dtype "object" ou o dtype "str"
    nativo introduzido no pandas 3.x — colunas já com dtype numérico,
    booleano ou datetime não têm esse problema por construção).

    Retorna apenas as colunas sinalizadas, no formato:
    {"<coluna>": {"numeric_pct": float, "non_numeric_pct": float,
                  "sample_numeric_values": [...], "sample_non_numeric_values": [...]}}
    """
    results: dict[str, dict[str, Any]] = {}

    for column in df.columns:
        series = df[column]

        if not pd.api.types.is_string_dtype(series) and not pd.api.types.is_object_dtype(series):
            continue

        non_null = series.dropna()
        if non_null.empty:
            continue

        parsed = pd.to_numeric(non_null, errors="coerce")
        is_numeric = parsed.notna()
        n_numeric = int(is_numeric.sum())
        n_total = int(non_null.shape[0])

        if 0 < n_numeric < n_total:
            non_numeric_values = non_null[~is_numeric]
            numeric_values = non_null[is_numeric]
            results[column] = {
                "numeric_pct": float(n_numeric / n_total),
                "non_numeric_pct": float((n_total - n_numeric) / n_total),
                "sample_numeric_values": [str(v) for v in numeric_values.head(5).tolist()],
                "sample_non_numeric_values": [str(v) for v in non_numeric_values.head(5).tolist()],
            }

    return results


def generate_dataset_overview(df: pd.DataFrame) -> dict[str, Any]:
    """Gera um resumo geral do dataset (não por coluna).

    Inclui dimensões, uso de memória, contagem de linhas duplicadas e
    percentual global de células ausentes — métricas de "saúde geral"
    exibidas no topo do relatório antes do detalhamento por coluna.
    """
    n_rows, n_columns = df.shape
    total_cells = n_rows * n_columns
    total_missing = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())

    return {
        "n_rows": int(n_rows),
        "n_columns": int(n_columns),
        "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
        "duplicate_rows": duplicate_rows,
        "duplicate_rows_pct": float(duplicate_rows / n_rows) if n_rows else 0.0,
        "total_missing_cells": total_missing,
        "total_missing_pct": float(total_missing / total_cells) if total_cells else 0.0,
    }


def generate_descriptive_stats(
    df: pd.DataFrame,
    config: AutoEDAConfig,
) -> dict[str, Any]:
    """Gera as estatísticas descritivas completas do dataset.

    Combina o resumo geral (generate_dataset_overview) com o
    detalhamento por coluna, escolhendo a função de descrição
    apropriada conforme o tipo lógico inferido por
    utils.infer_column_types.

    Retorna um dict no formato:
    {
        "overview": {...},
        "columns": {
            "<nome_coluna>": {
                "type": "numeric" | "categorical" | "datetime" |
                        "boolean" | "text" | "id",
                "stats": {...},
            },
            ...
        },
        "constant_and_near_zero_variance": {
            "<coluna>": {"constant": bool, "near_zero_variance": bool, ...}, ...
        },
        "mixed_type_columns": {
            "<coluna>": {"numeric_pct": float, ...}, ...
        },
    }
    """
    column_types = infer_column_types(
        df, config.categorical_max_cardinality, config.id_cardinality_ratio_threshold
    )

    columns_report: dict[str, Any] = {}
    for column, col_type in column_types.items():
        describe_fn = _DESCRIBE_DISPATCH[col_type]
        columns_report[column] = {
            "type": col_type,
            "stats": describe_fn(df[column]),
        }

    return {
        "overview": generate_dataset_overview(df),
        "columns": columns_report,
        "constant_and_near_zero_variance": detect_constant_and_near_zero_variance(df, config),
        "mixed_type_columns": detect_mixed_type_columns(df),
    }
