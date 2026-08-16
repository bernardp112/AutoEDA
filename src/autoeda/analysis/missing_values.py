"""
Análise de valores ausentes do AutoEDA.

Vai além da contagem simples de nulos por coluna (que já é coberta
por analysis/descriptive.py): aqui classificamos a severidade da
ausência por coluna conforme os thresholds de config.py, detectamos
padrões de coocorrência de ausência entre colunas (indício de dados
faltantes não completamente aleatórios — MAR/MNAR) e identificamos
linhas com ausência excessiva.

Este módulo apenas descreve e classifica; a decisão de "dropar",
"imputar com média/mediana/moda" ou "criar flag de ausência" é
responsabilidade de recommendations.py, que consome a saída daqui.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from autoeda.config import AutoEDAConfig

# Correlação mínima (valor absoluto) entre indicadores de ausência de
# duas colunas para reportá-la como um padrão relevante.
_MISSING_CORRELATION_MIN = 0.5

# Percentual de colunas ausentes em uma linha acima do qual a linha é
# sinalizada como candidata a remoção.
_ROW_MISSING_WARNING_THRESHOLD = 0.5


def compute_missing_summary(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Calcula contagem e percentual de valores ausentes por coluna.

    Retorna apenas as colunas que possuem ao menos um valor ausente
    (colunas 100% preenchidas não aparecem no resultado), no formato:
    {"<coluna>": {"missing": int, "missing_pct": float}}.
    """
    n_rows = len(df)
    summary: dict[str, dict[str, Any]] = {}

    if n_rows == 0:
        return summary

    missing_counts = df.isna().sum()
    for column, count in missing_counts.items():
        if count > 0:
            summary[column] = {
                "missing": int(count),
                "missing_pct": float(count / n_rows),
            }

    return summary


def classify_missing_severity(missing_pct: float, config: AutoEDAConfig) -> str:
    """Classifica a severidade de ausência de uma coluna em um rótulo.

    Retorna um de: "none", "low", "moderate", "high":
    - "none": missing_pct == 0
    - "low": 0 < missing_pct < config.missing_warning_threshold
    - "moderate": config.missing_warning_threshold <= missing_pct <
      config.missing_drop_threshold
    - "high": missing_pct >= config.missing_drop_threshold (candidata
      a remoção da coluna em vez de imputação)
    """
    if missing_pct <= 0:
        return "none"
    if missing_pct < config.missing_warning_threshold:
        return "low"
    if missing_pct < config.missing_drop_threshold:
        return "moderate"
    return "high"


def find_missing_correlations(
    df: pd.DataFrame,
    min_correlation: float = _MISSING_CORRELATION_MIN,
) -> list[dict[str, Any]]:
    """Detecta pares de colunas cuja ausência tende a ocorrer junta.

    Constrói uma matriz booleana de "é nulo" para as colunas que têm
    ao menos um valor ausente e calcula a correlação de Pearson entre
    esses indicadores. Pares com |correlação| >= min_correlation são
    reportados — um sinal de que a ausência não é aleatória (ex.: duas
    colunas preenchidas pelo mesmo formulário opcional), o que ajuda
    recommendations.py a sugerir tratamento conjunto em vez de
    imputação independente por coluna.

    Colunas sem nenhum valor ausente, ou sem nenhuma variação na
    ausência (sempre nula ou nunca nula dentro do subconjunto
    analisado), são ignoradas pois não produzem correlação definida.
    """
    missing_columns = [col for col in df.columns if df[col].isna().any()]

    if len(missing_columns) < 2:
        return []

    missing_mask = df[missing_columns].isna()
    # Remove colunas sem variância (100% nula ou 0% nula dentro deste
    # subconjunto), pois a correlação não é definida (desvio padrão 0).
    varying_columns = [col for col in missing_columns if missing_mask[col].nunique() > 1]

    if len(varying_columns) < 2:
        return []

    corr_matrix = missing_mask[varying_columns].astype(float).corr()

    pairs: list[dict[str, Any]] = []
    for i, col_a in enumerate(varying_columns):
        for col_b in varying_columns[i + 1:]:
            corr_value = corr_matrix.loc[col_a, col_b]
            if pd.notna(corr_value) and abs(corr_value) >= min_correlation:
                pairs.append(
                    {
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": float(corr_value),
                    }
                )

    pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)
    return pairs


def find_rows_with_high_missing(
    df: pd.DataFrame,
    threshold: float = _ROW_MISSING_WARNING_THRESHOLD,
) -> dict[str, Any]:
    """Identifica linhas com percentual de colunas ausentes acima de `threshold`.

    Retorna um resumo (não a lista completa de índices, para não
    inflar o relatório em datasets grandes): contagem de linhas
    afetadas, percentual sobre o total e uma amostra dos índices
    (até 20) para inspeção manual.
    """
    n_columns = df.shape[1]
    n_rows = df.shape[0]

    if n_rows == 0 or n_columns == 0:
        return {"count": 0, "pct": 0.0, "sample_indices": []}

    row_missing_pct = df.isna().sum(axis=1) / n_columns
    affected = row_missing_pct[row_missing_pct >= threshold]

    return {
        "count": int(affected.shape[0]),
        "pct": float(affected.shape[0] / n_rows),
        "sample_indices": [str(idx) for idx in affected.index[:20]],
    }


def analyze_missing_values(df: pd.DataFrame, config: AutoEDAConfig) -> dict[str, Any]:
    """Executa a análise completa de valores ausentes do dataset.

    Combina o resumo por coluna (com severidade classificada), os
    padrões de coocorrência de ausência entre colunas e as linhas
    com ausência excessiva.

    Retorna um dict no formato:
    {
        "has_missing_values": bool,
        "columns": {
            "<coluna>": {
                "missing": int,
                "missing_pct": float,
                "severity": "low" | "moderate" | "high",
            },
            ...  # apenas colunas com ao menos 1 valor ausente
        },
        "correlated_missingness": [
            {"column_a": ..., "column_b": ..., "correlation": ...}, ...
        ],
        "high_missing_rows": {
            "count": int, "pct": float, "sample_indices": [...]
        },
    }
    """
    missing_summary = compute_missing_summary(df)

    columns_report = {
        column: {
            **stats,
            "severity": classify_missing_severity(stats["missing_pct"], config),
        }
        for column, stats in missing_summary.items()
    }

    return {
        "has_missing_values": len(columns_report) > 0,
        "columns": columns_report,
        "correlated_missingness": find_missing_correlations(df),
        "high_missing_rows": find_rows_with_high_missing(df),
    }
