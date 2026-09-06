"""
Análise de valores ausentes do AutoEDA.

Vai além da contagem simples de nulos por coluna (que já é coberta
por analysis/descriptive.py): aqui classificamos a severidade da
ausência por coluna conforme os thresholds de config.py, buscamos
indícios dos 3 mecanismos de dados ausentes (MCAR, MAR, MNAR) e
identificamos linhas com ausência excessiva.

Importante: não é possível determinar com certeza, apenas olhando o
dataset, qual mecanismo está presente — MCAR, MAR e MNAR são
hipóteses sobre o *processo* que gerou a ausência, e dados idênticos
podem ser consistentes com mais de uma hipótese. Este módulo busca
*indícios observáveis* (coocorrência de ausência entre colunas, taxa
de ausência diferente entre classes do target) que apontam para MAR;
na ausência desses indícios, o resultado é reportado como
"indeterminado (MCAR ou MNAR)" — nunca como "MCAR confirmado", porque
MNAR (a ausência depender do próprio valor ausente ou de um fator não
observado) nunca pode ser descartado só com os dados em mãos.

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
    reportados — um indício de MAR (ex.: duas colunas preenchidas pelo
    mesmo formulário opcional), consumido tanto por
    build_missingness_mechanism_hints (abaixo) quanto por
    recommendations.py para sugerir tratamento conjunto em vez de
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


def find_missing_target_association(
    df: pd.DataFrame,
    target: str,
    threshold: float,
) -> list[dict[str, Any]]:
    """Verifica, para cada coluna com valores ausentes, se a taxa de
    ausência difere entre as duas classes do target.

    Uma diferença grande (>= threshold) é um indício de MAR
    especificamente ligado ao problema de classificação — por
    exemplo, "renda" pode faltar muito mais entre quem não pagou um
    empréstimo do que entre quem pagou, o que é informação relevante
    tanto para a estratégia de imputação quanto para o entendimento
    do domínio.

    A própria coluna target é ignorada (não teria sentido comparar sua
    taxa de ausência "contra si mesma"). Colunas cuja ausência é
    idêntica em ambas as classes (diferença 0) não são incluídas no
    resultado.
    """
    results: list[dict[str, Any]] = []

    for column in df.columns:
        if column == target or not df[column].isna().any():
            continue

        missing_by_class = df.groupby(target)[column].apply(lambda s: s.isna().mean())

        if missing_by_class.shape[0] < 2:
            continue

        diff = float(missing_by_class.max() - missing_by_class.min())
        if diff >= threshold:
            results.append(
                {
                    "column": column,
                    "rates_by_class": {str(k): float(v) for k, v in missing_by_class.items()},
                    "diff": diff,
                }
            )

    results.sort(key=lambda item: item["diff"], reverse=True)
    return results


def build_missingness_mechanism_hints(
    missing_columns: list[str],
    correlated_pairs: list[dict[str, Any]],
    target_associations: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Monta, para cada coluna com valores ausentes, um indício
    (nunca uma confirmação) sobre o mecanismo de ausência mais
    provável.

    Regras (sempre em linguagem de indício, nunca afirmação):
    - Se a coluna aparece em algum par de ausência correlacionada
      (find_missing_correlations) e/ou tem taxa de ausência diferente
      entre as classes do target (find_missing_target_association):
      hint="MAR", com a evidência observada listada.
    - Caso contrário: hint="indeterminado", porque a ausência dessas
      evidências não confirma MCAR — MNAR (a ausência depender do
      próprio valor ausente, ex.: quem ganha muito não informa a
      renda) não pode ser descartado só com os dados observados.
    """
    correlated_by_column: dict[str, list[dict[str, Any]]] = {col: [] for col in missing_columns}
    for pair in correlated_pairs:
        for col, other in ((pair["column_a"], pair["column_b"]), (pair["column_b"], pair["column_a"])):
            if col in correlated_by_column:
                correlated_by_column[col].append({"with": other, "correlation": pair["correlation"]})

    target_association_by_column = {item["column"]: item for item in target_associations}

    hints: dict[str, dict[str, Any]] = {}
    for column in missing_columns:
        evidence: list[str] = []

        for corr_info in correlated_by_column.get(column, []):
            evidence.append(
                f"ausência correlacionada com a de '{corr_info['with']}' "
                f"(r={corr_info['correlation']:.2f})"
            )

        target_info = target_association_by_column.get(column)
        if target_info is not None:
            rates = ", ".join(f"{cls}={rate:.1%}" for cls, rate in target_info["rates_by_class"].items())
            evidence.append(f"taxa de ausência difere entre as classes do target ({rates})")

        if evidence:
            hints[column] = {
                "hint": "MAR",
                "evidence": evidence,
                "note": (
                    "Indício de MAR (ausência relacionada a variáveis observadas). "
                    "Não é uma confirmação — o mecanismo real não pode ser "
                    "determinado com certeza apenas a partir dos dados."
                ),
            }
        else:
            hints[column] = {
                "hint": "indeterminado",
                "evidence": [],
                "note": (
                    "Nenhum indício de MAR encontrado (sem correlação com a "
                    "ausência de outras colunas nem diferença relevante entre as "
                    "classes do target). Isso não confirma MCAR: a ausência pode "
                    "ainda depender do próprio valor não observado (MNAR), o que "
                    "não é verificável a partir do dataset."
                ),
            }

    return hints


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


def analyze_missing_values(df: pd.DataFrame, target: str, config: AutoEDAConfig) -> dict[str, Any]:
    """Executa a análise completa de valores ausentes do dataset.

    Combina o resumo por coluna (com severidade classificada), os
    indícios de mecanismo de ausência (MCAR/MAR/MNAR, sempre em
    linguagem de indício) e as linhas com ausência excessiva.

    `target` é obrigatório (o AutoEDA está restrito a classificação
    binária e o target é sempre informado — ver utils.validate_target)
    e é usado para checar se a taxa de ausência de cada coluna difere
    entre as classes do target.

    Retorna um dict no formato:
    {
        "has_missing_values": bool,
        "columns": {
            "<coluna>": {
                "missing": int,
                "missing_pct": float,
                "severity": "low" | "moderate" | "high",
                "mechanism_hint": {"hint": ..., "evidence": [...], "note": ...},
            },
            ...  # apenas colunas com ao menos 1 valor ausente
        },
        "correlated_missingness": [
            {"column_a": ..., "column_b": ..., "correlation": ...}, ...
        ],
        "missing_target_association": [
            {"column": ..., "rates_by_class": {...}, "diff": float}, ...
        ],
        "high_missing_rows": {
            "count": int, "pct": float, "sample_indices": [...]
        },
    }
    """
    missing_summary = compute_missing_summary(df)
    missing_columns = list(missing_summary.keys())

    correlated_pairs = find_missing_correlations(df)
    target_associations = find_missing_target_association(
        df, target, config.missing_target_rate_diff_threshold
    )
    mechanism_hints = build_missingness_mechanism_hints(
        missing_columns, correlated_pairs, target_associations
    )

    columns_report = {
        column: {
            **stats,
            "severity": classify_missing_severity(stats["missing_pct"], config),
            "mechanism_hint": mechanism_hints[column],
        }
        for column, stats in missing_summary.items()
    }

    return {
        "has_missing_values": len(columns_report) > 0,
        "columns": columns_report,
        "correlated_missingness": correlated_pairs,
        "missing_target_association": target_associations,
        "high_missing_rows": find_rows_with_high_missing(df),
    }
