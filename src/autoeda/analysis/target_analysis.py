"""
Análise da relação entre as variáveis preditoras e a variável alvo (target).

Diferente de analysis/correlation.py (que só cobre pares
numérico-numérico), este módulo lida com as 4 combinações possíveis
entre o tipo do preditor e o tipo do target:

- numérico  x numérico   -> correlação (Pearson/Spearman)
- categórico x numérico  -> razão de correlação (eta²): quanto da
                             variância do target é explicada pelos
                             grupos do preditor categórico
- numérico  x categórico -> mesma métrica (eta²), papéis invertidos
- categórico x categórico -> V de Cramér, a partir da tabela de
                             contingência

O tipo do target (numérico ou categórico) é inferido pelo mesmo
critério de utils.infer_column_types: "numeric" -> target numérico,
qualquer outro tipo não-id -> target categórico.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autoeda.config import AutoEDAConfig
from autoeda.exceptions import AnalysisError
from autoeda.utils import infer_column_types

# Força mínima (eta² ou V de Cramér) para uma variável ser reportada
# como fortemente associada ao target.
_STRONG_ASSOCIATION_THRESHOLD = 0.30


def get_target_type(df: pd.DataFrame, target: str, config: AutoEDAConfig) -> str:
    """Determina se o target deve ser tratado como "numeric" ou
    "categorical" para fins de análise.

    Reaproveita utils.infer_column_types; colunas "boolean", "text" e
    "id" são tratadas como "categorical" aqui (associação categórica
    ainda se aplica; "id" como target é uma configuração incomum mas
    não impedida — cabe ao usuário avaliar a utilidade do resultado).
    """
    column_types = infer_column_types(
        df, config.categorical_max_cardinality, config.id_cardinality_ratio_threshold
    )
    target_type = column_types.get(target, "categorical")
    return "numeric" if target_type == "numeric" else "categorical"


def analyze_target_distribution(series: pd.Series, target_type: str) -> dict[str, Any]:
    """Resume a distribuição do target antes de avaliar as relações
    com os preditores.

    Para target categórico: contagem/percentual por classe e um
    indicador simples de desbalanceamento (razão entre a classe mais
    e a menos frequente) — relevante porque desbalanceamento forte
    tipicamente pede tratamento específico (reamostragem, métricas
    ponderadas) fora do escopo do AutoEDA, mas deve ser sinalizado.

    Para target numérico: estatísticas básicas de posição/dispersão
    (média, desvio, min, max) — versão resumida do que
    analysis/descriptive.py já calcula em detalhe.
    """
    non_null = series.dropna()

    if target_type == "categorical":
        counts = non_null.value_counts()
        total = counts.sum()
        imbalance_ratio = (
            float(counts.iloc[0] / counts.iloc[-1])
            if len(counts) > 1 and counts.iloc[-1] > 0
            else None
        )
        return {
            "target_type": "categorical",
            "n_classes": int(counts.shape[0]),
            "class_distribution": [
                {"value": str(value), "count": int(count), "pct": float(count / total)}
                for value, count in counts.items()
            ],
            "imbalance_ratio": imbalance_ratio,
        }

    return {
        "target_type": "numeric",
        "mean": float(non_null.mean()) if not non_null.empty else None,
        "std": float(non_null.std()) if non_null.shape[0] > 1 else None,
        "min": float(non_null.min()) if not non_null.empty else None,
        "max": float(non_null.max()) if not non_null.empty else None,
    }


def compute_eta_squared(numeric_values: pd.Series, groups: pd.Series) -> float | None:
    """Calcula a razão de correlação (eta²) entre uma variável
    numérica e uma variável categórica (agrupadora).

    eta² = soma dos quadrados entre grupos / soma dos quadrados total.
    Varia de 0 (grupos têm médias idênticas; o agrupamento não explica
    nada da variância) a 1 (toda a variância é explicada pelos
    grupos). É a métrica natural de "correlação categórica-numérica",
    análoga ao R² de uma ANOVA one-way.

    Retorna None se não houver variância total (todos os valores
    numéricos idênticos) ou menos de 2 grupos válidos.
    """
    paired = pd.DataFrame({"value": numeric_values, "group": groups}).dropna()

    if paired.empty:
        return None

    grand_mean = paired["value"].mean()
    total_ss = ((paired["value"] - grand_mean) ** 2).sum()

    if total_ss == 0:
        return None

    group_stats = paired.groupby("group")["value"].agg(["mean", "count"])
    if group_stats.shape[0] < 2:
        return None

    between_ss = (group_stats["count"] * (group_stats["mean"] - grand_mean) ** 2).sum()

    return float(between_ss / total_ss)


def compute_cramers_v(column_a: pd.Series, column_b: pd.Series) -> float | None:
    """Calcula o V de Cramér entre duas variáveis categóricas, a
    partir da tabela de contingência, com a correção de viés de
    Bergsma (2013), que evita superestimar a associação em amostras
    pequenas ou tabelas com muitas categorias.

    Retorna None se a tabela de contingência for degenerada (menos de
    2x2 categorias válidas após remover linhas com nulo em qualquer
    uma das duas colunas).
    """
    paired = pd.DataFrame({"a": column_a, "b": column_b}).dropna()

    if paired.empty:
        return None

    contingency = pd.crosstab(paired["a"], paired["b"])
    n_rows, n_cols = contingency.shape

    if n_rows < 2 or n_cols < 2:
        return None

    n = contingency.values.sum()
    row_totals = contingency.sum(axis=1).values
    col_totals = contingency.sum(axis=0).values
    expected = np.outer(row_totals, col_totals) / n

    # Evita divisão por zero em células esperadas nulas (combinação de
    # categorias que nunca coocorre nos dados observados).
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum(
            np.where(expected > 0, (contingency.values - expected) ** 2 / expected, 0.0)
        )

    phi2 = chi2 / n
    # Correção de Bergsma: reduz o viés de phi2, r e k para amostras finitas.
    phi2_corrected = max(0.0, phi2 - ((n_rows - 1) * (n_cols - 1)) / (n - 1))
    r_corrected = n_rows - ((n_rows - 1) ** 2) / (n - 1)
    k_corrected = n_cols - ((n_cols - 1) ** 2) / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)

    if denominator <= 0:
        return None

    return float(np.sqrt(phi2_corrected / denominator))


def analyze_predictor_vs_target(
    df: pd.DataFrame,
    predictor: str,
    predictor_type: str,
    target: str,
    target_type: str,
) -> dict[str, Any]:
    """Analisa a relação entre uma única variável preditora e o target.

    Escolhe a métrica de associação conforme a combinação de tipos
    (ver docstring do módulo) e retorna um resultado no formato:
    {
        "predictor": ..., "predictor_type": ...,
        "relationship": "numeric_numeric" | "categorical_numeric" |
                        "numeric_categorical" | "categorical_categorical",
        "metric": "pearson" | "eta_squared" | "cramers_v",
        "value": float | None,
        "pearson": float | None,   # apenas quando relationship == numeric_numeric
        "spearman": float | None,  # apenas quando relationship == numeric_numeric
    }
    """
    if predictor_type == "numeric" and target_type == "numeric":
        paired = df[[predictor, target]].dropna()
        pearson = float(paired[predictor].corr(paired[target], method="pearson")) if len(paired) > 1 else None
        spearman = float(paired[predictor].corr(paired[target], method="spearman")) if len(paired) > 1 else None
        return {
            "predictor": predictor,
            "predictor_type": predictor_type,
            "relationship": "numeric_numeric",
            "metric": "pearson",
            "value": pearson,
            "pearson": pearson,
            "spearman": spearman,
        }

    if predictor_type == "numeric" and target_type == "categorical":
        value = compute_eta_squared(df[predictor], df[target])
        return {
            "predictor": predictor,
            "predictor_type": predictor_type,
            "relationship": "numeric_categorical",
            "metric": "eta_squared",
            "value": value,
        }

    if predictor_type != "numeric" and target_type == "numeric":
        value = compute_eta_squared(df[target], df[predictor])
        return {
            "predictor": predictor,
            "predictor_type": predictor_type,
            "relationship": "categorical_numeric",
            "metric": "eta_squared",
            "value": value,
        }

    # categórico x categórico
    value = compute_cramers_v(df[predictor], df[target])
    return {
        "predictor": predictor,
        "predictor_type": predictor_type,
        "relationship": "categorical_categorical",
        "metric": "cramers_v",
        "value": value,
    }


def analyze_target(df: pd.DataFrame, target: str, config: AutoEDAConfig) -> dict[str, Any]:
    """Executa a análise completa da variável alvo em relação a todas
    as demais colunas do dataset.

    Colunas do tipo "id" são excluídas dos preditores (não fazem
    sentido como preditoras). O próprio target é excluído da lista de
    preditores.

    Levanta AnalysisError se `target` não existir em df — validação
    "de negócio" (target ausente do dataset) já deveria ter sido feita
    por utils.validate_target antes do pipeline chegar aqui; este
    erro é uma salvaguarda de robustez.

    Retorna um dict no formato:
    {
        "target": ..., "target_type": "numeric" | "categorical",
        "distribution": {... analyze_target_distribution ...},
        "predictors": [
            {... analyze_predictor_vs_target ...}, ...
        ],  # ordenado por força de associação (|value|), decrescente
        "strong_predictors": [<nomes de preditores com |value| >= threshold>],
    }
    """
    if target not in df.columns:
        raise AnalysisError(f"A coluna alvo '{target}' não existe no DataFrame.")

    target_type = get_target_type(df, target, config)
    distribution = analyze_target_distribution(df[target], target_type)

    column_types = infer_column_types(
        df, config.categorical_max_cardinality, config.id_cardinality_ratio_threshold
    )
    predictor_columns = [
        col for col, col_type in column_types.items() if col != target and col_type != "id"
    ]

    predictor_results = []
    for predictor in predictor_columns:
        predictor_type = column_types[predictor]
        result = analyze_predictor_vs_target(df, predictor, predictor_type, target, target_type)
        predictor_results.append(result)

    predictor_results.sort(
        key=lambda r: abs(r["value"]) if r["value"] is not None else -1,
        reverse=True,
    )

    strong_predictors = [
        r["predictor"]
        for r in predictor_results
        if r["value"] is not None and abs(r["value"]) >= _STRONG_ASSOCIATION_THRESHOLD
    ]

    return {
        "target": target,
        "target_type": target_type,
        "distribution": distribution,
        "predictors": predictor_results,
        "strong_predictors": strong_predictors,
    }
