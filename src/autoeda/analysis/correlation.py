"""
Análise de correlação entre variáveis do AutoEDA.

Escopo deste módulo: correlação entre colunas numéricas (tipo lógico
"numeric" conforme utils.infer_column_types). Colunas categóricas e
"id" ficam fora — associação entre variáveis categóricas (ex.: V de
Cramér) e a relação de cada variável com o target são tratadas em
analysis/target_analysis.py, que tem acesso ao contexto de qual
coluna é a variável alvo.

Reportamos três técnicas de multicolinearidade/redundância:
- Pearson: mede associação linear par a par; sensível a outliers.
- Spearman: mede associação monotônica par a par (baseada em ranks);
  mais robusto a outliers e captura relações não lineares mas
  monotônicas.
- VIF (Variance Inflation Factor): diferente das duas acima (que só
  olham pares), mede o quanto cada variável é explicada pela
  combinação linear de *todas* as outras — captura redundância
  multivariada que um par de colunas isoladamente pode não revelar
  (ex.: A = B + C sem A ter correlação alta com B ou C individualmente).

Divergência relevante entre Pearson e Spearman (ex.: Spearman alto e
Pearson baixo) é, em si, um sinal útil para recommendations.py sugerir
transformação (ex.: log) em vez de assumir relação linear direta.

Também reportamos disparidade de escala entre variáveis numéricas —
não é multicolinearidade, mas afeta o mesmo grupo de técnicas de
pré-processamento (modelos baseados em distância/regularização são
sensíveis a variáveis em escalas muito diferentes).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autoeda.config import AutoEDAConfig
from autoeda.utils import infer_column_types


def compute_correlation_matrix(df: pd.DataFrame, columns: list[str], method: str) -> pd.DataFrame:
    """Calcula a matriz de correlação (pearson ou spearman) para as
    colunas informadas.

    `pandas.DataFrame.corr` já ignora pares de linhas onde qualquer
    um dos dois valores é nulo (pairwise deletion), então colunas com
    valores ausentes não precisam de tratamento especial aqui.
    """
    return df[columns].corr(method=method)


def find_high_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float,
    method: str,
) -> list[dict[str, Any]]:
    """Extrai os pares de variáveis com correlação (em módulo) acima
    de `threshold`, evitando pares duplicados (A,B) e (B,A) e a
    diagonal (correlação de uma variável consigo mesma, sempre 1.0).

    Retorna a lista ordenada por força da correlação (do par mais
    correlacionado para o menos), no formato:
    [{"column_a": ..., "column_b": ..., "correlation": ..., "method": ...}, ...]
    """
    columns = corr_matrix.columns.tolist()
    pairs: list[dict[str, Any]] = []

    for i, col_a in enumerate(columns):
        for col_b in columns[i + 1:]:
            value = corr_matrix.loc[col_a, col_b]
            if pd.notna(value) and abs(value) >= threshold:
                pairs.append(
                    {
                        "column_a": col_a,
                        "column_b": col_b,
                        "correlation": float(value),
                        "method": method,
                    }
                )

    pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)
    return pairs


def compute_vif(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, dict[str, Any]]:
    """Calcula o VIF (Variance Inflation Factor) de cada variável
    numérica em relação a todas as outras.

    Para cada coluna, ajusta uma regressão linear (OLS via mínimos
    quadrados, sem dependência de statsmodels) prevendo essa coluna a
    partir de todas as demais + intercepto, calcula o R² dessa
    regressão e VIF = 1 / (1 - R²). VIF alto (convencionalmente >= 5
    ou 10) indica que a variável é quase uma combinação linear das
    outras — a informação que ela carrega é redundante com o
    restante do conjunto, o que infla a variância dos coeficientes em
    modelos lineares.

    Usa apenas linhas completas (sem nenhum valor ausente entre as
    colunas numéricas) — VIF não é definido com dados faltantes.

    Retorna {"<coluna>": {"vif": float | None, "note": str | None}}.
    Casos especiais:
    - menos de 2 colunas numéricas: dict vazio (VIF não é definido
      sem ao menos uma outra variável para explicar a coluna).
    - menos linhas completas do que variáveis + 1: vif=None por
      coluna, com nota (regressão subdeterminada, resultado não
      confiável).
    - R² == 1 (colinearidade perfeita, ex.: A = 2*B exatamente):
      vif=float("inf").
    """
    if len(numeric_columns) < 2:
        return {}

    complete = df[numeric_columns].dropna()
    n_samples = complete.shape[0]
    n_features = len(numeric_columns)

    if n_samples <= n_features:
        return {
            column: {
                "vif": None,
                "note": (
                    f"Apenas {n_samples} linha(s) completa(s) para {n_features} "
                    "variáveis; VIF não é confiável (regressão subdeterminada)."
                ),
            }
            for column in numeric_columns
        }

    results: dict[str, dict[str, Any]] = {}
    values = complete.to_numpy(dtype=float)

    for i, column in enumerate(numeric_columns):
        y = values[:, i]
        other_idx = [j for j in range(n_features) if j != i]
        x_others = values[:, other_idx]
        design = np.column_stack([np.ones(n_samples), x_others])

        beta, _residuals, _rank, _sv = np.linalg.lstsq(design, y, rcond=None)
        y_pred = design @ beta

        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))

        if ss_tot == 0:
            results[column] = {"vif": None, "note": "Variável sem variância; VIF não é definido."}
            continue

        r_squared = 1 - ss_res / ss_tot
        vif = float("inf") if r_squared >= 1.0 else 1.0 / (1.0 - r_squared)
        results[column] = {"vif": vif, "note": None}

    return results


def find_high_vif(vif_results: dict[str, dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Filtra as colunas com VIF >= threshold, ordenadas do maior
    para o menor.
    """
    high_vif = [
        {"column": column, "vif": info["vif"]}
        for column, info in vif_results.items()
        if info["vif"] is not None and info["vif"] >= threshold
    ]
    high_vif.sort(key=lambda item: item["vif"], reverse=True)
    return high_vif


def detect_scale_disparity(
    df: pd.DataFrame,
    numeric_columns: list[str],
    threshold: float,
) -> dict[str, Any] | None:
    """Verifica se as variáveis numéricas do dataset estão em escalas
    muito diferentes, comparando o desvio padrão da variável de maior
    escala com o da de menor escala.

    Não é uma medida de multicolinearidade — é um sinal
    independente, relevante para o mesmo momento de pré-processamento
    (modelos baseados em distância como KNN, ou com regularização
    L1/L2, são sensíveis a features em escalas muito diferentes).

    Retorna None se houver menos de 2 colunas numéricas com desvio
    padrão positivo (nada a comparar), ou se a razão entre a maior e
    a menor escala ficar abaixo de `threshold`.
    """
    stds = {
        column: df[column].std()
        for column in numeric_columns
        if df[column].dropna().shape[0] > 1 and df[column].std() > 0
    }

    if len(stds) < 2:
        return None

    max_column = max(stds, key=stds.get)
    min_column = min(stds, key=stds.get)
    ratio = float(stds[max_column] / stds[min_column])

    if ratio < threshold:
        return None

    return {
        "largest_scale_column": max_column,
        "largest_scale_std": float(stds[max_column]),
        "smallest_scale_column": min_column,
        "smallest_scale_std": float(stds[min_column]),
        "ratio": ratio,
    }


def analyze_correlation(df: pd.DataFrame, config: AutoEDAConfig) -> dict[str, Any]:
    """Executa a análise de correlação/multicolinearidade completa
    entre as variáveis numéricas do dataset.

    Se houver menos de 2 colunas do tipo "numeric", a correlação e o
    VIF não são definidos — retorna um resultado vazio com uma nota
    explicativa em vez de levantar erro, para não interromper o
    restante do pipeline de análise.

    Retorna um dict no formato:
    {
        "columns_analyzed": [...],
        "pearson": {"<col_a>": {"<col_b>": float, ...}, ...},
        "spearman": {"<col_a>": {"<col_b>": float, ...}, ...},
        "high_correlations": [
            {"column_a": ..., "column_b": ..., "correlation": ..., "method": "pearson"}, ...
        ],
        "vif": {"<coluna>": {"vif": float | None, "note": str | None}, ...},
        "high_vif": [{"column": ..., "vif": float}, ...],
        "scale_disparity": {...} | None,
        "note": str | None,
    }
    "high_correlations" combina pares fortes de ambos os métodos
    (deduplicados por método, já que os dois medem coisas distintas e
    um par pode aparecer forte em um e fraco no outro).
    """
    column_types = infer_column_types(
        df, config.categorical_max_cardinality, config.id_cardinality_ratio_threshold
    )
    numeric_columns = [col for col, col_type in column_types.items() if col_type == "numeric"]

    if len(numeric_columns) < 2:
        return {
            "columns_analyzed": numeric_columns,
            "pearson": {},
            "spearman": {},
            "high_correlations": [],
            "vif": {},
            "high_vif": [],
            "scale_disparity": None,
            "note": (
                "Menos de 2 colunas numéricas disponíveis; correlação, VIF e "
                "disparidade de escala não são aplicáveis a este dataset."
            ),
        }

    pearson_matrix = compute_correlation_matrix(df, numeric_columns, method="pearson")
    spearman_matrix = compute_correlation_matrix(df, numeric_columns, method="spearman")

    high_pearson = find_high_correlations(pearson_matrix, config.correlation_high_threshold, "pearson")
    high_spearman = find_high_correlations(spearman_matrix, config.correlation_high_threshold, "spearman")

    vif_results = compute_vif(df, numeric_columns)
    high_vif = find_high_vif(vif_results, config.vif_threshold)

    return {
        "columns_analyzed": numeric_columns,
        "pearson": pearson_matrix.round(4).to_dict(),
        "spearman": spearman_matrix.round(4).to_dict(),
        "high_correlations": high_pearson + high_spearman,
        "vif": vif_results,
        "high_vif": high_vif,
        "scale_disparity": detect_scale_disparity(df, numeric_columns, config.scale_disparity_ratio_threshold),
        "note": None,
    }

