"""
Análise de correlação entre variáveis do AutoEDA.

Escopo deste módulo: correlação entre colunas numéricas (tipo lógico
"numeric" conforme utils.infer_column_types). Colunas categóricas e
"id" ficam fora — associação entre variáveis categóricas (ex.: V de
Cramér) e a relação de cada variável com o target são tratadas em
analysis/target_analysis.py, que tem acesso ao contexto de qual
coluna é a variável alvo.

Reportamos dois coeficientes:
- Pearson: mede associação linear; sensível a outliers.
- Spearman: mede associação monotônica (baseada em ranks); mais
  robusto a outliers e captura relações não lineares mas monotônicas.

Divergência relevante entre os dois (ex.: Spearman alto e Pearson
baixo) é, em si, um sinal útil para recommendations.py sugerir
transformação (ex.: log) em vez de assumir relação linear direta.
"""

from __future__ import annotations

from typing import Any

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


def analyze_correlation(df: pd.DataFrame, config: AutoEDAConfig) -> dict[str, Any]:
    """Executa a análise de correlação completa entre as variáveis numéricas do dataset.

    Se houver menos de 2 colunas do tipo "numeric", a correlação não
    é definida — retorna um resultado vazio com uma nota explicativa
    em vez de levantar erro, para não interromper o restante do
    pipeline de análise.

    Retorna um dict no formato:
    {
        "columns_analyzed": [...],
        "pearson": {"<col_a>": {"<col_b>": float, ...}, ...},
        "spearman": {"<col_a>": {"<col_b>": float, ...}, ...},
        "high_correlations": [
            {"column_a": ..., "column_b": ..., "correlation": ..., "method": "pearson"}, ...
        ],
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
            "note": (
                "Menos de 2 colunas numéricas disponíveis; correlação entre "
                "variáveis não é aplicável a este dataset."
            ),
        }

    pearson_matrix = compute_correlation_matrix(df, numeric_columns, method="pearson")
    spearman_matrix = compute_correlation_matrix(df, numeric_columns, method="spearman")

    high_pearson = find_high_correlations(pearson_matrix, config.correlation_high_threshold, "pearson")
    high_spearman = find_high_correlations(spearman_matrix, config.correlation_high_threshold, "spearman")

    return {
        "columns_analyzed": numeric_columns,
        "pearson": pearson_matrix.round(4).to_dict(),
        "spearman": spearman_matrix.round(4).to_dict(),
        "high_correlations": high_pearson + high_spearman,
        "note": None,
    }
