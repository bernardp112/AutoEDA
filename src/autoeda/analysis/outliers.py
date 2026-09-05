"""
Detecção de outliers do AutoEDA.

Aplica-se apenas às colunas classificadas como "numeric" por
utils.infer_column_types (ou seja, exclui colunas numéricas de baixa
cardinalidade já tratadas como categóricas, e exclui colunas "id").

Suporta dois métodos, selecionáveis via AutoEDAConfig.outlier_method:
- "iqr": intervalo interquartil (robusto, não assume normalidade).
- "zscore": desvios em relação à média (assume distribuição
  aproximadamente normal; sensível a outliers extremos que distorcem
  a própria média/desvio usados no cálculo).

Este módulo apenas identifica e quantifica outliers; a decisão de
remover, capar (winsorize) ou manter é responsabilidade de
recommendations.py.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from autoeda.config import AutoEDAConfig
from autoeda.exceptions import AnalysisError
from autoeda.utils import infer_column_types

# Número máximo de valores de outlier reportados como amostra por
# coluna (evita relatórios gigantes em colunas com muitos outliers).
_SAMPLE_VALUES_LIMIT = 10

# Número mínimo de observações não nulas necessário para calcular
# outliers de forma minimamente confiável.
_MIN_OBSERVATIONS = 4


def detect_outliers_iqr(series: pd.Series, multiplier: float) -> dict[str, Any]:
    """Detecta outliers pelo método do intervalo interquartil (IQR).

    Um valor é outlier se estiver abaixo de Q1 - multiplier*IQR ou
    acima de Q3 + multiplier*IQR, onde IQR = Q3 - Q1.

    Retorna limites calculados, contagem/percentual de outliers e uma
    amostra dos valores. Se IQR == 0 (mais de 75% dos valores
    idênticos), qualquer valor diferente da moda seria sinalizado
    como outlier, o que costuma ser ruído estatístico — nesse caso
    retornamos 0 outliers explicitamente, com uma nota.
    """
    non_null = series.dropna()

    if non_null.shape[0] < _MIN_OBSERVATIONS:
        return _empty_outlier_result("iqr", non_null.shape[0], note="observações insuficientes")

    q1 = non_null.quantile(0.25)
    q3 = non_null.quantile(0.75)
    iqr = q3 - q1

    if iqr == 0:
        return _empty_outlier_result(
            "iqr",
            non_null.shape[0],
            lower_bound=float(q1),
            upper_bound=float(q3),
            note="IQR igual a zero (distribuição pouco variável); detecção não aplicável",
        )

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    outlier_mask = (non_null < lower_bound) | (non_null > upper_bound)
    outliers = non_null[outlier_mask]

    return {
        "method": "iqr",
        "count": int(outliers.shape[0]),
        "pct": float(outliers.shape[0] / non_null.shape[0]),
        "lower_bound": float(lower_bound),
        "upper_bound": float(upper_bound),
        "sample_values": sorted(outliers.head(_SAMPLE_VALUES_LIMIT).astype(float).tolist()),
        "note": None,
    }


def detect_outliers_zscore(series: pd.Series, threshold: float) -> dict[str, Any]:
    """Detecta outliers pelo método do z-score.

    Um valor é outlier se |z| = |(x - média) / desvio| > threshold.

    Menos robusto que o IQR em distribuições assimétricas ou com
    outliers extremos (que inflam a própria média/desvio usados no
    cálculo), mas é o método clássico e mais interpretável para
    dados aproximadamente normais.
    """
    non_null = series.dropna()

    if non_null.shape[0] < _MIN_OBSERVATIONS:
        return _empty_outlier_result("zscore", non_null.shape[0], note="observações insuficientes")

    mean = non_null.mean()
    std = non_null.std()

    if std == 0:
        return _empty_outlier_result(
            "zscore",
            non_null.shape[0],
            note="desvio padrão igual a zero (distribuição constante); detecção não aplicável",
        )

    z_scores = (non_null - mean) / std
    outlier_mask = z_scores.abs() > threshold
    outliers = non_null[outlier_mask]

    return {
        "method": "zscore",
        "count": int(outliers.shape[0]),
        "pct": float(outliers.shape[0] / non_null.shape[0]),
        "threshold": float(threshold),
        "mean": float(mean),
        "std": float(std),
        "sample_values": sorted(outliers.head(_SAMPLE_VALUES_LIMIT).astype(float).tolist()),
        "note": None,
    }


def _empty_outlier_result(
    method: str,
    n_observations: int,
    lower_bound: float | None = None,
    upper_bound: float | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    """Monta um resultado "sem outliers detectados" com metadados
    consistentes, usado nos casos de borda (poucas observações,
    variância zero).
    """
    result: dict[str, Any] = {
        "method": method,
        "count": 0,
        "pct": 0.0,
        "sample_values": [],
        "note": note,
    }
    if method == "iqr":
        result["lower_bound"] = lower_bound
        result["upper_bound"] = upper_bound
    return result


# Dispatch: nome do método (config.outlier_method) -> função de
# detecção correspondente.
_DETECT_DISPATCH = {
    "iqr": detect_outliers_iqr,
    "zscore": detect_outliers_zscore,
}


def detect_outliers_column(series: pd.Series, config: AutoEDAConfig) -> dict[str, Any]:
    """Detecta outliers em uma única coluna, usando o método
    configurado em config.outlier_method.

    Levanta AnalysisError se config.outlier_method não for um método
    suportado (deveria ser pego antes por validação de config, mas
    fica aqui como salvaguarda de robustez do pipeline).
    """
    detect_fn = _DETECT_DISPATCH.get(config.outlier_method)
    if detect_fn is None:
        raise AnalysisError(
            f"Método de detecção de outliers '{config.outlier_method}' não é "
            f"suportado. Métodos disponíveis: {list(_DETECT_DISPATCH)}."
        )

    if config.outlier_method == "iqr":
        return detect_fn(series, config.outlier_iqr_multiplier)
    return detect_fn(series, config.outlier_zscore_threshold)


def analyze_outliers(df: pd.DataFrame, config: AutoEDAConfig) -> dict[str, Any]:
    """Executa a detecção de outliers em todas as colunas numéricas do dataset.

    Colunas categóricas (mesmo se armazenadas como int/float, ex.:
    notas de 1 a 5) e colunas "id" são excluídas — outliers só fazem
    sentido semântico em variáveis numéricas contínuas/discretas de
    fato, conforme classificado por utils.infer_column_types.

    Retorna um dict no formato:
    {
        "method": "iqr" | "zscore",
        "columns": {
            "<coluna>": {... resultado de detect_outliers_column ...},
            ...  # apenas colunas do tipo "numeric"
        },
    }
    """
    column_types = infer_column_types(
        df, config.categorical_max_cardinality, config.id_cardinality_ratio_threshold
    )
    numeric_columns = [col for col, col_type in column_types.items() if col_type == "numeric"]

    columns_report = {
        column: detect_outliers_column(df[column], config) for column in numeric_columns
    }

    return {
        "method": config.outlier_method,
        "columns": columns_report,
    }
