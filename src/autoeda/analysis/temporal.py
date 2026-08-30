"""
Análise temporal básica do AutoEDA.

Ativada quando o usuário indica, na chamada da função principal, que
o dataset tem uma componente temporal (is_temporal=True) e informa a
coluna de data/hora correspondente.

Escopo (básico, conforme o TCC): inferir a granularidade da série,
agregar os valores ao longo do tempo, detectar tendência (crescente/
decrescente/estável via regressão linear simples) e dar uma pista de
sazonalidade (médias por mês / dia da semana). Modelos de série
temporal mais sofisticados (decomposição STL, ARIMA, etc.) ficam
fora do escopo — este módulo orienta o usuário, não substitui uma
análise de séries temporais completa.

Quando o target é numérico, a série analisada é o próprio target
agregado ao longo do tempo. Quando não há target ou o target é
categórico, a série analisada é o volume de registros por período
(contagem de linhas), útil para detectar tendência de crescimento/
queda no próprio volume de dados.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from autoeda.config import AutoEDAConfig

# Mapeia a granularidade inferida para o alias de frequência do
# pandas usado em resample/Grouper.
_FREQUENCY_ALIASES = {
    "daily": "D",
    "weekly": "W",
    "monthly": "MS",
    "yearly": "YS",
}

# Ordem do mais fino ao mais grosso, usada para decidir se ainda há
# granularidade suficiente para uma quebra de sazonalidade adicional
# (ex.: só faz sentido olhar "dia da semana" se a frequência de base
# for diária ou mais fina).
_GRANULARITY_ORDER = ["daily", "weekly", "monthly", "yearly", "irregular"]


def infer_frequency(dates: pd.Series) -> str:
    """Infere a granularidade aproximada de uma série de datas
    ordenadas, a partir da diferença mediana entre observações
    consecutivas (mais robusta a poucos espaçamentos irregulares do
    que a diferença mínima).

    Retorna um de: "daily", "weekly", "monthly", "yearly", "irregular"
    (quando a série tem menos de 2 pontos, ou o espaçamento não se
    aproxima de nenhuma granularidade usual).

    Datas duplicadas são removidas antes do cálculo: em dados de
    evento (várias linhas no mesmo dia), o passo entre timestamps
    brutos tenderia a 0 e mascararia a granularidade real do
    calendário subjacente — o que importa aqui é o espaçamento entre
    períodos distintos, não a taxa de chegada de eventos.
    """
    sorted_dates = dates.drop_duplicates().sort_values()
    diffs = sorted_dates.diff().dropna()

    if diffs.empty:
        return "irregular"

    median_step = diffs.median()
    days = median_step.total_seconds() / 86400

    if days <= 0:
        return "irregular"
    if 0.5 <= days <= 3:
        return "daily"
    if 5 <= days <= 9:
        return "weekly"
    if 25 <= days <= 35:
        return "monthly"
    if 350 <= days <= 380:
        return "yearly"
    return "irregular"


def aggregate_time_series(
    df: pd.DataFrame,
    temporal_column: str,
    value_column: str | None,
    frequency_alias: str,
) -> pd.Series:
    """Agrega os dados por período de tempo.

    Se `value_column` for informado, agrega pela média do valor no
    período (uso: acompanhar o target numérico ao longo do tempo).
    Caso contrário, agrega pela contagem de linhas no período (uso:
    acompanhar o volume de registros ao longo do tempo).

    Retorna uma Series indexada pelo início de cada período.
    """
    indexed = df.set_index(pd.DatetimeIndex(pd.to_datetime(df[temporal_column])))

    if value_column is not None:
        return indexed[value_column].resample(frequency_alias).mean().dropna()

    return indexed[temporal_column].resample(frequency_alias).count()


def detect_trend(series: pd.Series) -> dict[str, Any]:
    """Detecta a tendência de uma série temporal já agregada, via
    regressão linear simples (valor ~ índice do período).

    Retorna a inclinação (slope, na unidade "valor por período"), o
    coeficiente de correlação de Pearson entre o tempo e o valor
    (força da tendência) e uma direção categórica derivada da
    correlação: "crescente" (r > 0.3), "decrescente" (r < -0.3) ou
    "estavel" (|r| <= 0.3) — o mesmo threshold de força "moderada"
    usado em correlation.py, por consistência.

    Retorna direction="indeterminado" se houver menos de 3 pontos
    (regressão não confiável) ou variância nula na série.
    """
    if series.shape[0] < 3:
        return {"slope": None, "correlation": None, "direction": "indeterminado"}

    time_index = np.arange(series.shape[0], dtype=float)
    values = series.to_numpy(dtype=float)

    if np.std(values) == 0:
        return {"slope": 0.0, "correlation": 0.0, "direction": "estavel"}

    slope, _intercept = np.polyfit(time_index, values, deg=1)
    correlation = float(np.corrcoef(time_index, values)[0, 1])

    if correlation > 0.3:
        direction = "crescente"
    elif correlation < -0.3:
        direction = "decrescente"
    else:
        direction = "estavel"

    return {"slope": float(slope), "correlation": correlation, "direction": direction}


def analyze_seasonality_hint(
    df: pd.DataFrame,
    temporal_column: str,
    value_column: str | None,
    frequency: str,
) -> dict[str, Any] | None:
    """Dá uma pista simples de sazonalidade, agrupando os valores por
    mês (se a frequência de base for diária/semanal/mensal) ou por
    dia da semana (se a frequência de base for diária).

    Não é uma decomposição sazonal formal — apenas médias por
    subperíodo, suficiente para o relatório sinalizar "valores mais
    altos em dezembro", por exemplo, e recommendations.py sugerir
    incluir uma feature de mês/dia-da-semana no pré-processamento.

    Retorna None se a frequência for "yearly" ou "irregular" (não há
    subperíodo natural a agrupar).
    """
    if frequency not in ("daily", "weekly", "monthly"):
        return None

    dates = pd.to_datetime(df[temporal_column])
    values = df[value_column] if value_column is not None else pd.Series(1, index=df.index)

    result: dict[str, Any] = {}

    by_month = values.groupby(dates.dt.month).mean()
    result["by_month"] = {int(month): float(v) for month, v in by_month.items()}

    if frequency == "daily":
        by_weekday = values.groupby(dates.dt.dayofweek).mean()
        result["by_weekday"] = {int(day): float(v) for day, v in by_weekday.items()}

    return result


def analyze_temporal(
    df: pd.DataFrame,
    temporal_column: str,
    target: str | None,
    config: AutoEDAConfig,
) -> dict[str, Any]:
    """Executa a análise temporal básica do dataset.

    Se `target` for informado e for numérico, a série analisada
    (agregação, tendência, sazonalidade) é o target; caso contrário é
    o volume de registros por período.

    Se o número de observações não nulas na coluna temporal for menor
    que config.temporal_min_observations, a análise é considerada
    pouco confiável e retorna um resultado mínimo com nota explicativa
    em vez de tendência/sazonalidade calculadas sobre poucos pontos.

    Retorna um dict no formato:
    {
        "temporal_column": ..., "value_column": <target ou None>,
        "n_observations": int, "date_range": {"min": ..., "max": ...},
        "frequency": "daily" | "weekly" | "monthly" | "yearly" | "irregular",
        "trend": {... detect_trend ...} | None,
        "seasonality_hint": {...} | None,
        "note": str | None,
    }
    """
    valid_dates = pd.to_datetime(df[temporal_column].dropna())
    n_observations = int(valid_dates.shape[0])

    if n_observations < config.temporal_min_observations:
        return {
            "temporal_column": temporal_column,
            "value_column": None,
            "n_observations": n_observations,
            "date_range": None,
            "frequency": None,
            "trend": None,
            "seasonality_hint": None,
            "note": (
                f"Apenas {n_observations} observações temporais válidas "
                f"(mínimo configurado: {config.temporal_min_observations}); "
                "análise de tendência/sazonalidade não é confiável e foi omitida."
            ),
        }

    value_column = target if target is not None and pd.api.types.is_numeric_dtype(df[target]) else None

    frequency = infer_frequency(valid_dates)
    frequency_alias = _FREQUENCY_ALIASES.get(frequency)

    trend = None
    seasonality_hint = None
    if frequency_alias is not None:
        aggregated = aggregate_time_series(df, temporal_column, value_column, frequency_alias)
        trend = detect_trend(aggregated)
        seasonality_hint = analyze_seasonality_hint(df, temporal_column, value_column, frequency)

    return {
        "temporal_column": temporal_column,
        "value_column": value_column,
        "n_observations": n_observations,
        "date_range": {
            "min": valid_dates.min().isoformat(),
            "max": valid_dates.max().isoformat(),
        },
        "frequency": frequency,
        "trend": trend,
        "seasonality_hint": seasonality_hint,
        "note": None if frequency_alias is not None else (
            "Granularidade da série não pôde ser determinada com confiança "
            "(espaçamento irregular entre observações); tendência e "
            "sazonalidade foram omitidas."
        ),
    }
