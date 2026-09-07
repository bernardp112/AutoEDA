"""
Geração de gráficos do AutoEDA (Matplotlib), para embutir no
relatório final em Markdown.

Escopo (ver escopo do projeto, item 14): gerar gráficos apenas quando
fizerem sentido para o tipo da variável, evitando gráficos
desnecessários para todas as colunas indiscriminadamente. Este
módulo:
- reaproveita os resultados já calculados por analysis/ (não recalcula
  estatísticas — só visualiza o que já foi analisado);
- pula colunas sem valor visual (constantes, "id", "text");
- limita o número de colunas plotadas em datasets muito largos
  (config.max_charted_columns_per_type), para não gerar dezenas de
  imagens irrelevantes.

Cada função de plotagem salva um arquivo .png em `output_dir` e
retorna o Path do arquivo (ou None se o gráfico não fizer sentido
para aquele dado específico) — report/builder.py consome esses paths
para montar os links de imagem no Markdown.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend não interativo: necessário em ambiente sem display

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from pathlib import Path
from typing import Any

from autoeda.config import AutoEDAConfig

_FIGURE_DPI = 110
_FIGSIZE_SMALL = (6, 4)
_FIGSIZE_WIDE = (8, 5)


def _save_figure(fig: Figure, output_dir: Path, filename: str) -> Path:
    """Salva a figura em `output_dir/filename` e libera a memória do
    Matplotlib (plt.close) — necessário porque, em um relatório com
    muitos gráficos, figuras não fechadas se acumulam na memória.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=_FIGURE_DPI)
    plt.close(fig)
    return path


def plot_missing_values_chart(
    missing_result: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Gráfico de barras horizontais com o percentual de valores
    ausentes por coluna (só as colunas com ao menos 1 valor ausente).

    Retorna None se o dataset não tiver nenhum valor ausente — não
    faz sentido gerar um gráfico vazio.
    """
    columns_info = missing_result.get("columns", {})
    if not columns_info:
        return None

    items = sorted(columns_info.items(), key=lambda kv: kv[1]["missing_pct"])
    labels = [name for name, _ in items]
    values = [info["missing_pct"] * 100 for _, info in items]

    fig, ax = plt.subplots(figsize=(_FIGSIZE_WIDE[0], max(3, 0.4 * len(labels))))
    ax.barh(labels, values, color="#c0392b")
    ax.set_xlabel("% de valores ausentes")
    ax.set_title("Valores ausentes por coluna")

    return _save_figure(fig, output_dir, "missing_values.png")


def plot_correlation_heatmap(
    correlation_result: dict[str, Any],
    output_dir: Path,
) -> Path | None:
    """Heatmap da matriz de correlação de Pearson entre as variáveis
    numéricas.

    Retorna None se houver menos de 2 colunas numéricas (correlação
    não definida — ver analysis.correlation.analyze_correlation).
    """
    pearson = correlation_result.get("pearson", {})
    if not pearson:
        return None

    corr_df = pd.DataFrame(pearson)
    columns = corr_df.columns.tolist()

    fig, ax = plt.subplots(figsize=(max(5, 0.6 * len(columns)), max(4, 0.6 * len(columns))))
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha="right")
    ax.set_yticklabels(columns)

    for i in range(len(columns)):
        for j in range(len(columns)):
            ax.text(j, i, f"{corr_df.values[i, j]:.2f}", ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, label="Correlação de Pearson")
    ax.set_title("Matriz de correlação (Pearson)")

    return _save_figure(fig, output_dir, "correlation_heatmap.png")


def plot_target_distribution(
    target_result: dict[str, Any],
    output_dir: Path,
) -> Path:
    """Gráfico de barras com a distribuição das 2 classes do target
    (contagem e percentual em cada barra).

    Sempre gerado (o target é obrigatório no AutoEDA — ver escopo de
    classificação binária), diferente dos demais gráficos que podem
    retornar None quando não fazem sentido para o dataset.
    """
    distribution = target_result["distribution"]
    class_info = distribution["class_distribution"]

    labels = [item["value"] for item in class_info]
    counts = [item["count"] for item in class_info]
    pcts = [item["pct"] for item in class_info]

    fig, ax = plt.subplots(figsize=_FIGSIZE_SMALL)
    bars = ax.bar(labels, counts, color=["#2980b9", "#e67e22"])
    for bar, pct in zip(bars, pcts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{pct:.1%}",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Contagem")
    ax.set_title(f"Distribuição do target '{target_result['target']}'")

    return _save_figure(fig, output_dir, "target_distribution.png")


def plot_numeric_histogram(df: pd.DataFrame, column: str, output_dir: Path) -> Path | None:
    """Histograma de uma coluna numérica, para visualizar forma da
    distribuição (assimetria, multimodalidade) — complementa as
    estatísticas de skewness/kurtosis já calculadas em descriptive.py.

    Retorna None se a coluna não tiver nenhum valor não nulo.
    """
    non_null = df[column].dropna()
    if non_null.empty:
        return None

    fig, ax = plt.subplots(figsize=_FIGSIZE_SMALL)
    ax.hist(non_null, bins=min(30, max(10, non_null.nunique())), color="#2980b9", edgecolor="white")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequência")
    ax.set_title(f"Distribuição de '{column}'")

    safe_name = column.replace("/", "_").replace(" ", "_")
    return _save_figure(fig, output_dir, f"hist_{safe_name}.png")


def plot_numeric_boxplot(df: pd.DataFrame, column: str, output_dir: Path) -> Path | None:
    """Boxplot de uma coluna numérica, para visualizar outliers
    (consistente com o método IQR usado em analysis.outliers).

    Retorna None se a coluna não tiver nenhum valor não nulo.
    """
    non_null = df[column].dropna()
    if non_null.empty:
        return None

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(non_null, vert=True, patch_artist=True, boxprops={"facecolor": "#85c1e9"})
    ax.set_ylabel(column)
    ax.set_title(f"Boxplot de '{column}'")
    ax.set_xticks([])

    safe_name = column.replace("/", "_").replace(" ", "_")
    return _save_figure(fig, output_dir, f"boxplot_{safe_name}.png")


def plot_categorical_barplot(
    df: pd.DataFrame,
    column: str,
    output_dir: Path,
    max_categories: int,
) -> Path | None:
    """Gráfico de barras com a frequência de cada categoria de uma
    coluna categórica.

    Se houver mais categorias do que `max_categories`, as menos
    frequentes são agrupadas em uma barra "outras" — mantém o gráfico
    legível sem esconder a existência de categorias adicionais.

    Retorna None se a coluna não tiver nenhum valor não nulo.
    """
    counts = df[column].dropna().value_counts()
    if counts.empty:
        return None

    if counts.shape[0] > max_categories:
        top = counts.head(max_categories - 1)
        others_count = counts.iloc[max_categories - 1:].sum()
        counts = pd.concat([top, pd.Series({"outras": others_count})])

    fig, ax = plt.subplots(figsize=(max(5, 0.5 * counts.shape[0]), 4))
    ax.bar([str(v) for v in counts.index], counts.values, color="#8e44ad")
    ax.set_ylabel("Contagem")
    ax.set_title(f"Distribuição de '{column}'")
    ax.tick_params(axis="x", rotation=45)

    safe_name = column.replace("/", "_").replace(" ", "_")
    return _save_figure(fig, output_dir, f"barplot_{safe_name}.png")


def generate_all_charts(
    df: pd.DataFrame,
    config: AutoEDAConfig,
    descriptive_result: dict[str, Any],
    missing_result: dict[str, Any],
    correlation_result: dict[str, Any],
    target_result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Orquestra a geração de todos os gráficos do relatório,
    decidindo quais colunas recebem gráfico individual.

    Critério de seleção: colunas do tipo "numeric"/"categorical" (ver
    utils.infer_column_types), exceto a própria coluna do target (que
    já tem um gráfico dedicado em plot_target_distribution) e as
    marcadas como constantes
    (descriptive_result["constant_and_near_zero_variance"] com
    constant=True — um histograma/barplot de uma coluna de valor
    único não agrega nada). Limitado a
    config.max_charted_columns_per_type colunas por tipo, para não
    gerar dezenas de imagens em datasets muito largos.

    Retorna um dict no formato:
    {
        "missing_values": Path | None,
        "correlation_heatmap": Path | None,
        "target_distribution": Path,
        "numeric": {"<coluna>": {"histogram": Path, "boxplot": Path}, ...},
        "categorical": {"<coluna>": Path, ...},
        "omitted_columns": {"numeric": [...], "categorical": [...]},
    }
    """
    output_path = Path(output_dir)
    target_column = target_result["target"]
    constant_columns = {
        column
        for column, info in descriptive_result.get("constant_and_near_zero_variance", {}).items()
        if info["constant"]
    }

    numeric_columns = [
        column
        for column, report in descriptive_result.get("columns", {}).items()
        if report["type"] == "numeric" and column not in constant_columns and column != target_column
    ]
    categorical_columns = [
        column
        for column, report in descriptive_result.get("columns", {}).items()
        if report["type"] == "categorical"
        and column not in constant_columns
        and column != target_column
    ]

    charted_numeric = numeric_columns[: config.max_charted_columns_per_type]
    charted_categorical = categorical_columns[: config.max_charted_columns_per_type]

    numeric_charts: dict[str, dict[str, Path]] = {}
    for column in charted_numeric:
        histogram = plot_numeric_histogram(df, column, output_path)
        boxplot = plot_numeric_boxplot(df, column, output_path)
        if histogram or boxplot:
            numeric_charts[column] = {"histogram": histogram, "boxplot": boxplot}

    categorical_charts: dict[str, Path] = {}
    for column in charted_categorical:
        chart = plot_categorical_barplot(df, column, output_path, config.max_categories_in_barplot)
        if chart:
            categorical_charts[column] = chart

    return {
        "missing_values": plot_missing_values_chart(missing_result, output_path),
        "correlation_heatmap": plot_correlation_heatmap(correlation_result, output_path),
        "target_distribution": plot_target_distribution(target_result, output_path),
        "numeric": numeric_charts,
        "categorical": categorical_charts,
        "omitted_columns": {
            "numeric": numeric_columns[config.max_charted_columns_per_type:],
            "categorical": categorical_columns[config.max_charted_columns_per_type:],
        },
    }
