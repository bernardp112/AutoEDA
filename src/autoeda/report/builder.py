"""
Montagem do relatório final do AutoEDA em Markdown.

Este é o módulo "de fecho": não recalcula nada, apenas formata os
resultados já produzidos por analysis/, recommendations.py e
report/charts.py em um documento .md legível, com as imagens
embutidas via link relativo.

Cada seção do relatório é montada por uma função própria (uma função
por seção), para poder ser testada isoladamente e para manter
build_markdown_report um simples "esqueleto" que as concatena na
ordem certa.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _format_pct(value: float | None, decimals: int = 1) -> str:
    """Formata um valor 0-1 como percentual, tolerando None (retorna
    "N/D" — "não disponível" — em vez de quebrar a montagem do relatório)."""
    if value is None:
        return "N/D"
    return f"{value:.{decimals}%}"


def _format_number(value: Any, decimals: int = 2) -> str:
    """Formata um número com casas decimais fixas, tolerando None e
    valores não numéricos (ex.: "infinito" já formatado como string
    em recommendations.py)."""
    if value is None:
        return "N/D"
    if isinstance(value, (int, float)):
        if value == float("inf"):
            return "∞"
        return f"{value:.{decimals}f}"
    return str(value)


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    """Monta uma tabela em Markdown a partir de cabeçalhos e linhas
    já formatadas como string. Retorna uma nota "sem dados" em vez de
    uma tabela vazia (cabeçalho sem linhas confunde mais do que ajuda)."""
    if not rows:
        return "_Sem dados para esta seção._\n"

    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_lines = ["| " + " | ".join(row) + " |" for row in rows]

    return "\n".join([header_line, separator_line, *body_lines]) + "\n"


def _relative_image_link(image_path: Path | None, report_dir: Path, alt_text: str) -> str:
    """Monta um link Markdown de imagem com caminho relativo ao
    diretório do relatório, para o .md continuar funcionando se a
    pasta inteira for movida/copiada. Retorna string vazia se
    `image_path` for None (gráfico não gerado para este dado)."""
    if image_path is None:
        return ""
    relative_path = os.path.relpath(image_path, start=report_dir)
    return f"![{alt_text}]({relative_path})\n"


def build_overview_section(descriptive_result: dict[str, Any]) -> str:
    """Seção 1: visão geral do dataset (dimensões, memória,
    duplicatas, % de nulos) — a partir de
    analysis.descriptive.generate_dataset_overview."""
    overview = descriptive_result["overview"]

    rows = [
        ["Linhas", f"{overview['n_rows']:,}".replace(",", ".")],
        ["Colunas", str(overview["n_columns"])],
        ["Uso de memória", f"{overview['memory_usage_bytes'] / 1024:.1f} KB"],
        [
            "Linhas duplicadas",
            f"{overview['duplicate_rows']} ({_format_pct(overview['duplicate_rows_pct'])})",
        ],
        [
            "Células ausentes (total)",
            f"{overview['total_missing_cells']} ({_format_pct(overview['total_missing_pct'])})",
        ],
    ]

    return "## 1. Visão geral do dataset\n\n" + _render_table(["Métrica", "Valor"], rows)


def build_target_section(target_result: dict[str, Any], chart_path: Path | None, report_dir: Path) -> str:
    """Seção 2: distribuição da variável alvo (classe majoritária/
    minoritária, imbalance ratio) — a partir de
    analysis.target_analysis.analyze_target."""
    distribution = target_result["distribution"]
    target = target_result["target"]

    lines = [f"## 2. Variável alvo: '{target}'\n"]
    lines.append(_relative_image_link(chart_path, report_dir, f"Distribuição de {target}"))

    rows = [
        ["Classe majoritária", f"{distribution['majority_class']} ({_format_pct(distribution['majority_pct'])})"],
        ["Classe minoritária", f"{distribution['minority_class']} ({_format_pct(distribution['minority_pct'])})"],
        ["Imbalance ratio", _format_number(distribution["imbalance_ratio"])],
    ]
    lines.append(_render_table(["Métrica", "Valor"], rows))

    if distribution["imbalance_ratio"] is not None and distribution["imbalance_ratio"] >= 3:
        lines.append(
            "> ⚠️ Classes desbalanceadas — veja a recomendação correspondente na seção 8.\n"
        )

    return "\n".join(lines)


def build_missing_values_section(missing_result: dict[str, Any], chart_path: Path | None, report_dir: Path) -> str:
    """Seção 3: valores ausentes por coluna, com severidade e o
    indício de mecanismo (MCAR/MAR/MNAR), sempre em linguagem
    cautelosa — nunca uma afirmação categórica sobre o mecanismo real."""
    lines = ["## 3. Valores ausentes\n"]

    if not missing_result["has_missing_values"]:
        lines.append("Nenhum valor ausente encontrado no dataset.\n")
        return "\n".join(lines)

    lines.append(_relative_image_link(chart_path, report_dir, "Valores ausentes por coluna"))

    rows = []
    for column, info in sorted(
        missing_result["columns"].items(), key=lambda kv: kv[1]["missing_pct"], reverse=True
    ):
        rows.append(
            [
                column,
                _format_pct(info["missing_pct"]),
                info["severity"],
                info["mechanism_hint"]["hint"],
            ]
        )
    lines.append(_render_table(["Coluna", "% ausente", "Severidade", "Indício de mecanismo"], rows))

    lines.append(
        "> Nota: MCAR, MAR e MNAR não podem ser determinados com certeza apenas a "
        "partir dos dados. \"MAR\" acima indica indício (ausência correlacionada "
        "com outra coluna, ou com as classes do target); \"indeterminado\" não "
        "confirma MCAR — MNAR nunca pode ser descartado só pelos dados observados.\n"
    )

    return "\n".join(lines)


def build_descriptive_section(descriptive_result: dict[str, Any], outliers_result: dict[str, Any], target: str) -> str:
    """Seção 4: estatísticas descritivas por coluna — tabela separada
    para numéricas (incluindo IQR e % de outliers, cruzando com
    analysis.outliers) e para categóricas, além de constantes/
    quase-constantes e colunas de tipo misto."""
    lines = ["## 4. Estatísticas descritivas\n"]

    outlier_pct_by_column = {
        column: stats["pct"] for column, stats in outliers_result.get("columns", {}).items()
    }

    numeric_rows = []
    categorical_rows = []
    for column, report in descriptive_result["columns"].items():
        if column == target:
            continue
        stats = report["stats"]
        if report["type"] == "numeric":
            numeric_rows.append(
                [
                    column,
                    _format_number(stats["mean"]),
                    _format_number(stats["median"]),
                    _format_number(stats["std"]),
                    _format_number(stats["min"]),
                    _format_number(stats["max"]),
                    _format_number(stats["iqr"]),
                    _format_number(stats["skewness"]),
                    _format_pct(outlier_pct_by_column.get(column)) if column in outlier_pct_by_column else "N/D",
                ]
            )
        elif report["type"] in ("categorical", "boolean"):
            categorical_rows.append(
                [
                    column,
                    str(stats.get("unique", "N/D")),
                    str(stats.get("mode", "N/D")),
                    _format_pct(stats.get("mode_pct")),
                    _format_pct(stats.get("missing_pct")),
                ]
            )

    lines.append("### Variáveis numéricas\n")
    lines.append(
        _render_table(
            ["Coluna", "Média", "Mediana", "Desvio", "Mín", "Máx", "IQR", "Assimetria", "% Outliers"],
            numeric_rows,
        )
    )

    lines.append("\n### Variáveis categóricas\n")
    lines.append(
        _render_table(
            ["Coluna", "Categorias", "Categoria dominante", "% dominante", "% ausente"],
            categorical_rows,
        )
    )

    constant_and_nzv = descriptive_result.get("constant_and_near_zero_variance", {})
    if constant_and_nzv:
        lines.append("\n### Variáveis constantes / quase-constantes\n")
        rows = [
            [column, "constante" if info["constant"] else "quase-constante", _format_pct(info["top_value_pct"])]
            for column, info in constant_and_nzv.items()
        ]
        lines.append(_render_table(["Coluna", "Tipo", "% valor dominante"], rows))

    mixed_type = descriptive_result.get("mixed_type_columns", {})
    if mixed_type:
        lines.append("\n### Colunas com tipo misto\n")
        rows = [
            [column, _format_pct(info["numeric_pct"]), _format_pct(info["non_numeric_pct"])]
            for column, info in mixed_type.items()
        ]
        lines.append(_render_table(["Coluna", "% numérico", "% não numérico"], rows))

    return "\n".join(lines)


def build_outliers_section(
    outliers_result: dict[str, Any],
    charts: dict[str, Any],
    report_dir: Path,
) -> str:
    """Seção 5: outliers detectados (método IQR por padrão), com
    boxplot e histograma por coluna. Reforça que outliers não são
    removidos automaticamente."""
    lines = ["## 5. Outliers\n"]
    columns_with_outliers = {
        column: stats for column, stats in outliers_result.get("columns", {}).items() if stats["count"] > 0
    }

    if not columns_with_outliers:
        lines.append("Nenhum outlier relevante detectado nas variáveis numéricas.\n")
        return "\n".join(lines)

    lines.append(
        f"Método: {outliers_result['method'].upper()}. Outliers não são removidos "
        "automaticamente — podem representar informação legítima do domínio.\n"
    )

    rows = [
        [column, str(stats["count"]), _format_pct(stats["pct"])]
        for column, stats in columns_with_outliers.items()
    ]
    lines.append(_render_table(["Coluna", "Qtd. outliers", "% outliers"], rows))

    for column in columns_with_outliers:
        column_charts = charts.get("numeric", {}).get(column)
        if not column_charts:
            continue
        lines.append(f"\n**{column}**\n")
        lines.append(_relative_image_link(column_charts.get("boxplot"), report_dir, f"Boxplot de {column}"))
        lines.append(_relative_image_link(column_charts.get("histogram"), report_dir, f"Histograma de {column}"))

    return "\n".join(lines)


def build_correlation_section(correlation_result: dict[str, Any], chart_path: Path | None, report_dir: Path) -> str:
    """Seção 6: correlação/multicolinearidade — pares fortemente
    correlacionados, VIF e disparidade de escala."""
    lines = ["## 6. Correlação e multicolinearidade\n"]

    if correlation_result.get("note"):
        lines.append(f"{correlation_result['note']}\n")
        return "\n".join(lines)

    lines.append(_relative_image_link(chart_path, report_dir, "Matriz de correlação"))

    high_corr = correlation_result.get("high_correlations", [])
    if high_corr:
        lines.append("\n### Pares fortemente correlacionados\n")
        rows = [
            [pair["column_a"], pair["column_b"], pair["method"], _format_number(pair["correlation"])]
            for pair in high_corr
        ]
        lines.append(_render_table(["Coluna A", "Coluna B", "Método", "Correlação"], rows))

    high_vif = correlation_result.get("high_vif", [])
    if high_vif:
        lines.append("\n### VIF (Variance Inflation Factor) alto\n")
        rows = [[item["column"], _format_number(item["vif"])] for item in high_vif]
        lines.append(_render_table(["Coluna", "VIF"], rows))

    scale_disparity = correlation_result.get("scale_disparity")
    if scale_disparity:
        lines.append("\n### Disparidade de escala\n")
        lines.append(
            f"'{scale_disparity['largest_scale_column']}' está em escala "
            f"{scale_disparity['ratio']:.0f}x maior que "
            f"'{scale_disparity['smallest_scale_column']}'. Considerar padronização.\n"
        )

    return "\n".join(lines)


def build_target_relationship_section(target_result: dict[str, Any]) -> str:
    """Seção 7: relação de cada preditor com o target (Point-Biserial/
    Mutual Information, Qui-quadrado/V de Cramér ou Spearman conforme
    o tipo), incluindo os alertas de vazamento e múltiplas comparações."""
    lines = [f"## 7. Relação das variáveis com o target '{target_result['target']}'\n"]

    rows = []
    for predictor in target_result["predictors"]:
        metrics = predictor["metrics"]
        metric_display = metrics.get("correlation", metrics.get("cramers_v"))
        p_value = metrics.get("p_value")
        rows.append(
            [
                predictor["predictor"],
                predictor["predictor_type"],
                predictor["relationship"],
                _format_number(metric_display, 4),
                _format_number(p_value, 4) if p_value is not None else "N/D",
            ]
        )
    lines.append(_render_table(["Preditor", "Tipo", "Técnica", "Força", "p-valor"], rows))

    excluded = target_result.get("excluded_predictors", [])
    if excluded:
        lines.append("\n### Preditores excluídos da análise\n")
        rows = [[item["predictor"], item["type"], item["reason"]] for item in excluded]
        lines.append(_render_table(["Preditor", "Tipo", "Motivo"], rows))

    if target_result.get("possible_leakage"):
        lines.append("\n> ⚠️ **Possível vazamento de dados detectado** — veja a seção 8.\n")

    if target_result.get("multiple_comparisons_warning"):
        lines.append(f"\n> {target_result['multiple_comparisons_warning']}\n")

    return "\n".join(lines)


def build_recommendations_section(recommendations_result: dict[str, Any], json_relative_path: str | None) -> str:
    """Seção 8: recomendações de tratamento, agrupadas por severidade
    (alta primeiro). Cada recomendação segue o schema
    {feature, problem_type, metric, metric_value, severity,
    recommendation, explanation} definido em recommendations.py."""
    lines = ["## 8. Recomendações\n"]

    if json_relative_path:
        lines.append(f"Versão completa em JSON: [`{json_relative_path}`]({json_relative_path})\n")

    severity_labels = {"high": "Alta", "medium": "Média", "low": "Baixa"}
    for severity in ("high", "medium", "low"):
        items = [r for r in recommendations_result["recommendations"] if r["severity"] == severity]
        if not items:
            continue
        lines.append(f"\n### Severidade {severity_labels[severity]}\n")
        rows = [
            [
                item["feature"] or "_dataset_",
                item["problem_type"],
                item["recommendation"],
            ]
            for item in items
        ]
        lines.append(_render_table(["Feature", "Problema", "Recomendação"], rows))

    return "\n".join(lines)


def build_markdown_report(
    df_shape: tuple[int, int],
    descriptive_result: dict[str, Any],
    missing_result: dict[str, Any],
    outliers_result: dict[str, Any],
    correlation_result: dict[str, Any],
    target_result: dict[str, Any],
    recommendations_result: dict[str, Any],
    charts: dict[str, Any],
    report_path: str | Path,
    recommendations_json_path: str | Path | None = None,
) -> str:
    """Monta o relatório completo em Markdown, concatenando as 8
    seções na ordem: visão geral, target, ausentes, descritivas,
    outliers, correlação, relação com target, recomendações.

    `report_path` é o caminho onde o .md será salvo — usado para
    calcular os links de imagem relativos (não escreve o arquivo;
    ver export_markdown_report para isso).
    """
    report_dir = Path(report_path).parent
    target = target_result["target"]

    json_relative_path = None
    if recommendations_json_path is not None:
        json_relative_path = os.path.relpath(Path(recommendations_json_path), start=report_dir)

    sections = [
        f"# Relatório AutoEDA\n\nDataset: {df_shape[0]} linhas × {df_shape[1]} colunas. "
        f"Variável alvo: '{target}' (classificação binária).\n",
        build_overview_section(descriptive_result),
        build_target_section(target_result, charts.get("target_distribution"), report_dir),
        build_missing_values_section(missing_result, charts.get("missing_values"), report_dir),
        build_descriptive_section(descriptive_result, outliers_result, target),
        build_outliers_section(outliers_result, charts, report_dir),
        build_correlation_section(correlation_result, charts.get("correlation_heatmap"), report_dir),
        build_target_relationship_section(target_result),
        build_recommendations_section(recommendations_result, json_relative_path),
    ]

    return "\n\n".join(sections) + "\n"


def export_markdown_report(
    df_shape: tuple[int, int],
    descriptive_result: dict[str, Any],
    missing_result: dict[str, Any],
    outliers_result: dict[str, Any],
    correlation_result: dict[str, Any],
    target_result: dict[str, Any],
    recommendations_result: dict[str, Any],
    charts: dict[str, Any],
    report_path: str | Path,
    recommendations_json_path: str | Path | None = None,
) -> Path:
    """Monta o relatório (build_markdown_report) e escreve em
    `report_path`, criando diretórios intermediários se necessário.
    Retorna o Path final (resolvido)."""
    content = build_markdown_report(
        df_shape,
        descriptive_result,
        missing_result,
        outliers_result,
        correlation_result,
        target_result,
        recommendations_result,
        charts,
        report_path,
        recommendations_json_path,
    )

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return path.resolve()
