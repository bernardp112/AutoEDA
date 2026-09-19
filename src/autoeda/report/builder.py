"""
Montagem do relatório final do AutoEDA em Markdown.

Este é o módulo "de fecho": não recalcula nada, apenas formata os
resultados já produzidos por analysis/, recommendations.py e
report/charts.py em um documento .md legível, com as imagens
embutidas via link relativo.

Todo o texto fixo (títulos de seção, cabeçalhos de tabela, frases
estáticas) vem de autoeda.i18n, no idioma de config.language — o
mesmo idioma já usado para gerar o conteúdo de recommendations.py e
os indícios de mecanismo de ausência.

Cada seção do relatório é montada por uma função própria (uma função
por seção), para poder ser testada isoladamente e para manter
build_markdown_report um simples "esqueleto" que as concatena na
ordem certa.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from autoeda.config import AutoEDAConfig
from autoeda.i18n import get_translator


def _format_pct(value: float | None, t, decimals: int = 1) -> str:
    """Formata um valor 0-1 como percentual, tolerando None (retorna
    o texto de "não disponível" traduzido em vez de quebrar a
    montagem do relatório)."""
    if value is None:
        return t("report.table.na")
    return f"{value:.{decimals}%}"


def _format_number(value: Any, t, decimals: int = 2) -> str:
    """Formata um número com casas decimais fixas, tolerando None e
    valores não numéricos (ex.: "∞" já formatado como string em
    recommendations.py)."""
    if value is None:
        return t("report.table.na")
    if isinstance(value, (int, float)):
        if value == float("inf"):
            return "∞"
        return f"{value:.{decimals}f}"
    return str(value)


def _render_table(headers: list[str], rows: list[list[str]], t) -> str:
    """Monta uma tabela em Markdown a partir de cabeçalhos e linhas
    já formatadas como string. Retorna uma nota "sem dados" (traduzida)
    em vez de uma tabela vazia (cabeçalho sem linhas confunde mais do
    que ajuda)."""
    if not rows:
        return t("report.no_data") + "\n"

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


def build_overview_section(descriptive_result: dict[str, Any], config: AutoEDAConfig) -> str:
    """Seção 1: visão geral do dataset (dimensões, memória,
    duplicatas, % de nulos) — a partir de
    analysis.descriptive.generate_dataset_overview."""
    t = get_translator(config.language)
    overview = descriptive_result["overview"]

    rows = [
        [t("report.section1.rows"), f"{overview['n_rows']:,}".replace(",", ".")],
        [t("report.section1.columns"), str(overview["n_columns"])],
        [t("report.section1.memory"), f"{overview['memory_usage_bytes'] / 1024:.1f} KB"],
        [
            t("report.section1.duplicate_rows"),
            f"{overview['duplicate_rows']} ({_format_pct(overview['duplicate_rows_pct'], t)})",
        ],
        [
            t("report.section1.missing_cells"),
            f"{overview['total_missing_cells']} ({_format_pct(overview['total_missing_pct'], t)})",
        ],
    ]

    return f"## {t('report.section1.title')}\n\n" + _render_table(
        [t("report.table.metric"), t("report.table.value")], rows, t
    )


def build_target_section(
    target_result: dict[str, Any], chart_path: Path | None, report_dir: Path, config: AutoEDAConfig
) -> str:
    """Seção 2: distribuição da variável alvo (classe majoritária/
    minoritária, imbalance ratio) — a partir de
    analysis.target_analysis.analyze_target."""
    t = get_translator(config.language)
    distribution = target_result["distribution"]
    target = target_result["target"]

    lines = [f"## {t('report.section2.title', target=target)}\n"]
    lines.append(_relative_image_link(chart_path, report_dir, f"{t('report.section2.title', target=target)}"))

    rows = [
        [
            t("report.section2.majority_class"),
            f"{distribution['majority_class']} ({_format_pct(distribution['majority_pct'], t)})",
        ],
        [
            t("report.section2.minority_class"),
            f"{distribution['minority_class']} ({_format_pct(distribution['minority_pct'], t)})",
        ],
        [t("report.section2.imbalance_ratio"), _format_number(distribution["imbalance_ratio"], t)],
    ]
    lines.append(_render_table([t("report.table.metric"), t("report.table.value")], rows, t))

    if distribution["imbalance_ratio"] is not None and distribution["imbalance_ratio"] >= 3:
        lines.append(f"> {t('report.section2.imbalance_warning')}\n")

    return "\n".join(lines)


def build_missing_values_section(
    missing_result: dict[str, Any], chart_path: Path | None, report_dir: Path, config: AutoEDAConfig
) -> str:
    """Seção 3: valores ausentes por coluna, com severidade e o
    indício de mecanismo (MCAR/MAR/MNAR), sempre em linguagem
    cautelosa — nunca uma afirmação categórica sobre o mecanismo real."""
    t = get_translator(config.language)
    lines = [f"## {t('report.section3.title')}\n"]

    if not missing_result["has_missing_values"]:
        lines.append(t("report.section3.no_missing") + "\n")
        return "\n".join(lines)

    lines.append(_relative_image_link(chart_path, report_dir, t("report.section3.title")))

    rows = []
    for column, info in sorted(
        missing_result["columns"].items(), key=lambda kv: kv[1]["missing_pct"], reverse=True
    ):
        rows.append(
            [
                column,
                _format_pct(info["missing_pct"], t),
                info["severity"],
                info["mechanism_hint"]["hint"],
            ]
        )
    lines.append(
        _render_table(
            [
                t("report.section3.table.column"),
                t("report.section3.table.pct_missing"),
                t("report.section3.table.severity"),
                t("report.section3.table.mechanism_hint"),
            ],
            rows,
            t,
        )
    )

    lines.append(f"> {t('report.section3.mechanism_note')}\n")

    return "\n".join(lines)


def build_descriptive_section(
    descriptive_result: dict[str, Any], outliers_result: dict[str, Any], target: str, config: AutoEDAConfig
) -> str:
    """Seção 4: estatísticas descritivas por coluna — tabela separada
    para numéricas (incluindo IQR e % de outliers, cruzando com
    analysis.outliers) e para categóricas, além de constantes/
    quase-constantes e colunas de tipo misto."""
    t = get_translator(config.language)
    lines = [f"## {t('report.section4.title')}\n"]

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
                    _format_number(stats["mean"], t),
                    _format_number(stats["median"], t),
                    _format_number(stats["std"], t),
                    _format_number(stats["min"], t),
                    _format_number(stats["max"], t),
                    _format_number(stats["iqr"], t),
                    _format_number(stats["skewness"], t),
                    _format_pct(outlier_pct_by_column.get(column), t) if column in outlier_pct_by_column else t("report.table.na"),
                ]
            )
        elif report["type"] in ("categorical", "boolean"):
            categorical_rows.append(
                [
                    column,
                    str(stats.get("unique", t("report.table.na"))),
                    str(stats.get("mode", t("report.table.na"))),
                    _format_pct(stats.get("mode_pct"), t),
                    _format_pct(stats.get("missing_pct"), t),
                ]
            )

    lines.append(f"### {t('report.section4.numeric_subtitle')}\n")
    lines.append(
        _render_table(
            [
                t("report.section4.table.column"),
                t("report.section4.table.mean"),
                t("report.section4.table.median"),
                t("report.section4.table.std"),
                t("report.section4.table.min"),
                t("report.section4.table.max"),
                t("report.section4.table.iqr"),
                t("report.section4.table.skewness"),
                t("report.section4.table.pct_outliers"),
            ],
            numeric_rows,
            t,
        )
    )

    lines.append(f"\n### {t('report.section4.categorical_subtitle')}\n")
    lines.append(
        _render_table(
            [
                t("report.section4.table.column"),
                t("report.section4.table.categories"),
                t("report.section4.table.dominant_category"),
                t("report.section4.table.pct_dominant"),
                t("report.section4.table.pct_missing"),
            ],
            categorical_rows,
            t,
        )
    )

    constant_and_nzv = descriptive_result.get("constant_and_near_zero_variance", {})
    if constant_and_nzv:
        lines.append(f"\n### {t('report.section4.constant_subtitle')}\n")
        rows = [
            [
                column,
                t("report.section4.constant_label") if info["constant"] else t("report.section4.near_constant_label"),
                _format_pct(info["top_value_pct"], t),
            ]
            for column, info in constant_and_nzv.items()
        ]
        lines.append(
            _render_table(
                [
                    t("report.section4.table.column"),
                    t("report.section4.table.type"),
                    t("report.section4.table.pct_dominant_value"),
                ],
                rows,
                t,
            )
        )

    mixed_type = descriptive_result.get("mixed_type_columns", {})
    if mixed_type:
        lines.append(f"\n### {t('report.section4.mixed_type_subtitle')}\n")
        rows = [
            [column, _format_pct(info["numeric_pct"], t), _format_pct(info["non_numeric_pct"], t)]
            for column, info in mixed_type.items()
        ]
        lines.append(
            _render_table(
                [
                    t("report.section4.table.column"),
                    t("report.section4.table.pct_numeric"),
                    t("report.section4.table.pct_non_numeric"),
                ],
                rows,
                t,
            )
        )

    return "\n".join(lines)


def build_outliers_section(
    outliers_result: dict[str, Any],
    charts: dict[str, Any],
    report_dir: Path,
    config: AutoEDAConfig,
) -> str:
    """Seção 5: outliers detectados (método IQR por padrão), com
    boxplot e histograma por coluna. Reforça que outliers não são
    removidos automaticamente."""
    t = get_translator(config.language)
    lines = [f"## {t('report.section5.title')}\n"]
    columns_with_outliers = {
        column: stats for column, stats in outliers_result.get("columns", {}).items() if stats["count"] > 0
    }

    if not columns_with_outliers:
        lines.append(t("report.section5.no_outliers") + "\n")
        return "\n".join(lines)

    lines.append(t("report.section5.method_note", method=outliers_result["method"].upper()) + "\n")

    rows = [
        [column, str(stats["count"]), _format_pct(stats["pct"], t)]
        for column, stats in columns_with_outliers.items()
    ]
    lines.append(
        _render_table(
            [t("report.section5.table.column"), t("report.section5.table.count"), t("report.section5.table.pct")],
            rows,
            t,
        )
    )

    for column in columns_with_outliers:
        column_charts = charts.get("numeric", {}).get(column)
        if not column_charts:
            continue
        lines.append(f"\n**{column}**\n")
        lines.append(_relative_image_link(column_charts.get("boxplot"), report_dir, f"Boxplot - {column}"))
        lines.append(_relative_image_link(column_charts.get("histogram"), report_dir, f"Histogram - {column}"))

    return "\n".join(lines)


def build_correlation_section(
    correlation_result: dict[str, Any], chart_path: Path | None, report_dir: Path, config: AutoEDAConfig
) -> str:
    """Seção 6: correlação/multicolinearidade — pares fortemente
    correlacionados, VIF e disparidade de escala."""
    t = get_translator(config.language)
    lines = [f"## {t('report.section6.title')}\n"]

    if correlation_result.get("note"):
        lines.append(f"{correlation_result['note']}\n")
        return "\n".join(lines)

    lines.append(_relative_image_link(chart_path, report_dir, t("report.section6.title")))

    high_corr = correlation_result.get("high_correlations", [])
    if high_corr:
        lines.append(f"\n### {t('report.section6.strong_corr_subtitle')}\n")
        rows = [
            [pair["column_a"], pair["column_b"], pair["method"], _format_number(pair["correlation"], t)]
            for pair in high_corr
        ]
        lines.append(
            _render_table(
                [
                    t("report.section6.table.column_a"),
                    t("report.section6.table.column_b"),
                    t("report.section6.table.method"),
                    t("report.section6.table.correlation"),
                ],
                rows,
                t,
            )
        )

    high_vif = correlation_result.get("high_vif", [])
    if high_vif:
        lines.append(f"\n### {t('report.section6.vif_subtitle')}\n")
        rows = [[item["column"], _format_number(item["vif"], t)] for item in high_vif]
        lines.append(_render_table([t("report.section6.table.column"), t("report.section6.table.vif")], rows, t))

    scale_disparity = correlation_result.get("scale_disparity")
    if scale_disparity:
        lines.append(f"\n### {t('report.section6.scale_subtitle')}\n")
        lines.append(
            t(
                "report.section6.scale_text",
                largest_column=scale_disparity["largest_scale_column"],
                smallest_column=scale_disparity["smallest_scale_column"],
                ratio=scale_disparity["ratio"],
            )
            + "\n"
        )

    return "\n".join(lines)


def build_target_relationship_section(target_result: dict[str, Any], config: AutoEDAConfig) -> str:
    """Seção 7: relação de cada preditor com o target (Point-Biserial/
    Mutual Information, Qui-quadrado/V de Cramér ou Spearman conforme
    o tipo), incluindo os alertas de vazamento e múltiplas comparações."""
    t = get_translator(config.language)
    lines = [f"## {t('report.section7.title', target=target_result['target'])}\n"]

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
                _format_number(metric_display, t, 4),
                _format_number(p_value, t, 4) if p_value is not None else t("report.table.na"),
            ]
        )
    lines.append(
        _render_table(
            [
                t("report.section7.table.predictor"),
                t("report.section7.table.type"),
                t("report.section7.table.technique"),
                t("report.section7.table.strength"),
                t("report.section7.table.p_value"),
            ],
            rows,
            t,
        )
    )

    excluded = target_result.get("excluded_predictors", [])
    if excluded:
        lines.append(f"\n### {t('report.section7.excluded_subtitle')}\n")
        rows = [[item["predictor"], item["type"], item["reason"]] for item in excluded]
        lines.append(
            _render_table(
                [
                    t("report.section7.table.predictor"),
                    t("report.section7.table.type"),
                    t("report.section7.table.reason"),
                ],
                rows,
                t,
            )
        )

    if target_result.get("possible_leakage"):
        lines.append(f"\n> {t('report.section7.leakage_warning')}\n")

    if target_result.get("multiple_comparisons_warning"):
        lines.append(f"\n> {target_result['multiple_comparisons_warning']}\n")

    return "\n".join(lines)


def build_recommendations_section(
    recommendations_result: dict[str, Any], json_relative_path: str | None, config: AutoEDAConfig
) -> str:
    """Seção 8: recomendações de tratamento, agrupadas por severidade
    (alta primeiro). Cada recomendação segue o schema
    {feature, problem_type, metric, metric_value, severity,
    recommendation, explanation} definido em recommendations.py."""
    t = get_translator(config.language)
    lines = [f"## {t('report.section8.title')}\n"]

    if json_relative_path:
        lines.append(f"{t('report.section8.json_link')} [`{json_relative_path}`]({json_relative_path})\n")

    severity_keys = {
        "high": "report.section8.severity_high",
        "medium": "report.section8.severity_medium",
        "low": "report.section8.severity_low",
    }
    for severity, label_key in severity_keys.items():
        items = [r for r in recommendations_result["recommendations"] if r["severity"] == severity]
        if not items:
            continue
        lines.append(f"\n### {t(label_key)}\n")
        rows = [
            [
                item["feature"] or t("report.section8.dataset_placeholder"),
                item["problem_type"],
                item["recommendation"],
            ]
            for item in items
        ]
        lines.append(
            _render_table(
                [
                    t("report.section8.table.feature"),
                    t("report.section8.table.problem"),
                    t("report.section8.table.recommendation"),
                ],
                rows,
                t,
            )
        )

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
    config: AutoEDAConfig,
    recommendations_json_path: str | Path | None = None,
) -> str:
    """Monta o relatório completo em Markdown, concatenando as 8
    seções na ordem: visão geral, target, ausentes, descritivas,
    outliers, correlação, relação com target, recomendações.

    Todo o texto fixo é gerado no idioma de config.language (pt-br ou
    en-us) via autoeda.i18n.

    `report_path` é o caminho onde o .md será salvo — usado para
    calcular os links de imagem relativos (não escreve o arquivo;
    ver export_markdown_report para isso).
    """
    t = get_translator(config.language)
    report_dir = Path(report_path).parent
    target = target_result["target"]

    json_relative_path = None
    if recommendations_json_path is not None:
        json_relative_path = os.path.relpath(Path(recommendations_json_path), start=report_dir)

    sections = [
        f"# {t('report.title')}\n\n{t('report.summary', rows=df_shape[0], cols=df_shape[1], target=target)}\n",
        build_overview_section(descriptive_result, config),
        build_target_section(target_result, charts.get("target_distribution"), report_dir, config),
        build_missing_values_section(missing_result, charts.get("missing_values"), report_dir, config),
        build_descriptive_section(descriptive_result, outliers_result, target, config),
        build_outliers_section(outliers_result, charts, report_dir, config),
        build_correlation_section(correlation_result, charts.get("correlation_heatmap"), report_dir, config),
        build_target_relationship_section(target_result, config),
        build_recommendations_section(recommendations_result, json_relative_path, config),
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
    config: AutoEDAConfig,
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
        config,
        recommendations_json_path,
    )

    path = Path(report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")

    return path.resolve()
