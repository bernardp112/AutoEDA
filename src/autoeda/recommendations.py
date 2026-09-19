"""
Geração das recomendações de transformação/pré-processamento do AutoEDA.

Este é o módulo que conecta os resultados de analysis/ (que apenas
descrevem e classificam os dados) a ações concretas que o usuário
pode aplicar antes de modelar. A saída daqui é o conteúdo do JSON de
recomendações citado no escopo do projeto — report/json_export.py
apenas serializa o que este módulo produz.

Cada recomendação segue o schema definido pelo orientador:
{
    "feature": str | None,      # coluna afetada, ou None para recomendação de dataset inteiro
    "problem_type": str,        # "missing_values" | "outliers" | "correlation" |
                                 # "descriptive" | "target" | "data_leakage"
    "metric": str | None,       # nome da métrica que motivou a recomendação
    "metric_value": Any | None, # valor observado dessa métrica
    "severity": str,            # "high" | "medium" | "low"
    "recommendation": str,      # técnica recomendada, em linguagem natural
    "explanation": str,         # evidência + justificativa da recomendação
}

Todo o texto de "recommendation"/"explanation" é gerado via
autoeda.i18n (config.language decide pt-br ou en-us) — nenhuma string
literal de conteúdo fica hardcoded aqui, só os templates em
i18n/pt_br.py e i18n/en_us.py.

Por design, "recommendation" nunca é uma instrução absoluta ("remova
a coluna"), e sim uma sugestão a avaliar ("considerar remover",
"avaliar remover") — decisões de preparação de dados dependem de
contexto e conhecimento de domínio que o AutoEDA não tem acesso.

As decisões de threshold (o que é "severo", o que é "alta
correlação") já foram tomadas pelos módulos de analysis/ a partir de
AutoEDAConfig; este módulo não reaplica lógica de threshold, apenas
traduz os resultados já classificados em recomendações de texto e
severidade.
"""

from __future__ import annotations

from typing import Any

from autoeda.config import RECOMMENDATIONS_SCHEMA_VERSION, AutoEDAConfig
from autoeda.i18n import get_translator


def _build_recommendation(
    feature: str | None,
    problem_type: str,
    metric: str | None,
    metric_value: Any,
    severity: str,
    recommendation: str,
    explanation: str,
) -> dict[str, Any]:
    """Monta uma recomendação no schema padrão, garantindo que todo
    recommend_from_* produza exatamente os mesmos campos.
    """
    return {
        "feature": feature,
        "problem_type": problem_type,
        "metric": metric,
        "metric_value": metric_value,
        "severity": severity,
        "recommendation": recommendation,
        "explanation": explanation,
    }


def _get_column_type(descriptive_result: dict[str, Any], column: str) -> str | None:
    """Busca o tipo lógico de uma coluna no resultado de
    analysis.descriptive.generate_descriptive_stats.
    """
    column_report = descriptive_result.get("columns", {}).get(column)
    return column_report["type"] if column_report else None


def recommend_from_missing_values(
    missing_result: dict[str, Any],
    descriptive_result: dict[str, Any],
    config: AutoEDAConfig,
) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.missing_values.analyze_missing_values.

    Regras:
    - severidade "high" -> sugerir remoção da coluna.
    - severidade "moderate"/"low" -> sugerir imputação (mediana para
      numéricas, moda para categóricas).
    - pares de colunas com ausência correlacionada -> recomendação de
      dataset inteiro sugerindo tratamento conjunto.
    - taxa de ausência diferente entre classes do target -> sugerir
      indicador binário de ausência como feature.
    """
    t = get_translator(config.language)
    recommendations: list[dict[str, Any]] = []

    for column, stats in missing_result.get("columns", {}).items():
        severity = stats["severity"]
        column_type = _get_column_type(descriptive_result, column)
        missing_pct_display = round(stats["missing_pct"] * 100, 2)
        pct_text = f"{stats['missing_pct']:.1%}"

        if severity == "high":
            recommendations.append(
                _build_recommendation(
                    feature=column,
                    problem_type="missing_values",
                    metric="missing_percentage",
                    metric_value=missing_pct_display,
                    severity="high",
                    recommendation=t("rec.missing_high.recommendation", column=column),
                    explanation=t("rec.missing_high.explanation", pct=pct_text),
                )
            )
            continue

        if column_type == "numeric":
            technique = t("rec.missing_numeric.recommendation", column=column)
            reason = t("rec.missing_numeric.reason")
        else:
            technique = t("rec.missing_categorical.recommendation", column=column)
            reason = t("rec.missing_categorical.reason")

        recommendations.append(
            _build_recommendation(
                feature=column,
                problem_type="missing_values",
                metric="missing_percentage",
                metric_value=missing_pct_display,
                severity="medium" if severity == "moderate" else "low",
                recommendation=technique,
                explanation=t("rec.missing_generic.explanation", pct=pct_text, reason=reason),
            )
        )

    correlated_pairs = missing_result.get("correlated_missingness", [])
    if correlated_pairs:
        top_pair = correlated_pairs[0]
        recommendations.append(
            _build_recommendation(
                feature=None,
                problem_type="missing_values",
                metric="missing_correlation",
                metric_value=round(top_pair["correlation"], 4),
                severity="medium",
                recommendation=t(
                    "rec.missing_correlated.recommendation",
                    column_a=top_pair["column_a"],
                    column_b=top_pair["column_b"],
                ),
                explanation=t(
                    "rec.missing_correlated.explanation",
                    column_a=top_pair["column_a"],
                    column_b=top_pair["column_b"],
                    corr=top_pair["correlation"],
                ),
            )
        )

    for item in missing_result.get("missing_target_association", []):
        rates_text = ", ".join(f"{cls}={rate:.1%}" for cls, rate in item["rates_by_class"].items())
        recommendations.append(
            _build_recommendation(
                feature=item["column"],
                problem_type="missing_values",
                metric="missing_rate_diff_by_target",
                metric_value=round(item["diff"] * 100, 2),
                severity="medium",
                recommendation=t("rec.missing_target_assoc.recommendation", column=item["column"]),
                explanation=t(
                    "rec.missing_target_assoc.explanation", column=item["column"], rates=rates_text
                ),
            )
        )

    return recommendations


def recommend_from_outliers(outliers_result: dict[str, Any], config: AutoEDAConfig) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.outliers.analyze_outliers.

    Outliers nunca são recomendados para remoção automática — apenas
    winsorização (percentuais baixos) ou investigação manual
    (percentuais altos, que podem indicar erro de coleta ou uma
    subpopulação legítima).
    """
    t = get_translator(config.language)
    recommendations: list[dict[str, Any]] = []
    method = outliers_result.get("method", "iqr")

    for column, stats in outliers_result.get("columns", {}).items():
        if stats["count"] == 0:
            continue

        pct_display = round(stats["pct"] * 100, 2)

        if stats["pct"] <= 0.05:
            technique = t("rec.outliers.recommendation_low", column=column)
            severity = "low"
            reason = t("rec.outliers.reason_low")
        else:
            technique = t("rec.outliers.recommendation_medium", column=column)
            severity = "medium"
            reason = t("rec.outliers.reason_medium")

        recommendations.append(
            _build_recommendation(
                feature=column,
                problem_type="outliers",
                metric="outlier_percentage",
                metric_value=pct_display,
                severity=severity,
                recommendation=technique,
                explanation=t(
                    "rec.outliers.explanation",
                    count=stats["count"],
                    pct=f"{stats['pct']:.1%}",
                    method=method,
                    reason=reason,
                ),
            )
        )

    return recommendations


def recommend_from_descriptive(descriptive_result: dict[str, Any], config: AutoEDAConfig) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.descriptive.generate_descriptive_stats.

    Cobre: linhas duplicadas, colunas "id", assimetria, variáveis
    constantes/quase-constantes, colunas de tipo misto e cardinalidade
    alta em categóricas (recomendação de encoding).
    """
    t = get_translator(config.language)
    recommendations: list[dict[str, Any]] = []

    overview = descriptive_result.get("overview", {})
    if overview.get("duplicate_rows", 0) > 0:
        recommendations.append(
            _build_recommendation(
                feature=None,
                problem_type="descriptive",
                metric="duplicate_rows_percentage",
                metric_value=round(overview["duplicate_rows_pct"] * 100, 2),
                severity="medium",
                recommendation=t("rec.duplicate_rows.recommendation"),
                explanation=t(
                    "rec.duplicate_rows.explanation",
                    count=overview["duplicate_rows"],
                    pct=f"{overview['duplicate_rows_pct']:.1%}",
                ),
            )
        )

    for column, column_report in descriptive_result.get("columns", {}).items():
        col_type = column_report["type"]
        stats = column_report["stats"]

        if col_type == "id":
            unique_ratio = (
                round(stats["unique"] / stats["count"] * 100, 2)
                if stats.get("count")
                else None
            )
            recommendations.append(
                _build_recommendation(
                    feature=column,
                    problem_type="descriptive",
                    metric="unique_ratio",
                    metric_value=unique_ratio,
                    severity="low",
                    recommendation=t("rec.id_column.recommendation", column=column),
                    explanation=t("rec.id_column.explanation"),
                )
            )
            continue

        if col_type == "numeric" and stats.get("skewness") is not None and abs(stats["skewness"]) > 1:
            all_positive = stats.get("min") is not None and stats["min"] > 0
            technique = (
                t("rec.skewed.recommendation_log", column=column)
                if all_positive
                else t("rec.skewed.recommendation_yeo", column=column)
            )
            recommendations.append(
                _build_recommendation(
                    feature=column,
                    problem_type="descriptive",
                    metric="skewness",
                    metric_value=round(stats["skewness"], 4),
                    severity="low",
                    recommendation=technique,
                    explanation=t("rec.skewed.explanation", skewness=stats["skewness"]),
                )
            )

        if col_type in ("categorical", "text"):
            unique_count = stats.get("unique")
            is_high_cardinality = col_type == "text" or (
                unique_count is not None and unique_count >= config.high_cardinality_warning_threshold
            )
            if is_high_cardinality:
                recommendations.append(
                    _build_recommendation(
                        feature=column,
                        problem_type="descriptive",
                        metric="cardinality",
                        metric_value=unique_count,
                        severity="low",
                        recommendation=t("rec.high_cardinality.recommendation", column=column),
                        explanation=t(
                            "rec.high_cardinality.explanation",
                            unique_count=unique_count if unique_count is not None else "muitas/many",
                        ),
                    )
                )

    for column, info in descriptive_result.get("constant_and_near_zero_variance", {}).items():
        if info["constant"]:
            recommendations.append(
                _build_recommendation(
                    feature=column,
                    problem_type="descriptive",
                    metric="top_value_percentage",
                    metric_value=round(info["top_value_pct"] * 100, 2),
                    severity="high",
                    recommendation=t("rec.constant.recommendation", column=column),
                    explanation=t("rec.constant.explanation"),
                )
            )
        else:
            recommendations.append(
                _build_recommendation(
                    feature=column,
                    problem_type="descriptive",
                    metric="top_value_percentage",
                    metric_value=round(info["top_value_pct"] * 100, 2),
                    severity="low",
                    recommendation=t("rec.near_zero_variance.recommendation", column=column),
                    explanation=t("rec.near_zero_variance.explanation", pct=f"{info['top_value_pct']:.1%}"),
                )
            )

    for column, info in descriptive_result.get("mixed_type_columns", {}).items():
        recommendations.append(
            _build_recommendation(
                feature=column,
                problem_type="descriptive",
                metric="non_numeric_percentage",
                metric_value=round(info["non_numeric_pct"] * 100, 2),
                severity="medium",
                recommendation=t("rec.mixed_type.recommendation", column=column),
                explanation=t(
                    "rec.mixed_type.explanation",
                    numeric_pct=f"{info['numeric_pct']:.1%}",
                    non_numeric_pct=f"{info['non_numeric_pct']:.1%}",
                ),
            )
        )

    return recommendations


def recommend_from_correlation(correlation_result: dict[str, Any], config: AutoEDAConfig) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.correlation.analyze_correlation.

    Cobre 3 sinais independentes: pares fortemente correlacionados
    (Pearson/Spearman), VIF alto (redundância multivariada) e
    disparidade de escala entre variáveis numéricas.
    """
    t = get_translator(config.language)
    recommendations: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for pair in correlation_result.get("high_correlations", []):
        key = tuple(sorted((pair["column_a"], pair["column_b"])))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        recommendations.append(
            _build_recommendation(
                feature=None,
                problem_type="correlation",
                metric=pair["method"],
                metric_value=round(pair["correlation"], 4),
                severity="medium",
                recommendation=t(
                    "rec.correlation_pair.recommendation",
                    column_a=pair["column_a"],
                    column_b=pair["column_b"],
                ),
                explanation=t(
                    "rec.correlation_pair.explanation",
                    column_a=pair["column_a"],
                    column_b=pair["column_b"],
                    method=pair["method"],
                    corr=pair["correlation"],
                ),
            )
        )

    for item in correlation_result.get("high_vif", []):
        vif_value = item["vif"]
        vif_display = "∞" if vif_value == float("inf") else round(vif_value, 2)
        recommendations.append(
            _build_recommendation(
                feature=item["column"],
                problem_type="correlation",
                metric="vif",
                metric_value=str(vif_display) if vif_display == "∞" else vif_display,
                severity="medium",
                recommendation=t("rec.vif.recommendation", column=item["column"]),
                explanation=t("rec.vif.explanation", vif=vif_display),
            )
        )

    scale_disparity = correlation_result.get("scale_disparity")
    if scale_disparity:
        recommendations.append(
            _build_recommendation(
                feature=None,
                problem_type="correlation",
                metric="scale_ratio",
                metric_value=round(scale_disparity["ratio"], 2),
                severity="low",
                recommendation=t("rec.scale_disparity.recommendation"),
                explanation=t(
                    "rec.scale_disparity.explanation",
                    largest_column=scale_disparity["largest_scale_column"],
                    largest_std=scale_disparity["largest_scale_std"],
                    smallest_column=scale_disparity["smallest_scale_column"],
                    smallest_std=scale_disparity["smallest_scale_std"],
                    ratio=scale_disparity["ratio"],
                ),
            )
        )

    return recommendations


def recommend_from_target(target_result: dict[str, Any], config: AutoEDAConfig) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.target_analysis.analyze_target (target binário).

    Cobre: desbalanceamento de classes, vazamento direto (associação
    suspeitosamente quase perfeita com o target), preditores fortes e
    aviso de múltiplas comparações.
    """
    t = get_translator(config.language)
    recommendations: list[dict[str, Any]] = []
    target = target_result["target"]
    distribution = target_result.get("distribution", {})

    imbalance_ratio = distribution.get("imbalance_ratio")
    if imbalance_ratio is not None and imbalance_ratio >= 3:
        recommendations.append(
            _build_recommendation(
                feature=target,
                problem_type="target",
                metric="imbalance_ratio",
                metric_value=round(imbalance_ratio, 2),
                severity="high",
                recommendation=t("rec.target_imbalance.recommendation"),
                explanation=t(
                    "rec.target_imbalance.explanation",
                    majority_class=distribution.get("majority_class"),
                    majority_pct=f"{distribution.get('majority_pct', 0):.1%}",
                    minority_class=distribution.get("minority_class"),
                    minority_pct=f"{distribution.get('minority_pct', 0):.1%}",
                    ratio=imbalance_ratio,
                ),
            )
        )

    possible_leakage = target_result.get("possible_leakage", [])
    leaking_predictors = {item["predictor"] for item in possible_leakage}
    for item in possible_leakage:
        recommendations.append(
            _build_recommendation(
                feature=item["predictor"],
                problem_type="target",
                metric=item["metric"],
                metric_value=round(item["association"], 4),
                severity="high",
                recommendation=t("rec.target_leakage.recommendation", predictor=item["predictor"]),
                explanation=t(
                    "rec.target_leakage.explanation",
                    metric=item["metric"],
                    association=item["association"],
                ),
            )
        )

    strong_predictors = [
        p for p in target_result.get("strong_predictors", []) if p not in leaking_predictors
    ]
    if strong_predictors:
        recommendations.append(
            _build_recommendation(
                feature=None,
                problem_type="target",
                metric="n_strong_predictors",
                metric_value=len(strong_predictors),
                severity="low",
                recommendation=t("rec.target_strong_predictors.recommendation"),
                explanation=t(
                    "rec.target_strong_predictors.explanation",
                    n=len(strong_predictors),
                    predictors=", ".join(strong_predictors),
                ),
            )
        )

    if target_result.get("multiple_comparisons_warning"):
        recommendations.append(
            _build_recommendation(
                feature=None,
                problem_type="target",
                metric="n_predictors_tested",
                metric_value=target_result.get("n_predictors_tested"),
                severity="low",
                recommendation=t("rec.target_multiple_comparisons.recommendation"),
                explanation=target_result["multiple_comparisons_warning"],
            )
        )

    return recommendations


def recommend_data_leakage_workflow(config: AutoEDAConfig) -> dict[str, Any]:
    """Gera o alerta de Data Leakage — uma recomendação de nível
    geral do dataset, não derivada de uma métrica específica, mas de
    um lembrete de fluxo de trabalho que o AutoEDA sempre inclui.

    O AutoEDA calcula estatísticas e recomendações sobre o dataset
    completo (não separa treino/teste), então este alerta lembra
    explicitamente o usuário de que qualquer estatística usada para
    imputação, normalização, seleção de features ou balanceamento
    (SMOTE) deve ser recalculada apenas no conjunto de treino antes
    de ser aplicada ao conjunto de teste — nunca o contrário.
    """
    t = get_translator(config.language)
    return _build_recommendation(
        feature=None,
        problem_type="data_leakage",
        metric=None,
        metric_value=None,
        severity="high",
        recommendation=t("rec.data_leakage.recommendation"),
        explanation=t("rec.data_leakage.explanation"),
    )


def generate_recommendations(
    descriptive_result: dict[str, Any],
    missing_result: dict[str, Any],
    outliers_result: dict[str, Any],
    correlation_result: dict[str, Any],
    target_result: dict[str, Any],
    config: AutoEDAConfig,
) -> dict[str, Any]:
    """Agrega as recomendações de todos os módulos de análise.

    `target_result` não é mais opcional: o AutoEDA está restrito a
    classificação binária e o target é sempre obrigatório (ver
    utils.validate_target), então a análise de target sempre existe.

    Retorna um dict no formato:
    {
        "schema_version": "1.0",
        "recommendations": [
            {"feature": ..., "problem_type": ..., "metric": ..., "metric_value": ...,
             "severity": ..., "recommendation": ..., "explanation": ...},
            ...
        ],
    }
    A lista é ordenada por severidade (high -> medium -> low). O
    alerta de Data Leakage (recommend_data_leakage_workflow) é sempre
    incluído, independente dos dados — é um lembrete de processo, não
    uma detecção.
    """
    recommendations: list[dict[str, Any]] = []

    recommendations.append(recommend_data_leakage_workflow(config))
    recommendations.extend(recommend_from_missing_values(missing_result, descriptive_result, config))
    recommendations.extend(recommend_from_outliers(outliers_result, config))
    recommendations.extend(recommend_from_descriptive(descriptive_result, config))
    recommendations.extend(recommend_from_correlation(correlation_result, config))
    recommendations.extend(recommend_from_target(target_result, config))

    severity_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda rec: severity_order.get(rec["severity"], 3))

    return {
        "schema_version": RECOMMENDATIONS_SCHEMA_VERSION,
        "recommendations": recommendations,
    }
