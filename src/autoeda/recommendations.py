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
    recommendations: list[dict[str, Any]] = []

    for column, stats in missing_result.get("columns", {}).items():
        severity = stats["severity"]
        column_type = _get_column_type(descriptive_result, column)
        missing_pct_display = round(stats["missing_pct"] * 100, 2)

        if severity == "high":
            recommendations.append(
                _build_recommendation(
                    feature=column,
                    problem_type="missing_values",
                    metric="missing_percentage",
                    metric_value=missing_pct_display,
                    severity="high",
                    recommendation=f"Considerar remover a coluna '{column}' do dataset.",
                    explanation=(
                        f"Coluna com {stats['missing_pct']:.1%} de valores ausentes. "
                        "Percentual muito alto torna a imputação pouco confiável; a "
                        "coluna tende a agregar mais ruído do que informação útil."
                    ),
                )
            )
            continue

        if column_type == "numeric":
            technique = f"Considerar imputar valores ausentes de '{column}' com a mediana."
            reason = "A mediana é robusta a outliers, mais segura que a média como padrão."
        else:
            technique = f"Considerar imputar valores ausentes de '{column}' com a moda (categoria mais frequente)."
            reason = "Para colunas categóricas, a moda preserva a distribuição original das classes."

        recommendations.append(
            _build_recommendation(
                feature=column,
                problem_type="missing_values",
                metric="missing_percentage",
                metric_value=missing_pct_display,
                severity="medium" if severity == "moderate" else "low",
                recommendation=technique,
                explanation=f"Coluna com {stats['missing_pct']:.1%} de valores ausentes. {reason}",
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
                recommendation=(
                    f"Investigar se a ausência conjunta de '{top_pair['column_a']}' e "
                    f"'{top_pair['column_b']}' reflete um processo comum (ex.: mesma "
                    "etapa opcional de coleta) antes de imputar cada coluna separadamente."
                ),
                explanation=(
                    f"'{top_pair['column_a']}' e '{top_pair['column_b']}' tendem a estar "
                    f"ausentes juntas (correlação de ausência {top_pair['correlation']:.2f}). "
                    "Ausência correlacionada sugere um padrão sistemático (indício de MAR), "
                    "não aleatório — imputação independente por coluna pode distorcer a "
                    "relação entre elas."
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
                recommendation=(
                    f"Considerar criar uma feature binária indicando a ausência de "
                    f"'{item['column']}' (missing indicator) além de imputar o valor."
                ),
                explanation=(
                    f"A taxa de ausência de '{item['column']}' difere entre as classes "
                    f"do target ({rates_text}). Ausência que difere por classe é, em si, "
                    "informação preditiva (indício de MAR ligado ao problema) — "
                    "descartá-la na imputação joga fora sinal que o modelo poderia "
                    "aproveitar."
                ),
            )
        )

    return recommendations


def recommend_from_outliers(outliers_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.outliers.analyze_outliers.

    Outliers nunca são recomendados para remoção automática — apenas
    winsorização (percentuais baixos) ou investigação manual
    (percentuais altos, que podem indicar erro de coleta ou uma
    subpopulação legítima).
    """
    recommendations: list[dict[str, Any]] = []
    method = outliers_result.get("method", "iqr")

    for column, stats in outliers_result.get("columns", {}).items():
        if stats["count"] == 0:
            continue

        pct_display = round(stats["pct"] * 100, 2)

        if stats["pct"] <= 0.05:
            technique = f"Considerar aplicar winsorização (capping) nos valores extremos de '{column}'."
            severity = "low"
            reason = "baixo, compatível com ruído pontual"
        else:
            technique = f"Investigar manualmente os valores extremos de '{column}' antes de tratá-los."
            severity = "medium"
            reason = "alto para outliers isolados; pode indicar erro de coleta ou subpopulação distinta"

        recommendations.append(
            _build_recommendation(
                feature=column,
                problem_type="outliers",
                metric="outlier_percentage",
                metric_value=pct_display,
                severity=severity,
                recommendation=technique,
                explanation=(
                    f"{stats['count']} outlier(s) detectado(s) ({stats['pct']:.1%} das "
                    f"observações, método {method}) — percentual {reason}. Outliers não "
                    "devem ser removidos automaticamente, pois podem representar "
                    "informação legítima do domínio."
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
                recommendation="Remover linhas duplicadas antes de qualquer análise/modelagem.",
                explanation=(
                    f"{overview['duplicate_rows']} linha(s) duplicada(s) "
                    f"({overview['duplicate_rows_pct']:.1%} do dataset). Linhas duplicadas "
                    "distorcem estatísticas descritivas e podem causar vazamento de dados "
                    "entre treino e teste se não forem removidas antes da divisão."
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
                    recommendation=f"Avaliar remover '{column}' de análises de correlação e da modelagem.",
                    explanation=(
                        "Coluna identificada como possível identificador (proporção de "
                        "valores únicos próxima de 100%). Identificadores não carregam "
                        "relação causal/preditiva com o target; incluí-los pode gerar "
                        "correlações espúrias."
                    ),
                )
            )
            continue

        if col_type == "numeric" and stats.get("skewness") is not None and abs(stats["skewness"]) > 1:
            all_positive = stats.get("min") is not None and stats["min"] > 0
            technique = (
                f"Considerar aplicar transformação logarítmica em '{column}'."
                if all_positive
                else f"Considerar aplicar transformação Yeo-Johnson em '{column}' (há valores <= 0)."
            )
            recommendations.append(
                _build_recommendation(
                    feature=column,
                    problem_type="descriptive",
                    metric="skewness",
                    metric_value=round(stats["skewness"], 4),
                    severity="low",
                    recommendation=technique,
                    explanation=(
                        f"Assimetria de {stats['skewness']:.2f}. Distribuições fortemente "
                        "assimétricas violam a suposição de normalidade de vários modelos "
                        "e métricas; a transformação aproxima a distribuição de uma forma "
                        "mais simétrica."
                    ),
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
                        recommendation=(
                            f"Avaliar Frequency Encoding, Target Encoding ou agrupamento "
                            f"de categorias raras em '{column}' (ou remoção, caso seja "
                            "possível identificador)."
                        ),
                        explanation=(
                            f"Cardinalidade alta ({unique_count if unique_count is not None else 'muitas'} "
                            "categorias). One-hot encoding em colunas de alta cardinalidade "
                            "gera um número excessivo de novas colunas esparsas; técnicas "
                            "de encoding baseadas em frequência/target ou o agrupamento de "
                            "categorias raras em 'outros' tendem a generalizar melhor."
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
                    recommendation=f"Remover a coluna '{column}' do dataset.",
                    explanation=(
                        "Coluna constante (um único valor em toda a amostra). Uma coluna "
                        "constante tem variância zero e não pode, por definição, "
                        "contribuir para separar as classes do target."
                    ),
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
                    recommendation=f"Avaliar remover '{column}' ou tratá-la como de baixo poder informativo.",
                    explanation=(
                        f"Coluna quase constante: um único valor responde por "
                        f"{info['top_value_pct']:.1%} das observações. Carrega pouca "
                        "informação para separar as classes, mesmo sem variância "
                        "tecnicamente zero, e pode instabilizar modelos sensíveis a "
                        "features de baixa variância."
                    ),
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
                recommendation=(
                    f"Padronizar o formato de '{column}' (ex.: converter valores por "
                    "extenso para número, ou tratá-los como categoria 'inválido') "
                    "antes de qualquer análise."
                ),
                explanation=(
                    f"Coluna mistura valores numéricos ({info['numeric_pct']:.1%}) e não "
                    f"numéricos ({info['non_numeric_pct']:.1%}). Costuma indicar erro de "
                    "digitação ou de exportação; sem correção, a coluna é mal "
                    "classificada e os cálculos estatísticos ficam distorcidos."
                ),
            )
        )

    return recommendations


def recommend_from_correlation(correlation_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.correlation.analyze_correlation.

    Cobre 3 sinais independentes: pares fortemente correlacionados
    (Pearson/Spearman), VIF alto (redundância multivariada) e
    disparidade de escala entre variáveis numéricas.
    """
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
                recommendation=(
                    f"Avaliar remover uma das colunas ('{pair['column_a']}' ou "
                    f"'{pair['column_b']}') ou combiná-las em uma única feature."
                ),
                explanation=(
                    f"'{pair['column_a']}' e '{pair['column_b']}' têm correlação "
                    f"{pair['method']} de {pair['correlation']:.2f}. Colunas fortemente "
                    "correlacionadas carregam informação redundante; mantê-las ambas "
                    "aumenta a multicolinearidade sem ganho proporcional de sinal."
                ),
            )
        )

    for item in correlation_result.get("high_vif", []):
        vif_value = item["vif"]
        vif_display = "infinito" if vif_value == float("inf") else round(vif_value, 2)
        recommendations.append(
            _build_recommendation(
                feature=item["column"],
                problem_type="correlation",
                metric="vif",
                metric_value=vif_display,
                severity="medium",
                recommendation=(
                    f"Avaliar remover '{item['column']}' ou reduzir a dimensionalidade "
                    "do grupo de variáveis redundantes (ex.: PCA) antes de um modelo linear."
                ),
                explanation=(
                    f"VIF de {vif_display}. VIF alto indica que a variável é quase uma "
                    "combinação linear de outras variáveis do dataset — diferente da "
                    "correlação par a par, o VIF captura redundância multivariada, mesmo "
                    "quando nenhum par isolado parece fortemente correlacionado."
                ),
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
                recommendation=(
                    "Padronizar (StandardScaler) ou normalizar as variáveis numéricas "
                    "antes de modelos sensíveis a escala (ex.: KNN, SVM, regressão com "
                    "regularização L1/L2)."
                ),
                explanation=(
                    f"'{scale_disparity['largest_scale_column']}' "
                    f"(desvio padrão {scale_disparity['largest_scale_std']:.2f}) está em "
                    f"escala muito maior que '{scale_disparity['smallest_scale_column']}' "
                    f"(desvio padrão {scale_disparity['smallest_scale_std']:.2f}), razão de "
                    f"{scale_disparity['ratio']:.0f}x. Variáveis em escalas muito diferentes "
                    "dominam o cálculo de distância ou o termo de regularização apenas "
                    "por causa da magnitude, não porque carregam mais sinal."
                ),
            )
        )

    return recommendations


def recommend_from_target(target_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.target_analysis.analyze_target (target binário).

    Cobre: desbalanceamento de classes, vazamento direto (associação
    suspeitosamente quase perfeita com o target), preditores fortes e
    aviso de múltiplas comparações.
    """
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
                recommendation=(
                    "Considerar reamostragem (over/undersampling ou SMOTE, aplicados "
                    "somente no conjunto de treino) ou pesos de classe (class_weight) "
                    "na etapa de modelagem."
                ),
                explanation=(
                    f"Classes desbalanceadas: '{distribution.get('majority_class')}' "
                    f"({distribution.get('majority_pct', 0):.1%}) vs "
                    f"'{distribution.get('minority_class')}' "
                    f"({distribution.get('minority_pct', 0):.1%}), razão "
                    f"{imbalance_ratio:.1f}:1. Acurácia isoladamente pode ser enganosa "
                    "nesse cenário — prefira métricas como F1, recall da classe "
                    "minoritária ou AUC-ROC."
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
                recommendation=(
                    f"Investigar se '{item['predictor']}' é uma proxy do próprio target "
                    "(ex.: preenchida após o evento que o target representa) antes de "
                    "usá-la como preditora."
                ),
                explanation=(
                    f"Associação muito forte com o target ({item['metric']} = "
                    f"{item['association']:.2f}). Associação quase perfeita é mais "
                    "consistente com vazamento de informação do que com um preditor "
                    "legítimo — incluí-la infla artificialmente o desempenho do modelo "
                    "em treino/validação sem generalizar para produção."
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
                recommendation="Priorizar essas variáveis na seleção de features do modelo.",
                explanation=(
                    f"{len(strong_predictors)} variável(is) com associação forte ao "
                    f"target: {', '.join(strong_predictors)}. Variáveis com associação "
                    "forte (Point-Biserial, V de Cramér ou Spearman elevados) tendem a "
                    "carregar mais sinal preditivo."
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
                recommendation=(
                    "Interpretar os p-valores individuais dos testes de associação com "
                    "cautela; preferir os preditores com maior força de associação "
                    "(não só significância) na seleção de features."
                ),
                explanation=target_result["multiple_comparisons_warning"],
            )
        )

    return recommendations


def recommend_data_leakage_workflow() -> dict[str, Any]:
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
    return _build_recommendation(
        feature=None,
        problem_type="data_leakage",
        metric=None,
        metric_value=None,
        severity="high",
        recommendation=(
            "Separar treino e teste antes de calcular qualquer estatística de "
            "preparação de dados; ajustar imputação, normalização, seleção de "
            "features e SMOTE apenas no conjunto de treino, e aplicar as mesmas "
            "transformações (já ajustadas) ao conjunto de teste."
        ),
        explanation=(
            "As estatísticas e recomendações deste relatório foram calculadas sobre "
            "o dataset completo, para fins de diagnóstico exploratório. Usar médias, "
            "medianas, categorias ou parâmetros de reamostragem calculados sobre o "
            "conjunto de teste (ou sobre o dataset inteiro) para preparar os dados "
            "antes da divisão treino/teste é uma forma comum de vazamento de dados: "
            "o modelo passa a ter acesso indireto a informação do teste durante o "
            "treinamento, inflando métricas de validação de forma não realista."
        ),
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

    recommendations.append(recommend_data_leakage_workflow())
    recommendations.extend(recommend_from_missing_values(missing_result, descriptive_result))
    recommendations.extend(recommend_from_outliers(outliers_result))
    recommendations.extend(recommend_from_descriptive(descriptive_result, config))
    recommendations.extend(recommend_from_correlation(correlation_result))
    recommendations.extend(recommend_from_target(target_result))

    severity_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda rec: severity_order.get(rec["severity"], 3))

    return {
        "schema_version": RECOMMENDATIONS_SCHEMA_VERSION,
        "recommendations": recommendations,
    }
