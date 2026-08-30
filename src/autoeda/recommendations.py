"""
Geração das recomendações de transformação/pré-processamento do AutoEDA.

Este é o módulo que conecta os resultados de analysis/ (que apenas
descrevem e classificam os dados) a ações concretas que o usuário
pode aplicar antes de modelar. A saída daqui é o conteúdo do JSON de
recomendações citado no escopo do projeto — report/json_export.py
apenas serializa o que este módulo produz.

Cada recomendação é um dict com um formato fixo:
{
    "id": str,                 # identificador curto e estável, ex. "missing_high_idade"
    "column": str | None,      # coluna afetada, ou None para recomendação de dataset inteiro
    "category": str,           # "missing_values" | "outliers" | "correlation" |
                                # "descriptive" | "target" | "temporal"
    "priority": str,           # "high" | "medium" | "low"
    "issue": str,               # descrição do problema encontrado
    "action": str,               # ação sugerida
    "rationale": str,          # por que essa ação é sugerida
}

As decisões de threshold (o que é "severo", o que é "alta
correlação") já foram tomadas pelos módulos de analysis/ a partir de
AutoEDAConfig; este módulo não reaplica lógica de threshold, apenas
traduz os resultados já classificados em recomendações de texto e
prioridade.
"""

from __future__ import annotations

from typing import Any


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
    - severidade "high" -> sugerir remoção da coluna (imputar em >=
      60% dos valores tende a introduzir mais ruído do que sinal).
    - severidade "moderate" -> sugerir imputação, com estratégia
      dependente do tipo (mediana para numéricas, robusta a
      outliers; moda para categóricas).
    - severidade "low" -> mesma sugestão de imputação, prioridade
      menor (o impacto de poucos valores ausentes é limitado mesmo
      sem tratamento).
    - pares de colunas com ausência correlacionada -> recomendação
      adicional (dataset-level) sugerindo tratar o padrão em conjunto
      em vez de coluna a coluna.
    """
    recommendations: list[dict[str, Any]] = []

    for column, stats in missing_result.get("columns", {}).items():
        severity = stats["severity"]
        column_type = _get_column_type(descriptive_result, column)

        if severity == "high":
            recommendations.append(
                {
                    "id": f"missing_high_{column}",
                    "column": column,
                    "category": "missing_values",
                    "priority": "high",
                    "issue": (
                        f"Coluna '{column}' possui {stats['missing_pct']:.1%} de "
                        "valores ausentes."
                    ),
                    "action": f"Considerar remover a coluna '{column}' do dataset.",
                    "rationale": (
                        "Percentual de ausência muito alto torna a imputação "
                        "pouco confiável; a coluna tende a agregar mais ruído "
                        "do que informação útil ao modelo."
                    ),
                }
            )
            continue

        if column_type == "numeric":
            action = f"Imputar valores ausentes de '{column}' com a mediana."
            rationale = "A mediana é robusta a outliers, mais segura que a média como padrão."
        else:
            action = f"Imputar valores ausentes de '{column}' com a moda (categoria mais frequente)."
            rationale = "Para colunas categóricas, a moda preserva a distribuição original das classes."

        recommendations.append(
            {
                "id": f"missing_{severity}_{column}",
                "column": column,
                "category": "missing_values",
                "priority": "medium" if severity == "moderate" else "low",
                "issue": (
                    f"Coluna '{column}' possui {stats['missing_pct']:.1%} de "
                    "valores ausentes."
                ),
                "action": action,
                "rationale": rationale,
            }
        )

    correlated_pairs = missing_result.get("correlated_missingness", [])
    if correlated_pairs:
        top_pair = correlated_pairs[0]
        recommendations.append(
            {
                "id": "missing_correlated_pattern",
                "column": None,
                "category": "missing_values",
                "priority": "medium",
                "issue": (
                    f"As colunas '{top_pair['column_a']}' e '{top_pair['column_b']}' "
                    f"tendem a estar ausentes juntas (correlação de ausência "
                    f"{top_pair['correlation']:.2f})."
                ),
                "action": (
                    "Investigar se a ausência conjunta reflete um processo comum "
                    "(ex.: mesma etapa opcional de coleta) antes de imputar cada "
                    "coluna separadamente."
                ),
                "rationale": (
                    "Ausência correlacionada sugere um padrão sistemático (MAR), "
                    "não aleatório — imputação independente por coluna pode "
                    "distorcer a relação entre elas."
                ),
            }
        )

    return recommendations


def recommend_from_outliers(outliers_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.outliers.analyze_outliers.

    Regras:
    - coluna sem outliers detectados (count == 0) -> nenhuma
      recomendação (não polui o relatório).
    - percentual de outliers <= 5% -> sugerir capping/winsorização
      nos limites calculados, prioridade baixa (efeito pontual).
    - percentual de outliers > 5% -> prioridade média e recomendação
      de investigação manual antes de qualquer tratamento automático,
      já que um percentual alto pode indicar erro de coleta ou uma
      subpopulação legítima, não outliers no sentido usual.
    """
    recommendations: list[dict[str, Any]] = []
    method = outliers_result.get("method", "iqr")

    for column, stats in outliers_result.get("columns", {}).items():
        if stats["count"] == 0:
            continue

        if stats["pct"] <= 0.05:
            action = f"Aplicar winsorização (capping) nos valores extremos de '{column}'."
            priority = "low"
        else:
            action = (
                f"Investigar manualmente os valores extremos de '{column}' antes de "
                "tratá-los automaticamente."
            )
            priority = "medium"

        recommendations.append(
            {
                "id": f"outliers_{column}",
                "column": column,
                "category": "outliers",
                "priority": priority,
                "issue": (
                    f"{stats['count']} outlier(s) detectado(s) em '{column}' "
                    f"({stats['pct']:.1%} das observações, método {method})."
                ),
                "action": action,
                "rationale": (
                    f"{stats['pct']:.1%} é um percentual "
                    + ("baixo, compatível com ruído pontual." if stats["pct"] <= 0.05
                       else "alto para outliers isolados; pode indicar erro de coleta "
                            "ou uma subpopulação distinta que merece análise separada.")
                ),
            }
        )

    return recommendations


def recommend_from_descriptive(descriptive_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.descriptive.generate_descriptive_stats.

    Regras:
    - linhas duplicadas presentes -> sugerir remoção (dataset-level).
    - coluna do tipo "id" -> sugerir exclusão de correlação/modelagem
      (identificadores não carregam sinal preditivo).
    - coluna numérica com |assimetria| > 1 -> sugerir transformação
      (log, se todos os valores forem positivos; caso contrário,
      Box-Cox/Yeo-Johnson) para aproximar de uma distribuição
      simétrica.
    """
    recommendations: list[dict[str, Any]] = []

    overview = descriptive_result.get("overview", {})
    if overview.get("duplicate_rows", 0) > 0:
        recommendations.append(
            {
                "id": "duplicate_rows",
                "column": None,
                "category": "descriptive",
                "priority": "medium",
                "issue": (
                    f"{overview['duplicate_rows']} linha(s) duplicada(s) "
                    f"({overview['duplicate_rows_pct']:.1%} do dataset)."
                ),
                "action": "Remover linhas duplicadas antes de qualquer análise/modelagem.",
                "rationale": (
                    "Linhas duplicadas distorcem estatísticas descritivas e podem "
                    "causar vazamento de dados entre treino e teste se não forem "
                    "removidas antes da divisão."
                ),
            }
        )

    for column, column_report in descriptive_result.get("columns", {}).items():
        col_type = column_report["type"]
        stats = column_report["stats"]

        if col_type == "id":
            recommendations.append(
                {
                    "id": f"id_column_{column}",
                    "column": column,
                    "category": "descriptive",
                    "priority": "low",
                    "issue": f"Coluna '{column}' identificada como identificador (valores únicos).",
                    "action": f"Excluir '{column}' de análises de correlação e da modelagem.",
                    "rationale": (
                        "Identificadores não carregam relação causal/preditiva com o "
                        "target; incluí-los pode gerar correlações espúrias."
                    ),
                }
            )
            continue

        if col_type == "numeric" and stats.get("skewness") is not None and abs(stats["skewness"]) > 1:
            all_positive = stats.get("min") is not None and stats["min"] > 0
            action = (
                f"Aplicar transformação logarítmica em '{column}'."
                if all_positive
                else f"Aplicar transformação Yeo-Johnson em '{column}' (há valores <= 0)."
            )
            recommendations.append(
                {
                    "id": f"skewed_{column}",
                    "column": column,
                    "category": "descriptive",
                    "priority": "low",
                    "issue": f"Coluna '{column}' com assimetria de {stats['skewness']:.2f}.",
                    "action": action,
                    "rationale": (
                        "Distribuições fortemente assimétricas violam a suposição de "
                        "normalidade de vários modelos e métricas; a transformação "
                        "aproxima a distribuição de uma forma mais simétrica."
                    ),
                }
            )

    return recommendations


def recommend_from_correlation(correlation_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.correlation.analyze_correlation.

    Para cada par de colunas fortemente correlacionadas (já
    identificado pelo módulo de análise), sugere avaliar a remoção de
    uma delas — redundância entre preditoras aumenta a variância de
    modelos lineares (multicolinearidade) sem agregar informação.
    """
    recommendations: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for pair in correlation_result.get("high_correlations", []):
        key = tuple(sorted((pair["column_a"], pair["column_b"])))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)

        recommendations.append(
            {
                "id": f"correlation_{key[0]}_{key[1]}",
                "column": None,
                "category": "correlation",
                "priority": "medium",
                "issue": (
                    f"'{pair['column_a']}' e '{pair['column_b']}' têm correlação "
                    f"{pair['method']} de {pair['correlation']:.2f}."
                ),
                "action": (
                    f"Avaliar remover uma das colunas ('{pair['column_a']}' ou "
                    f"'{pair['column_b']}') ou combiná-las em uma única feature."
                ),
                "rationale": (
                    "Colunas fortemente correlacionadas carregam informação "
                    "redundante; mantê-las ambas aumenta a multicolinearidade "
                    "sem ganho proporcional de sinal."
                ),
            }
        )

    return recommendations


def recommend_from_target(target_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.target_analysis.analyze_target.

    Regras:
    - target categórico com imbalance_ratio >= 3 -> sugerir técnica de
      balanceamento (reamostragem ou pesos de classe).
    - preditores fortes (já identificados pelo módulo de análise) ->
      recomendação informativa (prioridade baixa) destacando quais
      features merecem atenção prioritária na modelagem.
    """
    recommendations: list[dict[str, Any]] = []

    distribution = target_result.get("distribution", {})
    if distribution.get("target_type") == "categorical":
        imbalance_ratio = distribution.get("imbalance_ratio")
        if imbalance_ratio is not None and imbalance_ratio >= 3:
            recommendations.append(
                {
                    "id": "target_imbalance",
                    "column": target_result["target"],
                    "category": "target",
                    "priority": "high",
                    "issue": (
                        f"Classes do target '{target_result['target']}' desbalanceadas "
                        f"(razão {imbalance_ratio:.1f}:1 entre a classe mais e a menos "
                        "frequente)."
                    ),
                    "action": (
                        "Considerar reamostragem (over/undersampling) ou pesos de "
                        "classe (class_weight) na etapa de modelagem."
                    ),
                    "rationale": (
                        "Desbalanceamento forte tende a enviesar modelos em favor "
                        "da classe majoritária e a inflar métricas como acurácia "
                        "sem refletir desempenho real na classe minoritária."
                    ),
                }
            )

    strong_predictors = target_result.get("strong_predictors", [])
    if strong_predictors:
        recommendations.append(
            {
                "id": "target_strong_predictors",
                "column": None,
                "category": "target",
                "priority": "low",
                "issue": (
                    f"{len(strong_predictors)} variável(is) com associação forte ao "
                    f"target '{target_result['target']}': {', '.join(strong_predictors)}."
                ),
                "action": "Priorizar essas variáveis na seleção de features do modelo.",
                "rationale": (
                    "Variáveis com associação forte ao target (correlação/eta²/V de "
                    "Cramér elevados) tendem a carregar mais sinal preditivo."
                ),
            }
        )

    return recommendations


def recommend_from_temporal(temporal_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.temporal.analyze_temporal.

    Regras:
    - tendência forte detectada (direction != "estavel" e
      != "indeterminado") -> sugerir extrair a tendência como feature
      (ex.: índice de tempo) e/ou aplicar diferenciação se o objetivo
      for modelar a série em si.
    - frequência "irregular" -> sugerir reamostrar para uma grade de
      tempo regular antes de qualquer modelagem de série temporal.
    """
    recommendations: list[dict[str, Any]] = []

    trend = temporal_result.get("trend")
    column = temporal_result.get("value_column") or temporal_result.get("temporal_column")

    if trend and trend["direction"] in ("crescente", "decrescente"):
        recommendations.append(
            {
                "id": "temporal_trend",
                "column": column,
                "category": "temporal",
                "priority": "low",
                "issue": (
                    f"Tendência {trend['direction']} detectada ao longo do tempo "
                    f"(correlação tempo-valor de {trend['correlation']:.2f})."
                ),
                "action": (
                    "Incluir um índice de tempo como feature, ou aplicar "
                    "diferenciação caso o objetivo seja modelar a série "
                    "diretamente (ex.: ARIMA)."
                ),
                "rationale": (
                    "Uma tendência não tratada pode ser confundida com efeito de "
                    "outras variáveis correlacionadas ao tempo (confounding), e "
                    "modelos de série temporal geralmente assumem estacionariedade."
                ),
            }
        )

    if temporal_result.get("frequency") == "irregular":
        recommendations.append(
            {
                "id": "temporal_irregular_frequency",
                "column": temporal_result.get("temporal_column"),
                "category": "temporal",
                "priority": "low",
                "issue": "Espaçamento irregular entre as observações temporais.",
                "action": (
                    "Reamostrar os dados para uma grade de tempo regular "
                    "(ex.: diária, semanal) antes de análises de série temporal."
                ),
                "rationale": (
                    "A maioria dos métodos de série temporal assume observações "
                    "igualmente espaçadas; espaçamento irregular pode distorcer "
                    "tendência e sazonalidade estimadas."
                ),
            }
        )

    return recommendations


def generate_recommendations(
    descriptive_result: dict[str, Any],
    missing_result: dict[str, Any],
    outliers_result: dict[str, Any],
    correlation_result: dict[str, Any],
    target_result: dict[str, Any] | None = None,
    temporal_result: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Agrega as recomendações de todos os módulos de análise em uma
    única lista, ordenada por prioridade (high -> medium -> low).

    `target_result` e `temporal_result` são opcionais porque nem toda
    execução do AutoEDA tem uma variável alvo ou uma componente
    temporal informada (ver a assinatura de autoeda.core.autoeda).

    Esta é a função consumida por report/json_export.py para montar
    o JSON de recomendações citado no escopo do projeto.
    """
    recommendations: list[dict[str, Any]] = []

    recommendations.extend(recommend_from_missing_values(missing_result, descriptive_result))
    recommendations.extend(recommend_from_outliers(outliers_result))
    recommendations.extend(recommend_from_descriptive(descriptive_result))
    recommendations.extend(recommend_from_correlation(correlation_result))

    if target_result is not None:
        recommendations.extend(recommend_from_target(target_result))

    if temporal_result is not None:
        recommendations.extend(recommend_from_temporal(temporal_result))

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda rec: priority_order.get(rec["priority"], 3))

    return recommendations
