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
                                # "descriptive" | "target"
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

    for item in missing_result.get("missing_target_association", []):
        rates_text = ", ".join(f"{cls}={rate:.1%}" for cls, rate in item["rates_by_class"].items())
        recommendations.append(
            {
                "id": f"missing_target_association_{item['column']}",
                "column": item["column"],
                "category": "missing_values",
                "priority": "medium",
                "issue": (
                    f"A taxa de ausência de '{item['column']}' difere entre as "
                    f"classes do target ({rates_text})."
                ),
                "action": (
                    f"Considerar criar uma feature binária indicando a ausência de "
                    f"'{item['column']}' (missing indicator) além de imputar o valor, "
                    "e evitar imputar com a média/mediana global sem checar se ela "
                    "faz sentido para ambas as classes."
                ),
                "rationale": (
                    "Ausência que difere por classe é, em si, informação preditiva "
                    "(indício de MAR ligado ao problema) — descartá-la na imputação "
                    "joga fora sinal que o modelo poderia aproveitar."
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

    for column, info in descriptive_result.get("constant_and_near_zero_variance", {}).items():
        if info["constant"]:
            recommendations.append(
                {
                    "id": f"constant_{column}",
                    "column": column,
                    "category": "descriptive",
                    "priority": "high",
                    "issue": f"Coluna '{column}' é constante (um único valor em toda a amostra).",
                    "action": f"Remover a coluna '{column}' do dataset.",
                    "rationale": (
                        "Uma coluna constante tem variância zero e não pode, por "
                        "definição, contribuir para separar as classes do target."
                    ),
                }
            )
        else:
            recommendations.append(
                {
                    "id": f"near_zero_variance_{column}",
                    "column": column,
                    "category": "descriptive",
                    "priority": "low",
                    "issue": (
                        f"Coluna '{column}' é quase constante: um único valor responde "
                        f"por {info['top_value_pct']:.1%} das observações."
                    ),
                    "action": f"Avaliar remover '{column}' ou tratá-la como baixo poder informativo.",
                    "rationale": (
                        "Colunas quase constantes carregam pouca informação para "
                        "separar as classes, mesmo sem variância tecnicamente zero, "
                        "e podem instabilizar modelos sensíveis a features de baixa "
                        "variância."
                    ),
                }
            )

    for column, info in descriptive_result.get("mixed_type_columns", {}).items():
        recommendations.append(
            {
                "id": f"mixed_type_{column}",
                "column": column,
                "category": "descriptive",
                "priority": "medium",
                "issue": (
                    f"Coluna '{column}' mistura valores numéricos ({info['numeric_pct']:.1%}) "
                    f"e não numéricos ({info['non_numeric_pct']:.1%})."
                ),
                "action": (
                    f"Padronizar o formato de '{column}' (ex.: converter valores por "
                    "extenso para número, ou tratá-los como categoria 'inválido') "
                    "antes de qualquer análise."
                ),
                "rationale": (
                    "Uma coluna de tipo misto costuma indicar erro de digitação ou "
                    "de exportação; sem correção, a coluna é mal classificada "
                    "(numérica vira categórica ou vice-versa) e os cálculos "
                    "estatísticos ficam distorcidos."
                ),
            }
        )

    return recommendations


def recommend_from_correlation(correlation_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.correlation.analyze_correlation.

    Cobre 3 sinais independentes:
    - pares fortemente correlacionados (Pearson/Spearman) -> avaliar
      remover uma das colunas ou combiná-las.
    - VIF alto -> mesma recomendação de fundo (redundância), mas
      motivada pela variável ser explicada pela combinação de
      *várias* outras, não necessariamente por um único par.
    - disparidade de escala -> sugerir padronização antes de modelos
      sensíveis a escala.
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

    for item in correlation_result.get("high_vif", []):
        vif_display = "infinito" if item["vif"] == float("inf") else f"{item['vif']:.1f}"
        recommendations.append(
            {
                "id": f"vif_{item['column']}",
                "column": item["column"],
                "category": "correlation",
                "priority": "medium",
                "issue": f"Coluna '{item['column']}' tem VIF de {vif_display}.",
                "action": (
                    f"Avaliar remover '{item['column']}' ou reduzir a dimensionalidade "
                    "do grupo de variáveis redundantes (ex.: PCA) antes de um modelo "
                    "linear."
                ),
                "rationale": (
                    "VIF alto indica que a variável é quase uma combinação linear de "
                    "outras variáveis do dataset — diferente da correlação par a par, "
                    "o VIF captura redundância multivariada, mesmo quando nenhum par "
                    "isolado parece fortemente correlacionado."
                ),
            }
        )

    scale_disparity = correlation_result.get("scale_disparity")
    if scale_disparity:
        recommendations.append(
            {
                "id": "scale_disparity",
                "column": None,
                "category": "correlation",
                "priority": "low",
                "issue": (
                    f"'{scale_disparity['largest_scale_column']}' "
                    f"(desvio padrão {scale_disparity['largest_scale_std']:.2f}) está em "
                    f"escala muito maior que '{scale_disparity['smallest_scale_column']}' "
                    f"(desvio padrão {scale_disparity['smallest_scale_std']:.2f}), razão "
                    f"de {scale_disparity['ratio']:.0f}x."
                ),
                "action": (
                    "Padronizar (StandardScaler) ou normalizar as variáveis numéricas "
                    "antes de modelos sensíveis a escala (ex.: KNN, SVM, regressão "
                    "com regularização L1/L2)."
                ),
                "rationale": (
                    "Variáveis em escalas muito diferentes dominam o cálculo de "
                    "distância ou o termo de regularização apenas por causa da "
                    "magnitude, não porque carregam mais sinal."
                ),
            }
        )

    return recommendations


def recommend_from_target(target_result: dict[str, Any]) -> list[dict[str, Any]]:
    """Gera recomendações a partir do resultado de
    analysis.target_analysis.analyze_target (target binário).

    Regras:
    - imbalance_ratio >= 3 -> sugerir técnica de balanceamento
      (reamostragem ou pesos de classe).
    - preditores com associação suspeitosamente quase perfeita
      (possible_leakage) -> alerta de alta prioridade, separado da
      recomendação de "preditor forte": aqui a hipótese é que a
      variável é uma proxy do próprio target, não um preditor
      legítimo.
    - preditores fortes (mas não classificados como vazamento) ->
      recomendação informativa (prioridade baixa) destacando quais
      features merecem atenção prioritária na modelagem.
    - aviso de múltiplas comparações (já calculado pelo módulo de
      análise) -> recomendação informativa sobre como interpretar os
      p-valores individuais com cautela.
    """
    recommendations: list[dict[str, Any]] = []
    target = target_result["target"]
    distribution = target_result.get("distribution", {})

    imbalance_ratio = distribution.get("imbalance_ratio")
    if imbalance_ratio is not None and imbalance_ratio >= 3:
        recommendations.append(
            {
                "id": "target_imbalance",
                "column": target,
                "category": "target",
                "priority": "high",
                "issue": (
                    f"Classes do target '{target}' desbalanceadas: "
                    f"'{distribution.get('majority_class')}' ({distribution.get('majority_pct', 0):.1%}) "
                    f"vs '{distribution.get('minority_class')}' ({distribution.get('minority_pct', 0):.1%}), "
                    f"razão {imbalance_ratio:.1f}:1."
                ),
                "action": (
                    "Considerar reamostragem (over/undersampling ou SMOTE, aplicados "
                    "somente no conjunto de treino) ou pesos de classe (class_weight) "
                    "na etapa de modelagem."
                ),
                "rationale": (
                    "Desbalanceamento forte tende a enviesar modelos em favor da "
                    "classe majoritária. Acurácia isoladamente pode ser enganosa "
                    "nesse cenário — prefira métricas como F1, recall da classe "
                    "minoritária ou AUC-ROC."
                ),
            }
        )

    possible_leakage = target_result.get("possible_leakage", [])
    leaking_predictors = {item["predictor"] for item in possible_leakage}
    for item in possible_leakage:
        recommendations.append(
            {
                "id": f"target_possible_leakage_{item['predictor']}",
                "column": item["predictor"],
                "category": "target",
                "priority": "high",
                "issue": (
                    f"'{item['predictor']}' tem associação muito forte com o target "
                    f"'{target}' ({item['metric']} = {item['association']:.2f})."
                ),
                "action": (
                    f"Investigar se '{item['predictor']}' é uma proxy do próprio target "
                    "(ex.: preenchida após o evento que o target representa) antes de "
                    "usá-la como preditora."
                ),
                "rationale": (
                    "Associação quase perfeita com o target é mais consistente com "
                    "vazamento de informação do que com um preditor legítimo — "
                    "incluí-la infla artificialmente o desempenho do modelo em "
                    "treino/validação sem generalizar para produção."
                ),
            }
        )

    strong_predictors = [
        p for p in target_result.get("strong_predictors", []) if p not in leaking_predictors
    ]
    if strong_predictors:
        recommendations.append(
            {
                "id": "target_strong_predictors",
                "column": None,
                "category": "target",
                "priority": "low",
                "issue": (
                    f"{len(strong_predictors)} variável(is) com associação forte ao "
                    f"target '{target}': {', '.join(strong_predictors)}."
                ),
                "action": "Priorizar essas variáveis na seleção de features do modelo.",
                "rationale": (
                    "Variáveis com associação forte ao target (Point-Biserial, V de "
                    "Cramér ou Spearman elevados) tendem a carregar mais sinal "
                    "preditivo."
                ),
            }
        )

    if target_result.get("multiple_comparisons_warning"):
        recommendations.append(
            {
                "id": "target_multiple_comparisons",
                "column": None,
                "category": "target",
                "priority": "low",
                "issue": target_result["multiple_comparisons_warning"],
                "action": (
                    "Interpretar os p-valores individuais dos testes de associação "
                    "com cautela; preferir os preditores com maior força de "
                    "associação (não só significância) na seleção de features."
                ),
                "rationale": (
                    "Testar muitos preditores simultaneamente aumenta a chance de "
                    "falsos positivos (associações 'significativas' por acaso)."
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
) -> list[dict[str, Any]]:
    """Agrega as recomendações de todos os módulos de análise em uma
    única lista, ordenada por prioridade (high -> medium -> low).

    `target_result` é opcional porque nem toda execução do AutoEDA tem
    uma variável alvo informada (ver a assinatura de autoeda.core.autoeda).

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

    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda rec: priority_order.get(rec["priority"], 3))

    return recommendations
