"""
Catálogo de textos em português (pt-br) do AutoEDA.

Chaves usam namespace por ponto (ex.: "rec.missing_high.recommendation")
para agrupar por módulo/regra de origem. Os valores são templates
Python (str.format), preenchidos pelo chamador via
i18n.get_translator("pt-br")("chave", **kwargs).

Este arquivo e en_us.py precisam manter exatamente o mesmo conjunto
de chaves — i18n/__init__.py tem uma função de verificação
(assert_catalogs_in_sync) usada nos testes para garantir isso.
"""

TRANSLATIONS: dict[str, str] = {
    # --- analysis.missing_values: indícios de mecanismo -------------
    "missing.evidence.correlated": "ausência correlacionada com a de '{other}' (r={corr:.2f})",
    "missing.evidence.target_diff": "taxa de ausência difere entre as classes do target ({rates})",
    "missing.hint.mar": "MAR",
    "missing.hint.indeterminate": "indeterminado",
    "missing.note.mar": (
        "Indício de MAR (ausência relacionada a variáveis observadas). Não é uma "
        "confirmação — o mecanismo real não pode ser determinado com certeza "
        "apenas a partir dos dados."
    ),
    "missing.note.indeterminate": (
        "Nenhum indício de MAR encontrado (sem correlação com a ausência de "
        "outras colunas nem diferença relevante entre as classes do target). "
        "Isso não confirma MCAR: a ausência pode ainda depender do próprio "
        "valor não observado (MNAR), o que não é verificável a partir do dataset."
    ),

    # --- analysis.target_analysis -----------------------------------
    "target.multiple_comparisons_warning": (
        "{n} preditores foram testados contra o target simultaneamente; com "
        "tantos testes, é esperado que algumas associações pareçam significativas "
        "por acaso. Considere um nível de significância corrigido (ex.: "
        "Bonferroni, alfa ≈ {alpha:.4f}) ao interpretar os p-valores individualmente."
    ),

    # --- recommendations: valores ausentes ---------------------------
    "rec.missing_high.recommendation": "Considerar remover a coluna '{column}' do dataset.",
    "rec.missing_high.explanation": (
        "Coluna com {pct} de valores ausentes. Percentual muito alto torna a "
        "imputação pouco confiável; a coluna tende a agregar mais ruído do que "
        "informação útil."
    ),
    "rec.missing_numeric.recommendation": "Considerar imputar valores ausentes de '{column}' com a mediana.",
    "rec.missing_numeric.reason": "A mediana é robusta a outliers, mais segura que a média como padrão.",
    "rec.missing_categorical.recommendation": (
        "Considerar imputar valores ausentes de '{column}' com a moda (categoria mais frequente)."
    ),
    "rec.missing_categorical.reason": (
        "Para colunas categóricas, a moda preserva a distribuição original das classes."
    ),
    "rec.missing_generic.explanation": "Coluna com {pct} de valores ausentes. {reason}",
    "rec.missing_correlated.recommendation": (
        "Investigar se a ausência conjunta de '{column_a}' e '{column_b}' reflete "
        "um processo comum (ex.: mesma etapa opcional de coleta) antes de imputar "
        "cada coluna separadamente."
    ),
    "rec.missing_correlated.explanation": (
        "'{column_a}' e '{column_b}' tendem a estar ausentes juntas (correlação de "
        "ausência {corr:.2f}). Ausência correlacionada sugere um padrão "
        "sistemático (indício de MAR), não aleatório — imputação independente "
        "por coluna pode distorcer a relação entre elas."
    ),
    "rec.missing_target_assoc.recommendation": (
        "Considerar criar uma feature binária indicando a ausência de '{column}' "
        "(missing indicator) além de imputar o valor."
    ),
    "rec.missing_target_assoc.explanation": (
        "A taxa de ausência de '{column}' difere entre as classes do target "
        "({rates}). Ausência que difere por classe é, em si, informação "
        "preditiva (indício de MAR ligado ao problema) — descartá-la na "
        "imputação joga fora sinal que o modelo poderia aproveitar."
    ),

    # --- recommendations: outliers ------------------------------------
    "rec.outliers.recommendation_low": (
        "Considerar aplicar winsorização (capping) nos valores extremos de '{column}'."
    ),
    "rec.outliers.recommendation_medium": (
        "Investigar manualmente os valores extremos de '{column}' antes de tratá-los."
    ),
    "rec.outliers.reason_low": "baixo, compatível com ruído pontual",
    "rec.outliers.reason_medium": (
        "alto para outliers isolados; pode indicar erro de coleta ou subpopulação distinta"
    ),
    "rec.outliers.explanation": (
        "{count} outlier(s) detectado(s) ({pct} das observações, método {method}) "
        "— percentual {reason}. Outliers não devem ser removidos automaticamente, "
        "pois podem representar informação legítima do domínio."
    ),

    # --- recommendations: descritivas ---------------------------------
    "rec.duplicate_rows.recommendation": "Remover linhas duplicadas antes de qualquer análise/modelagem.",
    "rec.duplicate_rows.explanation": (
        "{count} linha(s) duplicada(s) ({pct} do dataset). Linhas duplicadas "
        "distorcem estatísticas descritivas e podem causar vazamento de dados "
        "entre treino e teste se não forem removidas antes da divisão."
    ),
    "rec.id_column.recommendation": "Avaliar remover '{column}' de análises de correlação e da modelagem.",
    "rec.id_column.explanation": (
        "Coluna identificada como possível identificador (proporção de valores "
        "únicos próxima de 100%). Identificadores não carregam relação "
        "causal/preditiva com o target; incluí-los pode gerar correlações espúrias."
    ),
    "rec.skewed.recommendation_log": "Considerar aplicar transformação logarítmica em '{column}'.",
    "rec.skewed.recommendation_yeo": (
        "Considerar aplicar transformação Yeo-Johnson em '{column}' (há valores <= 0)."
    ),
    "rec.skewed.explanation": (
        "Assimetria de {skewness:.2f}. Distribuições fortemente assimétricas "
        "violam a suposição de normalidade de vários modelos e métricas; a "
        "transformação aproxima a distribuição de uma forma mais simétrica."
    ),
    "rec.high_cardinality.recommendation": (
        "Avaliar Frequency Encoding, Target Encoding ou agrupamento de categorias "
        "raras em '{column}' (ou remoção, caso seja possível identificador)."
    ),
    "rec.high_cardinality.explanation": (
        "Cardinalidade alta ({unique_count} categorias). One-hot encoding em "
        "colunas de alta cardinalidade gera um número excessivo de novas colunas "
        "esparsas; técnicas de encoding baseadas em frequência/target ou o "
        "agrupamento de categorias raras em 'outros' tendem a generalizar melhor."
    ),
    "rec.constant.recommendation": "Remover a coluna '{column}' do dataset.",
    "rec.constant.explanation": (
        "Coluna constante (um único valor em toda a amostra). Uma coluna "
        "constante tem variância zero e não pode, por definição, contribuir "
        "para separar as classes do target."
    ),
    "rec.near_zero_variance.recommendation": (
        "Avaliar remover '{column}' ou tratá-la como de baixo poder informativo."
    ),
    "rec.near_zero_variance.explanation": (
        "Coluna quase constante: um único valor responde por {pct} das "
        "observações. Carrega pouca informação para separar as classes, mesmo "
        "sem variância tecnicamente zero, e pode instabilizar modelos "
        "sensíveis a features de baixa variância."
    ),
    "rec.mixed_type.recommendation": (
        "Padronizar o formato de '{column}' (ex.: converter valores por extenso "
        "para número, ou tratá-los como categoria 'inválido') antes de qualquer análise."
    ),
    "rec.mixed_type.explanation": (
        "Coluna mistura valores numéricos ({numeric_pct}) e não numéricos "
        "({non_numeric_pct}). Costuma indicar erro de digitação ou de "
        "exportação; sem correção, a coluna é mal classificada e os cálculos "
        "estatísticos ficam distorcidos."
    ),

    # --- recommendations: correlação -----------------------------------
    "rec.correlation_pair.recommendation": (
        "Avaliar remover uma das colunas ('{column_a}' ou '{column_b}') ou "
        "combiná-las em uma única feature."
    ),
    "rec.correlation_pair.explanation": (
        "'{column_a}' e '{column_b}' têm correlação {method} de {corr:.2f}. "
        "Colunas fortemente correlacionadas carregam informação redundante; "
        "mantê-las ambas aumenta a multicolinearidade sem ganho proporcional de sinal."
    ),
    "rec.vif.recommendation": (
        "Avaliar remover '{column}' ou reduzir a dimensionalidade do grupo de "
        "variáveis redundantes (ex.: PCA) antes de um modelo linear."
    ),
    "rec.vif.explanation": (
        "VIF de {vif}. VIF alto indica que a variável é quase uma combinação "
        "linear de outras variáveis do dataset — diferente da correlação par a "
        "par, o VIF captura redundância multivariada, mesmo quando nenhum par "
        "isolado parece fortemente correlacionado."
    ),
    "rec.scale_disparity.recommendation": (
        "Padronizar (StandardScaler) ou normalizar as variáveis numéricas antes "
        "de modelos sensíveis a escala (ex.: KNN, SVM, regressão com "
        "regularização L1/L2)."
    ),
    "rec.scale_disparity.explanation": (
        "'{largest_column}' (desvio padrão {largest_std:.2f}) está em escala "
        "muito maior que '{smallest_column}' (desvio padrão {smallest_std:.2f}), "
        "razão de {ratio:.0f}x. Variáveis em escalas muito diferentes dominam o "
        "cálculo de distância ou o termo de regularização apenas por causa da "
        "magnitude, não porque carregam mais sinal."
    ),

    # --- recommendations: target -----------------------------------------
    "rec.target_imbalance.recommendation": (
        "Considerar reamostragem (over/undersampling ou SMOTE, aplicados "
        "somente no conjunto de treino) ou pesos de classe (class_weight) na "
        "etapa de modelagem."
    ),
    "rec.target_imbalance.explanation": (
        "Classes desbalanceadas: '{majority_class}' ({majority_pct}) vs "
        "'{minority_class}' ({minority_pct}), razão {ratio:.1f}:1. Acurácia "
        "isoladamente pode ser enganosa nesse cenário — prefira métricas como "
        "F1, recall da classe minoritária ou AUC-ROC."
    ),
    "rec.target_leakage.recommendation": (
        "Investigar se '{predictor}' é uma proxy do próprio target (ex.: "
        "preenchida após o evento que o target representa) antes de usá-la "
        "como preditora."
    ),
    "rec.target_leakage.explanation": (
        "Associação muito forte com o target ({metric} = {association:.2f}). "
        "Associação quase perfeita é mais consistente com vazamento de "
        "informação do que com um preditor legítimo — incluí-la infla "
        "artificialmente o desempenho do modelo em treino/validação sem "
        "generalizar para produção."
    ),
    "rec.target_strong_predictors.recommendation": (
        "Priorizar essas variáveis na seleção de features do modelo."
    ),
    "rec.target_strong_predictors.explanation": (
        "{n} variável(is) com associação forte ao target: {predictors}. "
        "Variáveis com associação forte (Point-Biserial, V de Cramér ou "
        "Spearman elevados) tendem a carregar mais sinal preditivo."
    ),
    "rec.target_multiple_comparisons.recommendation": (
        "Interpretar os p-valores individuais dos testes de associação com "
        "cautela; preferir os preditores com maior força de associação (não só "
        "significância) na seleção de features."
    ),

    # --- recommendations: data leakage (workflow) -------------------------
    "rec.data_leakage.recommendation": (
        "Separar treino e teste antes de calcular qualquer estatística de "
        "preparação de dados; ajustar imputação, normalização, seleção de "
        "features e SMOTE apenas no conjunto de treino, e aplicar as mesmas "
        "transformações (já ajustadas) ao conjunto de teste."
    ),
    "rec.data_leakage.explanation": (
        "As estatísticas e recomendações deste relatório foram calculadas "
        "sobre o dataset completo, para fins de diagnóstico exploratório. Usar "
        "médias, medianas, categorias ou parâmetros de reamostragem calculados "
        "sobre o conjunto de teste (ou sobre o dataset inteiro) para preparar "
        "os dados antes da divisão treino/teste é uma forma comum de "
        "vazamento de dados: o modelo passa a ter acesso indireto a "
        "informação do teste durante o treinamento, inflando métricas de "
        "validação de forma não realista."
    ),

    # --- report/builder: estrutura do relatório ----------------------------
    "report.title": "Relatório AutoEDA",
    "report.summary": (
        "Dataset: {rows} linhas × {cols} colunas. Variável alvo: '{target}' "
        "(classificação binária)."
    ),
    "report.table.metric": "Métrica",
    "report.table.value": "Valor",
    "report.table.na": "N/D",

    "report.section1.title": "1. Visão geral do dataset",
    "report.section1.rows": "Linhas",
    "report.section1.columns": "Colunas",
    "report.section1.memory": "Uso de memória",
    "report.section1.duplicate_rows": "Linhas duplicadas",
    "report.section1.missing_cells": "Células ausentes (total)",

    "report.section2.title": "2. Variável alvo: '{target}'",
    "report.section2.majority_class": "Classe majoritária",
    "report.section2.minority_class": "Classe minoritária",
    "report.section2.imbalance_ratio": "Imbalance ratio",
    "report.section2.imbalance_warning": (
        "⚠️ Classes desbalanceadas — veja a recomendação correspondente na seção 8."
    ),

    "report.section3.title": "3. Valores ausentes",
    "report.section3.no_missing": "Nenhum valor ausente encontrado no dataset.",
    "report.section3.table.column": "Coluna",
    "report.section3.table.pct_missing": "% ausente",
    "report.section3.table.severity": "Severidade",
    "report.section3.table.mechanism_hint": "Indício de mecanismo",
    "report.section3.mechanism_note": (
        "Nota: MCAR, MAR e MNAR não podem ser determinados com certeza apenas "
        "a partir dos dados. \"MAR\" acima indica indício (ausência "
        "correlacionada com outra coluna, ou com as classes do target); "
        "\"indeterminado\" não confirma MCAR — MNAR nunca pode ser descartado "
        "só pelos dados observados."
    ),

    "report.section4.title": "4. Estatísticas descritivas",
    "report.section4.numeric_subtitle": "Variáveis numéricas",
    "report.section4.categorical_subtitle": "Variáveis categóricas",
    "report.section4.constant_subtitle": "Variáveis constantes / quase-constantes",
    "report.section4.mixed_type_subtitle": "Colunas com tipo misto",
    "report.section4.table.column": "Coluna",
    "report.section4.table.mean": "Média",
    "report.section4.table.median": "Mediana",
    "report.section4.table.std": "Desvio",
    "report.section4.table.min": "Mín",
    "report.section4.table.max": "Máx",
    "report.section4.table.iqr": "IQR",
    "report.section4.table.skewness": "Assimetria",
    "report.section4.table.pct_outliers": "% Outliers",
    "report.section4.table.categories": "Categorias",
    "report.section4.table.dominant_category": "Categoria dominante",
    "report.section4.table.pct_dominant": "% dominante",
    "report.section4.table.pct_missing": "% ausente",
    "report.section4.table.type": "Tipo",
    "report.section4.table.pct_dominant_value": "% valor dominante",
    "report.section4.table.pct_numeric": "% numérico",
    "report.section4.table.pct_non_numeric": "% não numérico",
    "report.section4.constant_label": "constante",
    "report.section4.near_constant_label": "quase-constante",

    "report.section5.title": "5. Outliers",
    "report.section5.no_outliers": "Nenhum outlier relevante detectado nas variáveis numéricas.",
    "report.section5.method_note": (
        "Método: {method}. Outliers não são removidos automaticamente — podem "
        "representar informação legítima do domínio."
    ),
    "report.section5.table.column": "Coluna",
    "report.section5.table.count": "Qtd. outliers",
    "report.section5.table.pct": "% outliers",

    "report.section6.title": "6. Correlação e multicolinearidade",
    "report.section6.strong_corr_subtitle": "Pares fortemente correlacionados",
    "report.section6.vif_subtitle": "VIF (Variance Inflation Factor) alto",
    "report.section6.scale_subtitle": "Disparidade de escala",
    "report.section6.scale_text": (
        "'{largest_column}' está em escala {ratio:.0f}x maior que "
        "'{smallest_column}'. Considerar padronização."
    ),
    "report.section6.table.column_a": "Coluna A",
    "report.section6.table.column_b": "Coluna B",
    "report.section6.table.method": "Método",
    "report.section6.table.correlation": "Correlação",
    "report.section6.table.column": "Coluna",
    "report.section6.table.vif": "VIF",

    "report.section7.title": "7. Relação das variáveis com o target '{target}'",
    "report.section7.table.predictor": "Preditor",
    "report.section7.table.type": "Tipo",
    "report.section7.table.technique": "Técnica",
    "report.section7.table.strength": "Força",
    "report.section7.table.p_value": "p-valor",
    "report.section7.excluded_subtitle": "Preditores excluídos da análise",
    "report.section7.table.reason": "Motivo",
    "report.section7.leakage_warning": "⚠️ **Possível vazamento de dados detectado** — veja a seção 8.",

    "report.section8.title": "8. Recomendações",
    "report.section8.json_link": "Versão completa em JSON:",
    "report.section8.severity_high": "Severidade Alta",
    "report.section8.severity_medium": "Severidade Média",
    "report.section8.severity_low": "Severidade Baixa",
    "report.section8.table.feature": "Feature",
    "report.section8.table.problem": "Problema",
    "report.section8.table.recommendation": "Recomendação",
    "report.section8.dataset_placeholder": "_dataset_",

    "report.no_data": "_Sem dados para esta seção._",
}
