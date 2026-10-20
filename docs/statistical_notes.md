# Notas estatísticas do AutoEDA

Este documento descreve, para cada técnica usada, o que ela mede, as premissas e as **limitações conhecidas**. Todos os limiares citados são heurísticas configuráveis em `AutoEDAConfig`, não testes estatísticos.

## Associação com o alvo

| Tipo da variável | Técnica | O que mede |
|---|---|---|
| Numérica | Correlação Point-Biserial | Associação **linear** e direção (equivale à correlação de Pearson com o alvo codificado em 0/1), com p-valor |
| Numérica | Informação Mútua (`mutual_info_classif`) | Dependência estatística em geral, inclusive não linear e não monotônica; sem p-valor; sem limite superior fixo (não comparável em escala com os demais) |
| Categórica nominal | Qui-quadrado + V de Cramér | Significância (qui-quadrado) e força (V, de 0 a 1) da associação |
| Categórica "ordinal" | Correlação de Spearman | Associação **monotônica** com o alvo |

### V de Cramér

- É calculado a partir do qui-quadrado **de Pearson, sem correção de continuidade**, com a correção de viés de Bergsma (2013), que **reduz** (não elimina) a superestimação em tabelas pequenas ou esparsas.
- O scipy aplica por padrão a correção de Yates em tabelas 2x2; o AutoEDA a usa apenas no p-valor, não no V (o V usa o qui-quadrado sem correção).
- O V é normalizado pelo tamanho da amostra e pelo número de categorias, o que o torna comparável entre tabelas de tamanhos diferentes, mas **não é direcional** e não indica *qual* categoria se associa a qual classe.
- O teste qui-quadrado pressupõe frequências esperadas razoáveis (a regra usual é pelo menos 5 por célula); em categorias muito raras, o p-valor é aproximado.

### Ordinalidade

A distinção entre variável ordinal e nominal é feita **pelo dtype**: coluna categórica com dtype numérico é tratada como ordinal (Spearman). É uma heurística: o AutoEDA não sabe se `["baixo", "médio", "alto"]` (texto) é ordinal.

### Comparações múltiplas e p-valores

Quando mais de 10 preditores são testados, o AutoEDA alerta e informa o nível de significância corrigido de Bonferroni (α/n). Em conjuntos grandes, associações muito fracas podem ser estatisticamente significativas; por isso o relatório mostra também a **força** da associação, e o ranking usa a força, não o p-valor.

## Possível vazamento do alvo (*target leakage*)

O AutoEDA sinaliza uma variável como **candidata a vazamento** por dois critérios, ambos configuráveis:

1. associação principal (correlação ou V de Cramér) em módulo ≥ `leakage_association_threshold` (0,95); ou
2. **AUC de variável única** ≥ `leakage_auc_threshold` (0,99), com pelo menos `leakage_auc_min_observations` (30) observações pareadas. A AUC é calculada usando a própria variável como escore e reportada como `max(AUC, 1 − AUC)`, para independer da direção da relação.

### O que a heurística captura

- Relações **monotônicas**, inclusive de limiar: um alvo definido como `x > 0` tem AUC 1,0 mesmo com correlação de Point-Biserial moderada (no College Major ROI, 0,46).

### O que a heurística **não** captura

- **Relações não monotônicas** (ex.: alvo = `|x| > 1`): a AUC fica próxima de 0,5, embora uma variável separe as classes. A Informação Mútua poderia indicar essas relações, mas não é usada como critério de vazamento.
- **Vazamento por combinação de variáveis** (nenhuma delas sozinha separa as classes).
- **Variáveis categóricas nominais por AUC**: o critério de AUC só é calculado para variáveis numéricas e ordinais; as nominais dependem apenas do V de Cramér.
- **Contaminação entre treino e teste**: o AutoEDA analisa o dataset completo, para fins de diagnóstico, e não tem como detectar esse tipo de vazamento. Por isso emite sempre a recomendação de separar treino e teste antes de ajustar qualquer transformação.

### Falsos positivos e falsos negativos

- Uma AUC alta **não implica** vazamento: um preditor legítimo muito forte também a produz (no Breast Cancer, a maior AUC foi 0,975, abaixo do limiar, mas valores próximos seriam esperados em dados clínicos legítimos). Por isso a saída é redigida como "avaliar se a variável é proxy do alvo".
- Uma AUC baixa **não garante** ausência de vazamento (ver acima).
- Com poucas observações, uma AUC alta ocorre por acaso: com 12 observações e 200 variáveis de ruído, a maior AUC chegou a cerca de 0,97. Daí o mínimo de 30 observações.
- A AUC é calculada **na própria amostra**, sem validação cruzada e sem intervalo de confiança.

## Multicolinearidade

- **Correlação par a par** (Pearson e Spearman) e **VIF**, calculado por mínimos quadrados de cada variável contra as demais. VIF ≥ 5 é sinalizado; colinearidade perfeita resulta em VIF infinito (tratado como valor válido no JSON).
- Em dados com identidades algébricas (como o College Major ROI), o VIF chega a dezenas de milhões: é o diagnóstico esperado, mas pouco informativo em magnitude.
- Em conjuntos com muitas variáveis redundantes, o relatório emite um alerta por par correlacionado, o que gera muitas linhas (132 recomendações no Breast Cancer).

## Valores ausentes

MCAR, MAR e MNAR **não podem ser determinados** só com os dados observados. O AutoEDA reporta apenas indícios (ex.: a ausência de uma coluna está associada à ausência de outra, ou às classes do alvo, o que aponta para MAR) e nunca afirma o mecanismo.

## Outliers

IQR (limites Q1 − 1,5·IQR e Q3 + 1,5·IQR) ou Z-score (|z| > 3). Outliers **não são removidos**: podem ser erros ou fenômenos legítimos. O Z-score é sensível aos próprios outliers (média e desvio inflados).

## Variáveis constantes e quase constantes

- Constante: apenas um valor distinto.
- Quase constante: razão de frequência (valor mais comum / segundo mais comum) ≥ 19, proporção de valores únicos ≤ 10% **e** o valor mais comum concentrando ≥ 90% das observações. O terceiro critério evita marcar como quase constante uma variável contínua inflada por zeros (ex.: 38% de zeros e o restante disperso).

## Identificadores

Coluna não flutuante com proporção de valores únicos ≥ 0,95 (e pouca ausência) é tratada como possível identificador.
