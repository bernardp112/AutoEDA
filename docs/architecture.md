# Arquitetura do AutoEDA

Este documento descreve as decisões de design do projeto, para quem for dar manutenção ou continuar o desenvolvimento. Para uso da biblioteca, ver o [README](../README.md).

## Visão geral do pipeline

```
DataFrame + target
      │
      ▼
utils.validate_dataframe / validate_target / validate_language   (falha cedo)
      │
      ▼
AutoEDAConfig  (thresholds: severidade, outlier, correlação, vazamento, etc.)
      │
      ├──► analysis.descriptive.generate_descriptive_stats
      ├──► analysis.missing_values.analyze_missing_values
      ├──► analysis.outliers.analyze_outliers
      ├──► analysis.correlation.analyze_correlation
      └──► analysis.target_analysis.analyze_target
      │
      ▼
recommendations.generate_recommendations   (traduz achados em ações)
      │
      ├──► report.charts.generate_all_charts        (.png)
      ├──► report.json_export.export_recommendations_json  (.json)
      └──► report.builder.export_markdown_report     (.md)
      │
      ▼
core.AutoEDAResult  (dataclass com todos os resultados + caminhos de arquivo)
```

Cada módulo de `analysis/` recebe o `DataFrame` (já validado) e a `AutoEDAConfig`, e devolve um `dict` documentado na docstring da própria função pública do módulo (`generate_descriptive_stats`, `analyze_missing_values`, etc.). Esses dicts são a "fonte da verdade" — `recommendations.py` e `report/` só consomem, nunca recalculam.

## Por que separar analysis/ de recommendations/

`analysis/*.py` **descreve e classifica** (ex.: "esta coluna tem 22% de ausência, severidade moderada"). `recommendations.py` **decide o que sugerir** a partir dessa classificação (ex.: "imputar com mediana"). Essa separação existe porque:

- os thresholds de severidade (`AutoEDAConfig`) são aplicados uma única vez, dentro de `analysis/`, evitando que a mesma regra de negócio seja reimplementada em dois lugares;
- permite consumir os resultados de `analysis/` sem as recomendações (ex.: alguém que só quer as estatísticas em JSON);
- facilita testar cada camada isoladamente — um teste de `analysis/missing_values.py` não precisa saber nada sobre como o texto da recomendação é formatado.

## Tipagem lógica de colunas (`utils.infer_column_types`)

Quase todo módulo de `analysis/` começa chamando `infer_column_types(df, categorical_max_cardinality, id_cardinality_ratio_threshold)`, que classifica cada coluna em um de: `numeric`, `categorical`, `datetime`, `boolean`, `text`, `id`. Essa classificação (não o `dtype` do pandas) é o que orienta qual técnica cada módulo aplica — por exemplo, `target_analysis.py` decide entre Point-Biserial, Qui-quadrado/V de Cramér ou Spearman com base nesse tipo lógico, não no dtype bruto.

Ponto de atenção: colunas `categorical` com dtype numérico (ex.: nota de 1 a 5) são tratadas como **ordinais** em `target_analysis.py` (usa Spearman); colunas `categorical` com dtype de texto são tratadas como **nominais** (usa Qui-quadrado/V de Cramér). Essa distinção é heurística (baseada no dtype), não uma detecção real de ordinalidade — o AutoEDA não tem como saber se `["baixo","médio","alto"]` é ordinal sem essa informação vir do dtype ou de configuração explícita.

## Sistema de configuração (`AutoEDAConfig`)

Todo threshold usado pelo pipeline (limite de severidade de ausência, multiplicador do IQR, correlação "alta", VIF "alto", nível de significância, etc.) é um campo de `AutoEDAConfig` (dataclass em `config.py`), nunca um número mágico espalhado pelo código. Isso permite:

- ajustar o comportamento sem editar código-fonte;
- rastrear facilmente todo threshold do projeto num único arquivo;
- testar o mesmo dataset com configurações diferentes.

`AutoEDAConfig.language` também mora aqui — é o campo que decide qual catálogo do `i18n/` é usado em todo o pipeline.

## Internacionalização (`i18n/`)

Todo texto gerado dinamicamente (recomendações, indícios de mecanismo de ausência, aviso de múltiplas comparações, estrutura do relatório) vem de `i18n/pt_br.py` / `i18n/en_us.py`, nunca de string literal hardcoded nos módulos de lógica. O padrão em cada módulo é:

```python
from autoeda.i18n import get_translator

t = get_translator(config.language)
mensagem = t("rec.missing_high.recommendation", column=nome_da_coluna)
```

As chaves usam namespace por ponto (`rec.*` para recomendações, `report.*` para o relatório, `missing.*`/`target.*` para textos de análise) para indicar a origem. Os dois catálogos precisam ter exatamente o mesmo conjunto de chaves — `i18n.assert_catalogs_in_sync()` verifica isso e deve ser chamada em qualquer suíte de testes antes de um release.

**Limitação conhecida**: adicionar um 3º idioma exige adicionar um novo arquivo de catálogo espelhando todas as chaves de `pt_br.py`/`en_us.py` — não há fallback automático de tradução parcial.

## Schema das recomendações

Cada recomendação segue um schema fixo (definido a partir do formato pedido pelo orientador do projeto):

```json
{
  "feature": "renda",
  "problem_type": "missing_values",
  "metric": "missing_percentage",
  "metric_value": 22.5,
  "severity": "high",
  "recommendation": "Considerar remover a coluna 'renda' do dataset.",
  "explanation": "Coluna com 22.5% de valores ausentes. ..."
}
```

`feature` é `None` para recomendações de nível de dataset inteiro (ex.: linhas duplicadas, alerta de Data Leakage). O JSON final (`recommendations.json`) envolve a lista em `{"schema_version": "1.0", "recommendations": [...]}"` — a versão do schema existe para que consumidores externos detectem mudanças de formato no futuro.

## Geração de gráficos (`report/charts.py`)

Os gráficos são gerados com Matplotlib (`backend="Agg"`, sem necessidade de display) e salvos como `.png` em `output_dir/charts/`. `report/builder.py` referencia essas imagens por **caminho relativo** ao `.md` (via `os.path.relpath`), para o relatório continuar funcionando se a pasta de saída inteira for movida ou copiada.

Para evitar gerar dezenas de gráficos irrelevantes em datasets largos, `generate_all_charts` pula colunas constantes e limita o número de colunas plotadas por tipo (`config.max_charted_columns_per_type`), reportando as omitidas em `charts["omitted_columns"]`.

## Erros e validação

Toda exceção do AutoEDA herda de `AutoEDAError` (`exceptions.py`). A validação de entrada (`utils.validate_dataframe`, `validate_target`, `validate_language`) roda **antes** de qualquer análise, então uma entrada inválida nunca chega a executar parte do pipeline e falhar de forma confusa no meio do caminho.

## Limitações conhecidas / trabalho futuro

- **Classificação binária apenas**: multiclasse e regressão estão fora do escopo atual (decisão do orientador do projeto).
- **Testes automatizados**: `tests/` ainda não tem casos formais em `pytest` — a validação até aqui foi feita via scripts ad-hoc durante o desenvolvimento. Portar esses cenários para `pytest` é o próximo passo natural.
- **Validação com datasets públicos reais**: ainda não foi feita (ver seção de validação do escopo original do projeto).
- **Ordinalidade detectada por dtype, não por metadado explícito** (ver seção "Tipagem lógica de colunas" acima).
