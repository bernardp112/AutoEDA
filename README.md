# AutoEDA

Biblioteca Python de **Análise Exploratória de Dados (EDA) automatizada**, com foco em problemas de **classificação binária**.

Diferente de ferramentas de EDA que só descrevem os dados (estatísticas, gráficos), o AutoEDA vai um passo além: **interpreta** os resultados e gera **recomendações de pré-processamento** acionáveis, sempre em linguagem de sugestão ("considerar", "avaliar") — nunca instruções absolutas, já que decisões de preparação de dados dependem de contexto e conhecimento de domínio.

## O problema que o AutoEDA resolve

Bibliotecas como o `ydata-profiling` são excelentes para descrever um dataset (média, mediana, distribuição, correlação), mas param aí — cabe ao analista interpretar os números e decidir o que fazer. O AutoEDA automatiza essa segunda etapa: recebe o dataset e a variável alvo, e devolve um relatório com **o que foi encontrado** e **o que fazer a respeito**, com a justificativa de cada recomendação.

## Instalação

```bash
pip install -r requirements.txt
# ou, em modo desenvolvimento (instala o pacote 'autoeda' via pyproject.toml):
pip install -e .
```

Requer Python 3.10+.

## Uso básico

```python
import pandas as pd
from autoeda import autoeda

df = pd.read_csv("meu_dataset.csv")

result = autoeda(df, target="inadimplente", lang="pt-br")

print(result.report_path)                 # relatório completo em Markdown
print(result.recommendations_json_path)    # recomendações em JSON

for rec in result.recommendations["recommendations"]:
    print(f"[{rec['severity']}] {rec['recommendation']}")
```

Por padrão, os arquivos de saída (relatório `.md`, gráficos `.png`, `recommendations.json`) são salvos em `./autoeda_output/`. Para mudar:

```python
result = autoeda(df, target="inadimplente", output_dir="minha_pasta_de_saida")
```

Para gerar o relatório em inglês:

```python
result = autoeda(df, target="inadimplente", lang="en-us")
```

## O que o AutoEDA analisa

| Módulo | O que faz |
|---|---|
| **Estatísticas descritivas** | Métricas por coluna (média, mediana, IQR, assimetria/curtose para numéricas; cardinalidade, categoria dominante para categóricas), variáveis constantes/quase-constantes, colunas de tipo misto |
| **Valores ausentes** | Severidade por coluna, indícios de mecanismo (MCAR/MAR/MNAR — sempre como indício, nunca confirmação), ausência condicionada às classes do target |
| **Outliers** | Detecção via IQR (ou Z-score), sem remoção automática |
| **Correlação / multicolinearidade** | Pearson, Spearman, VIF (Variance Inflation Factor), disparidade de escala entre variáveis |
| **Relação com o target** | Point-Biserial + Mutual Information (numérica), Qui-quadrado + V de Cramér (categórica nominal), Spearman (categórica ordinal), desbalanceamento de classes, alerta de possível vazamento de dados |

## Escopo e limitações (por design)

- **Restrito a classificação binária**: a coluna alvo precisa ter exatamente 2 classes. Datasets multiclasse ou de regressão não são suportados nesta versão.
- **Não remove nada automaticamente**: toda recomendação é uma sugestão para o usuário avaliar (remoção de coluna, imputação, encoding, etc.).
- **Não decide o mecanismo de dados ausentes com certeza**: MCAR/MAR/MNAR nunca podem ser determinados só a partir dos dados — o AutoEDA reporta indícios, não conclusões.
- **Alerta de Data Leakage é um lembrete de processo**, não uma detecção: o AutoEDA calcula estatísticas sobre o dataset completo (para fins de diagnóstico exploratório) e sempre lembra o usuário de recalcular tudo apenas no conjunto de treino antes de aplicar ao teste.

## Estrutura do projeto

```
src/autoeda/
├── core.py              # função pública autoeda() — ponto de entrada da API
├── config.py             # thresholds e configurações (AutoEDAConfig)
├── exceptions.py          # exceções customizadas
├── utils.py               # validação de entrada e inferência de tipo de coluna
├── analysis/
│   ├── descriptive.py         # estatísticas descritivas
│   ├── missing_values.py      # valores ausentes e indícios de mecanismo
│   ├── outliers.py            # detecção de outliers (IQR/Z-score)
│   ├── correlation.py         # correlação, VIF, disparidade de escala
│   └── target_analysis.py     # relação de cada variável com o target binário
├── recommendations.py     # motor de recomendações (consome analysis/, gera o JSON)
├── report/
│   ├── charts.py              # gráficos (Matplotlib)
│   ├── json_export.py         # exportação do JSON de recomendações
│   └── builder.py             # montagem do relatório final em Markdown
└── i18n/                   # catálogos de texto pt-br / en-us
    ├── pt_br.py
    └── en_us.py
```

Fluxo de dados: `core.autoeda()` valida a entrada → roda os módulos de `analysis/` → `recommendations.py` traduz os resultados em ações → `report/` gera os artefatos finais (gráficos, JSON, Markdown).

## Desenvolvimento

```bash
pip install -e ".[dev]"
pytest
```

## Licença

MIT.
