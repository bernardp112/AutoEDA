# AutoEDA

Biblioteca Python de **Análise Exploratória de Dados (EDA) automatizada**, com foco em problemas de **classificação binária**.

Diferente de ferramentas de EDA que só descrevem os dados (estatísticas, gráficos), o AutoEDA vai um passo além: **interpreta** os resultados e gera **recomendações de pré-processamento** acionáveis, sempre em linguagem de sugestão ("considerar", "avaliar") — nunca instruções absolutas, já que decisões de preparação de dados dependem de contexto e conhecimento de domínio.

## O problema que o AutoEDA resolve

Bibliotecas como o `ydata-profiling` são excelentes para descrever um dataset (média, mediana, distribuição, correlação), mas param aí — cabe ao analista interpretar os números e decidir o que fazer. O AutoEDA automatiza essa segunda etapa: recebe o dataset e a variável alvo, e devolve um relatório com **o que foi encontrado** e **o que fazer a respeito**, com a justificativa de cada recomendação.

## Instalação

O AutoEDA precisa ser **instalado como pacote** — não basta copiar um script solto para outra pasta, ele precisa achar o pacote `autoeda` (a pasta `src/autoeda/`) em algum lugar do seu ambiente Python.

1. **Extraia o projeto inteiro** (não só um arquivo de exemplo) em uma pasta — o nome da pasta não importa, pode ser `AutoEDA`, `autoeda-project`, o que preferir. O importante é que dentro dela exista `pyproject.toml` e a subpasta `src/autoeda/`.

2. **Abra um terminal dentro dessa pasta** (onde está o `pyproject.toml`):

   ```bash
   cd caminho/para/AutoEDA
   ```

3. **Instale o pacote em modo desenvolvimento**:

   ```bash
   pip install -e .
   ```

   Isso lê o `pyproject.toml`, instala as dependências (pandas, numpy, matplotlib, scipy, scikit-learn) e registra o pacote `autoeda` no seu Python — depois disso, `from autoeda import autoeda` funciona de **qualquer lugar**, não só de dentro dessa pasta.

Requer Python 3.10+.

> **Erro comum**: `ModuleNotFoundError: No module named 'autoeda'` acontece quando um script (como os de `examples/`) é copiado para outra pasta/ambiente sem o pacote ter sido instalado antes. A solução é sempre o passo 3 acima — depois de instalado, o script pode morar em qualquer pasta.
>
> Alternativa rápida para teste, sem instalar o pacote (não recomendada para uso contínuo): rodar a partir da raiz do projeto com o `PYTHONPATH` apontando para `src`:
> ```bash
> # Linux/macOS
> PYTHONPATH=src python examples/example_real_dataset.py
> # Windows (PowerShell)
> $env:PYTHONPATH = "src"; python examples\example_real_dataset.py
> ```

## Demonstração rápida

Para ver o AutoEDA funcionando sem precisar de nenhum arquivo de dados, rode o script de demonstração (com o pacote instalado):

```bash
python examples/example_basic.py                 # relatório em português
python examples/example_basic.py --lang en-us    # relatório em inglês
```

Ele gera um dataset sintético com problemas conhecidos (valores ausentes, outliers, coluna constante, identificador, variáveis redundantes e uma variável que "vaza" o alvo), executa a análise e imprime as recomendações de severidade alta e média. Os artefatos ficam em `autoeda_demo_output/` (`report.md`, `recommendations.json` e a pasta `charts/`).

Para usar um arquivo seu (CSV, Excel, Parquet, JSON etc.): `python examples/example_real_dataset.py --input meus_dados.csv --target coluna_alvo`.

Para reproduzir a validação do relatório técnico, ver [docs/validation.md](docs/validation.md).

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
| **Relação com o target** | Point-Biserial + Mutual Information (numérica), Qui-quadrado + V de Cramér (categórica nominal), Spearman (categórica ordinal), desbalanceamento de classes, indícios de possível vazamento do alvo (associação quase perfeita ou AUC de variável única muito alta) |

## Escopo e limitações (por design)

- **Restrito a classificação binária**: a coluna alvo precisa ter exatamente 2 classes. Datasets multiclasse ou de regressão não são suportados nesta versão.
- **Não remove nada automaticamente**: toda recomendação é uma sugestão para o usuário avaliar (remoção de coluna, imputação, encoding, etc.), redigida em linguagem de sugestão ("considerar", "avaliar").
- **Não decide o mecanismo de dados ausentes com certeza**: MCAR/MAR/MNAR nunca podem ser determinados só a partir dos dados — o AutoEDA reporta indícios, não conclusões.
- **Vazamento de dados: o AutoEDA aponta candidatas, não prova.** Ele sinaliza variáveis cuja associação com o alvo é quase perfeita (correlação ou V de Cramér ≥ 0,95) ou cuja AUC isolada é ≥ 0,99 (com pelo menos 30 observações). Isso **não** detecta contaminação entre treino e teste (o AutoEDA analisa o dataset completo, para diagnóstico) nem vazamento que dependa da combinação de várias variáveis ou de relações não monotônicas. A recomendação de separar treino e teste antes de ajustar qualquer transformação é sempre emitida.
- **Os limiares são heurísticas**, não testes estatísticos: todos são configuráveis em `AutoEDAConfig`.

Detalhes e limites de cada técnica estatística: [docs/statistical_notes.md](docs/statistical_notes.md).

## Estrutura do projeto

```
examples/
├── example_basic.py           # demonstração com dados sintéticos (sem arquivos externos)
└── example_real_dataset.py    # carrega CSV/Excel/Parquet/JSON e roda a análise
docs/                          # arquitetura, notas estatísticas, validação
tests/                         # suíte pytest (121 casos de teste coletados)
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
pytest                                   # 121 testes
pytest --cov=autoeda --cov-report=term   # cobertura (94% das instruções na última medição)
```

Os testes usam apenas dados sintéticos gerados no próprio código (com semente fixa), sem depender de arquivos externos.

## Documentação

- [docs/architecture.md](docs/architecture.md) — arquitetura e decisões de design.
- [docs/statistical_notes.md](docs/statistical_notes.md) — técnicas estatísticas, premissas e limitações.
- [docs/validation.md](docs/validation.md) — como reproduzir a validação com conjuntos públicos.
- [CHANGELOG.md](CHANGELOG.md) — histórico de mudanças.

## Licença

MIT (ver [LICENSE](LICENSE)).
