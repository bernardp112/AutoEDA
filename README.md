# autoeda

> **Automated Exploratory Data Analysis Library for Python**

`autoeda` é uma biblioteca Python desenvolvida para automatizar e padronizar a Análise Exploratória de Dados (EDA) em datasets estruturados (dados tabulares), orientando o analista desde as estatísticas descritivas até recomendações de transformação de dados.

---

## 🎯 Funcionalidades

- **Análise estatística descritiva** — resumo completo de variáveis numéricas e categóricas
- **Valores ausentes** — detecção, quantificação e visualização de missings
- **Outliers** — identificação por IQR, Z-score e outros critérios
- **Correlações** — matriz de correlação e análise de multicolinearidade
- **Análise por variável alvo** — relação das features com a variável de interesse
- **Análise temporal** — tendências e sazonalidade básica (quando aplicável)
- **Recomendações** — sugestões de transformações de dados baseadas na análise

---

## 📦 Instalação

```bash
# Instalar em modo de desenvolvimento (recomendado durante o desenvolvimento do TCC)
pip install -e ".[dev]"
```

---

## 🚀 Uso rápido

```python
import pandas as pd
from autoeda import AutoEDA

df = pd.read_csv("seu_dataset.csv")

eda = AutoEDA(df, target="coluna_alvo")
eda.run()
eda.report()
```

---

## 🗂️ Estrutura do Projeto

```
autoeda/
├── src/
│   └── autoeda/
│       ├── analyzers/          # Módulos de análise
│       │   ├── descriptive.py  # Estatísticas descritivas
│       │   ├── missing.py      # Valores ausentes
│       │   ├── outliers.py     # Detecção de outliers
│       │   ├── correlation.py  # Análise de correlação
│       │   ├── target.py       # Análise vs. variável alvo
│       │   └── temporal.py     # Análise temporal
│       ├── reporters/
│       │   └── report_generator.py  # Geração de relatórios
│       ├── recommenders/
│       │   └── transformations.py   # Recomendações de transformações
│       └── utils/
│           ├── validators.py   # Validação de inputs
│           └── helpers.py      # Funções auxiliares
├── tests/
│   ├── unit/                   # Testes unitários por módulo
│   └── integration/            # Testes de pipeline completo
├── docs/                       # Documentação
├── notebooks/                  # Notebooks de exemplos e experimentos
├── scripts/                    # Scripts utilitários
├── pyproject.toml
└── README.md
```

---

## 🛠️ Desenvolvimento

```bash
# Clonar o repositório
git clone https://github.com/your-user/autoeda.git
cd autoeda

# Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Instalar dependências de desenvolvimento
pip install -e ".[dev]"

# Rodar testes
pytest
```

---

## 📋 Requisitos

- Python >= 3.9
- pandas, numpy, scipy, matplotlib, seaborn, scikit-learn

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para mais detalhes.

---

> Desenvolvido como Trabalho de Conclusão de Curso (TCC).
