# Validação

Este documento explica como reproduzir a validação descrita na Seção 10 do relatório técnico e quais são os limites dela.

## Como reproduzir

Com o pacote instalado (`pip install -e ".[dev]"`):

```bash
# Breast Cancer Wisconsin (embutido no scikit-learn)
python examples/validate_public_datasets.py

# + College Major ROI (baixar college_major_roi.csv do Kaggle: sergionefedov/college-major-roi)
python examples/validate_public_datasets.py --college-csv college_major_roi.csv

# + tempo do ydata-profiling (pip install ydata-profiling)
python examples/validate_public_datasets.py --college-csv college_major_roi.csv --with-ydata
```

O script imprime, para cada conjunto e alvo, o tempo do AutoEDA (média de `--repeats` execuções, padrão 3), o Imbalance Ratio, o vazamento sinalizado e o número de recomendações por severidade.

## Resultados de referência (relatório, Seção 10)

| Conjunto / alvo | Vazamento sinalizado | Recomendações (alta / média / baixa) |
|---|---|---|
| Breast Cancer / diagnosis | nenhum | 132 (1 / 82 / 49) |
| College Major ROI / positive_roi | net_roi_usd, roi_pct | 34 (4 / 12 / 18) |
| College Major ROI / high_roi | net_roi_usd, added_earnings_10yr_usd, earnings_10yr_usd | 35 (5 / 12 / 18) |

Tempos (contêiner Linux de 2 núcleos): AutoEDA ≈ 3,6 a 4,4 s no College Major ROI e ≈ 5,6 s no Breast Cancer; o tempo varia entre execuções e entre sessões. Os números de recomendações e de vazamento são determinísticos.

## Limites da validação

- **O College Major ROI é sintético** (descrição do próprio autor no Kaggle): os resultados mostram coerência das recomendações com uma estrutura conhecida, não generalização a dados reais.
- **A validação em dados reais é pequena**: o Breast Cancer tem 569 linhas e só variáveis numéricas, sem valores ausentes nem categóricas.
- **A qualidade das recomendações foi avaliada qualitativamente.** Não foi medido se aplicá-las melhora o desempenho de modelos.
- **O comparativo com o ydata-profiling é restrito a tempo e tipo de saída**, em um único ambiente e com uma execução por modo do ydata-profiling. As ferramentas têm propósitos diferentes (descrição exploratória × recomendação orientada ao alvo), o que limita a comparação direta.
- Os tempos não constituem um benchmark estatístico.
