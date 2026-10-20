# Changelog

## 1.0.1 (revisão de outubro de 2026)

- **Vazamento do alvo**: além da associação quase perfeita (correlação/V de Cramér ≥ 0,95), passa a sinalizar variáveis com AUC isolada ≥ 0,99 (mínimo de 30 observações pareadas). Novos parâmetros: `leakage_auc_threshold`, `leakage_auc_min_observations`.
- **Quase constante**: acrescentado o critério de que o valor mais frequente concentre ≥ 90% das observações (`near_zero_variance_top_value_pct_threshold`), eliminando falsos positivos em variáveis contínuas com muitos zeros.
- **V de Cramér**: passa a usar o qui-quadrado de Pearson sem correção de Yates (o p-valor continua o do teste padrão do scipy).
- **Linguagem de sugestão**: recomendações que estavam em forma imperativa ("Remover…", "Padronizar…", "Investigar…") passaram a "Avaliar/Considerar/Recomenda-se…", em português e inglês; novo teste garante o padrão.
- Novos exemplos: `examples/example_basic.py` (demonstração com dados sintéticos) e `examples/validate_public_datasets.py` (reproduz a validação).
- Documentação: `docs/statistical_notes.md`, `docs/validation.md`; README e `docs/architecture.md` atualizados.
- Suíte de testes: 121 testes.

## 0.1.0

- Primeira versão: análises descritiva, de valores ausentes, outliers, correlação/VIF e relação com o alvo; motor de recomendações; relatório em Markdown e JSON em pt-br e en-us.
