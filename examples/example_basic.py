"""Demonstração rápida do AutoEDA — não precisa de nenhum arquivo externo.

Gera um dataset sintético de classificação binária com problemas
conhecidos (valores ausentes, outliers, coluna constante, identificador,
variáveis redundantes e uma variável que "vaza" o alvo), roda o AutoEDA e
mostra as recomendações geradas.

Uso (com o pacote instalado: `pip install -e .`):

    python examples/example_basic.py
    python examples/example_basic.py --lang en-us --output-dir demo_output
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from autoeda import autoeda


def build_demo_dataset(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Dataset sintético com problemas de qualidade injetados de propósito."""
    rng = np.random.default_rng(seed)

    idade = rng.integers(18, 80, n_rows)
    renda = rng.lognormal(mean=8.0, sigma=0.7, size=n_rows).round(2)
    divida = (renda * rng.uniform(0.1, 0.6, n_rows)).round(2)

    # alvo binário (inadimplente) dependente de renda e dívida, com ruído
    score = -0.00015 * renda + 0.0004 * divida + rng.normal(0, 1, n_rows)
    inadimplente = (score > np.quantile(score, 0.80)).astype(int)

    df = pd.DataFrame(
        {
            "id_cliente": np.arange(1, n_rows + 1),               # possível identificador
            "idade": idade,
            "renda": renda,
            "divida": divida,
            "renda_mensal": (renda / 12).round(2),                 # redundante com renda
            "pais": "BR",                                          # coluna constante
            "estado_civil": rng.choice(["solteiro", "casado", "divorciado"], n_rows),
            "valor_pos_evento": inadimplente * 100 + rng.normal(0, 1, n_rows),  # vaza o alvo
            "inadimplente": inadimplente,
        }
    )

    # valores ausentes em renda (mais frequentes entre os inadimplentes) e outliers em divida
    ausente = rng.random(n_rows) < np.where(inadimplente == 1, 0.25, 0.05)
    df.loc[ausente, "renda"] = np.nan
    df.loc[rng.choice(n_rows, 8, replace=False), "divida"] *= 40
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstração do AutoEDA com dados sintéticos.")
    parser.add_argument("--lang", default="pt-br", choices=["pt-br", "en-us"])
    parser.add_argument("--output-dir", default="autoeda_demo_output")
    args = parser.parse_args()

    df = build_demo_dataset()
    print(f"Dataset de demonstração: {df.shape[0]} linhas x {df.shape[1]} colunas")

    result = autoeda(df, target="inadimplente", lang=args.lang, output_dir=args.output_dir)

    recs = result.recommendations["recommendations"]
    print(f"\n{len(recs)} recomendações geradas. As de severidade alta e média:\n")
    for rec in recs:
        if rec["severity"] in ("high", "medium"):
            print(f"[{rec['severity']}] {rec['feature'] or '(dataset)'}: {rec['recommendation']}")

    print(f"\nRelatório em Markdown: {result.report_path}")
    print(f"Recomendações em JSON: {result.recommendations_json_path}")


if __name__ == "__main__":
    main()
