"""
Exemplo de uso do AutoEDA com um dataset real.

Por padrão, roda sobre o dataset "Breast Cancer Wisconsin" (via
scikit-learn — já vem com o ambiente, não precisa baixar nada), um
clássico de classificação binária (diagnóstico maligno/benigno) com
30 variáveis numéricas reais derivadas de exames de imagem.

Também aceita seu próprio arquivo, em vários formatos (detectado pela
extensão): .csv, .tsv, .xlsx/.xls (Excel), .json, .parquet, .feather.

    python examples/example_real_dataset.py                          # dataset de exemplo (breast cancer)
    python examples/example_real_dataset.py --input meus_dados.csv --target minha_coluna_alvo
    python examples/example_real_dataset.py --input meus_dados.xlsx --target minha_coluna_alvo
    python examples/example_real_dataset.py --input meus_dados.parquet --target minha_coluna_alvo --lang en-us

Ao final, imprime um resumo no terminal e mostra onde o relatório
completo (.md), os gráficos (.png) e o JSON de recomendações foram
salvos.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from autoeda import autoeda
from autoeda.exceptions import AutoEDAError

# Extensão de arquivo -> função de leitura do pandas correspondente.
# .xlsx/.xls exigem o pacote extra "openpyxl" (Excel) instalado;
# .parquet/.feather exigem "pyarrow" — nenhum dos dois é dependência
# obrigatória do AutoEDA em si (mantém a lib enxuta), só é necessário
# se você quiser carregar esses formatos especificamente.
_LOADERS = {
    ".csv": pd.read_csv,
    ".tsv": lambda path: pd.read_csv(path, sep="\t"),
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
}


def load_dataset_from_path(path: str) -> pd.DataFrame:
    """Carrega um dataset a partir do caminho informado, escolhendo o
    leitor do pandas pela extensão do arquivo.

    Levanta ValueError com uma mensagem clara se a extensão não for
    suportada, e uma mensagem específica se faltar um pacote opcional
    (ex.: tentar ler .xlsx sem o openpyxl instalado) em vez de deixar
    vazar o ImportError cru do pandas.
    """
    extension = Path(path).suffix.lower()
    loader = _LOADERS.get(extension)

    if loader is None:
        supported = ", ".join(sorted(_LOADERS))
        raise ValueError(
            f"Formato '{extension or '(sem extensão)'}' não suportado. "
            f"Formatos aceitos: {supported}."
        )

    try:
        return loader(path)
    except ImportError as exc:
        extra_package = {".xlsx": "openpyxl", ".xls": "openpyxl", ".parquet": "pyarrow", ".feather": "pyarrow"}
        package = extra_package.get(extension)
        hint = f" Rode: pip install {package}" if package else ""
        raise ImportError(
            f"Faltou um pacote para ler arquivos '{extension}'.{hint}"
        ) from exc


def load_breast_cancer_dataset() -> tuple[pd.DataFrame, str]:
    """Carrega o dataset Breast Cancer Wisconsin (scikit-learn) como
    DataFrame, com a coluna alvo já traduzida para valores legíveis
    ("maligno"/"benigno") em vez de 0/1 — só para deixar o relatório
    mais claro; a lib funciona igual com 0/1.
    """
    from sklearn.datasets import load_breast_cancer

    raw = load_breast_cancer(as_frame=True)
    df = raw.frame.copy()
    # no dataset original, target 0 = maligno, 1 = benigno
    df["diagnostico"] = df["target"].map({0: "maligno", 1: "benigno"})
    df = df.drop(columns=["target"])
    return df, "diagnostico"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Roda o AutoEDA sobre um dataset real.")
    parser.add_argument(
        "--input",
        "--csv",
        dest="input_path",
        type=str,
        default=None,
        help=(
            "Caminho para o seu próprio dataset (.csv, .tsv, .xlsx, .xls, .json, "
            ".parquet ou .feather — detectado pela extensão). Se omitido, usa o "
            "dataset de exemplo (Breast Cancer Wisconsin). '--csv' é aceito como "
            "sinônimo, por compatibilidade."
        ),
    )
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Nome da coluna alvo no seu dataset (obrigatório se --input for usado).",
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="pt-br",
        choices=["pt-br", "en-us"],
        help="Idioma do relatório e das recomendações (padrão: pt-br).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="autoeda_output",
        help="Pasta onde salvar o relatório, os gráficos e o JSON de recomendações.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.input_path is not None:
        if args.target is None:
            print("Erro: --target é obrigatório quando --input é usado.", file=sys.stderr)
            sys.exit(1)
        print(f"Carregando dataset de '{args.input_path}'...")
        try:
            df = load_dataset_from_path(args.input_path)
        except (ValueError, ImportError) as exc:
            print(f"Erro ao carregar o arquivo: {exc}", file=sys.stderr)
            sys.exit(1)
        target = args.target
    else:
        print("Nenhum --input informado, usando o dataset de exemplo (Breast Cancer Wisconsin).")
        df, target = load_breast_cancer_dataset()

    print(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas. Target: '{target}'.")
    print("Rodando o AutoEDA...\n")

    try:
        result = autoeda(df, target=target, lang=args.lang, output_dir=args.output_dir)
    except AutoEDAError as exc:
        print(f"O AutoEDA não conseguiu processar este dataset: {exc}", file=sys.stderr)
        sys.exit(1)

    # --- Resumo no terminal -------------------------------------------------
    distribution = result.target_analysis["distribution"]
    print("=" * 70)
    print("RESUMO")
    print("=" * 70)
    print(
        f"Classe majoritária: {distribution['majority_class']} "
        f"({distribution['majority_pct']:.1%})"
    )
    print(
        f"Classe minoritária: {distribution['minority_class']} "
        f"({distribution['minority_pct']:.1%})"
    )
    print(f"Imbalance ratio: {distribution['imbalance_ratio']:.2f}")
    print()

    strong = result.target_analysis["strong_predictors"]
    print(f"Preditores fortes ({len(strong)}): {', '.join(strong) if strong else '(nenhum)'}")

    leakage = result.target_analysis["possible_leakage"]
    if leakage:
        print(f"⚠️  Possível vazamento de dados em: {[item['predictor'] for item in leakage]}")

    print()
    recommendations = result.recommendations["recommendations"]
    by_severity: dict[str, int] = {}
    for rec in recommendations:
        by_severity[rec["severity"]] = by_severity.get(rec["severity"], 0) + 1
    print(f"Total de recomendações: {len(recommendations)} {dict(by_severity)}")

    print()
    print("Top 5 recomendações (por severidade):")
    for rec in recommendations[:5]:
        feature = rec["feature"] or "(dataset)"
        print(f"  [{rec['severity'].upper():6}] {feature}: {rec['recommendation']}")

    # --- Onde encontrar os artefatos ----------------------------------------
    print()
    print("=" * 70)
    print("ARQUIVOS GERADOS")
    print("=" * 70)
    print(f"Relatório completo:        {result.report_path}")
    print(f"Recomendações (JSON):      {result.recommendations_json_path}")
    print(f"Gráficos:                  {result.report_path.parent / 'charts'}")


if __name__ == "__main__":
    main()
