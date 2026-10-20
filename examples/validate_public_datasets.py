"""Reproduz a validação do relatório técnico com conjuntos públicos.

Executa o AutoEDA no Breast Cancer Wisconsin (Diagnostic), que já vem
embutido no scikit-learn, e, opcionalmente, no College Major ROI (arquivo
CSV do Kaggle, que precisa ser baixado à parte). Mede o tempo e resume o
vazamento sinalizado e as recomendações geradas. Com --with-ydata, mede
também o tempo do ydata-profiling (se estiver instalado).

Uso:

    python examples/validate_public_datasets.py
    python examples/validate_public_datasets.py --college-csv college_major_roi.csv
    python examples/validate_public_datasets.py --college-csv college_major_roi.csv --with-ydata

Os tempos dependem da máquina; no relatório foram obtidos em um contêiner
de 2 núcleos (ver Seção 10.1).
"""

from __future__ import annotations

import argparse
import time
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from autoeda import autoeda

warnings.filterwarnings("ignore")


def run_autoeda(name: str, df: pd.DataFrame, target: str, output_dir: Path, repeats: int) -> None:
    times = []
    result = None
    for i in range(repeats):
        start = time.perf_counter()
        result = autoeda(df, target=target, lang="pt-br", output_dir=output_dir / f"run{i}")
        times.append(time.perf_counter() - start)

    target_info = result.target_analysis
    dist = target_info["distribution"]
    recs = result.recommendations["recommendations"]
    severities = Counter(r["severity"] for r in recs)

    print(f"\n=== {name} (alvo: {target}) — {df.shape[0]} linhas x {df.shape[1]} colunas ===")
    print(f"Tempo AutoEDA ({repeats} execuções): " + ", ".join(f"{t:.2f}s" for t in times)
          + f"  | média {sum(times) / len(times):.2f}s")
    print(f"Imbalance Ratio: {dist['imbalance_ratio']:.2f}  (classe majoritária {dist['majority_pct']:.1%})")
    leakage = [f"{i['predictor']} ({i['metric']}={i['association']:.3f})" for i in target_info["possible_leakage"]]
    print("Possível vazamento:", ", ".join(leakage) if leakage else "nenhum")
    print(f"Recomendações: {len(recs)}  (alta {severities['high']}, média {severities['medium']}, baixa {severities['low']})")


def time_ydata(name: str, df: pd.DataFrame, output_dir: Path) -> None:
    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        print("ydata-profiling não está instalado (pip install ydata-profiling); pulando a comparação.")
        return
    for mode in ("minimal", "completo"):
        start = time.perf_counter()
        ProfileReport(df, minimal=(mode == "minimal"), progress_bar=False).to_file(
            output_dir / f"ydata_{name}_{mode}.html"
        )
        print(f"Tempo ydata-profiling ({mode}) em {name}: {time.perf_counter() - start:.2f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validação com conjuntos públicos.")
    parser.add_argument("--college-csv", type=Path, help="caminho do college_major_roi.csv (Kaggle)")
    parser.add_argument("--output-dir", type=Path, default=Path("validation_output"))
    parser.add_argument("--repeats", type=int, default=3, help="execuções do AutoEDA para a média de tempo")
    parser.add_argument("--with-ydata", action="store_true", help="medir também o ydata-profiling")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.datasets import load_breast_cancer

    breast = load_breast_cancer(as_frame=True).frame.rename(columns={"target": "diagnosis"})
    run_autoeda("Breast Cancer Wisconsin", breast, "diagnosis", args.output_dir / "breast_cancer", args.repeats)
    if args.with_ydata:
        time_ydata("breast_cancer", breast, args.output_dir)

    if args.college_csv:
        college = pd.read_csv(args.college_csv)
        for target in ("positive_roi", "high_roi"):
            run_autoeda("College Major ROI", college, target, args.output_dir / f"college_{target}", args.repeats)
        if args.with_ydata:
            time_ydata("college_major_roi", college, args.output_dir)
    else:
        print("\n(College Major ROI não executado: informe --college-csv com o arquivo do Kaggle.)")


if __name__ == "__main__":
    main()
