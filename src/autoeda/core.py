"""
Ponto de entrada público do AutoEDA.

A função `autoeda()` é a API principal da biblioteca: recebe um
dataset e a coluna alvo, valida a entrada, roda o pipeline completo
de análise (estatísticas descritivas, valores ausentes, outliers,
correlação/multicolinearidade, relação com o target), gera as
recomendações de pré-processamento, os gráficos e o relatório final
em Markdown — e devolve tudo isso em um único objeto de resultado.

Uso típico:

    from autoeda import autoeda

    result = autoeda(df, target="inadimplente", lang="pt-br")
    print(result.report_path)            # relatório .md
    print(result.recommendations_json_path)  # recomendações .json
    for rec in result.recommendations["recommendations"]:
        print(rec["severity"], rec["recommendation"])
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from autoeda.analysis.correlation import analyze_correlation
from autoeda.analysis.descriptive import generate_descriptive_stats
from autoeda.analysis.missing_values import analyze_missing_values
from autoeda.analysis.outliers import analyze_outliers
from autoeda.analysis.target_analysis import analyze_target
from autoeda.config import AutoEDAConfig
from autoeda.recommendations import generate_recommendations
from autoeda.report.charts import generate_all_charts
from autoeda.report.builder import export_markdown_report
from autoeda.report.json_export import export_recommendations_json
from autoeda.utils import validate_dataframe, validate_language, validate_target


@dataclass
class AutoEDAResult:
    """Reúne todos os resultados de uma execução do AutoEDA.

    Os 5 dicts de análise (descriptive, missing_values, outliers,
    correlation, target_analysis) são os mesmos formatos documentados
    em cada módulo de analysis/ — quem quiser inspecionar um número
    específico sem passar pelo relatório pode acessar diretamente
    aqui, em vez de reprocessar nada.
    """

    descriptive: dict[str, Any]
    missing_values: dict[str, Any]
    outliers: dict[str, Any]
    correlation: dict[str, Any]
    target_analysis: dict[str, Any]
    recommendations: dict[str, Any]
    charts: dict[str, Any]
    report_path: Path
    recommendations_json_path: Path
    config: AutoEDAConfig


def autoeda(
    df: pd.DataFrame,
    target: str,
    lang: str = "pt-br",
    output_dir: str | Path = "autoeda_output",
    config: AutoEDAConfig | None = None,
) -> AutoEDAResult:
    """Executa o pipeline completo do AutoEDA sobre `df`.

    Parâmetros
    ----------
    df:
        Dataset a analisar. Precisa ter ao menos 1 linha, 1 coluna,
        sem nomes de coluna duplicados ou vazios/"Unnamed" (ver
        utils.validate_dataframe).
    target:
        Nome da coluna alvo. Obrigatória — o AutoEDA está restrito a
        classificação binária, então `target` precisa ter exatamente
        2 classes distintas (ver utils.validate_target). Levanta
        InvalidTargetError caso contrário (inclusive se `target` for
        None, inexistente, ou tiver 1 ou 3+ classes).
    lang:
        Idioma de saída dos textos gerados (relatório, recomendações,
        indícios de ausência): "pt-br" ou "en-us". Padrão "pt-br".
    output_dir:
        Diretório onde o relatório (.md), o JSON de recomendações e
        os gráficos (.png) são salvos. Criado automaticamente se não
        existir.
    config:
        Configuração avançada (thresholds de severidade, métodos de
        detecção de outlier, etc). Se None, usa AutoEDAConfig(language=lang)
        com todos os demais valores padrão. Se fornecida, `lang` é
        ignorado em favor de `config.language` — evita o caso
        ambíguo de `lang` e `config.language` divergirem.

    Retorna
    -------
    AutoEDAResult com todos os resultados de análise, as
    recomendações, os caminhos dos gráficos gerados e os caminhos do
    relatório (.md) e do JSON de recomendações no disco.

    Levanta
    -------
    InvalidDataFrameError, InvalidTargetError, UnsupportedLanguageError
    (ver autoeda.exceptions) — validação de entrada acontece antes de
    qualquer análise, então uma entrada inválida nunca chega a rodar
    parte do pipeline.
    """
    if config is None:
        validated_lang = validate_language(lang)
        config = AutoEDAConfig(language=validated_lang)
    else:
        validate_language(config.language)

    df = validate_dataframe(df)
    target = validate_target(df, target)

    descriptive_result = generate_descriptive_stats(df, config)
    missing_result = analyze_missing_values(df, target, config)
    outliers_result = analyze_outliers(df, config)
    correlation_result = analyze_correlation(df, config)
    target_result = analyze_target(df, target, config)

    recommendations_result = generate_recommendations(
        descriptive_result,
        missing_result,
        outliers_result,
        correlation_result,
        target_result,
        config,
    )

    output_path = Path(output_dir)
    charts = generate_all_charts(
        df,
        config,
        descriptive_result,
        missing_result,
        correlation_result,
        target_result,
        output_path / "charts",
    )

    recommendations_json_path = export_recommendations_json(
        recommendations_result, output_path / "recommendations.json"
    )

    report_path = export_markdown_report(
        df.shape,
        descriptive_result,
        missing_result,
        outliers_result,
        correlation_result,
        target_result,
        recommendations_result,
        charts,
        output_path / "report.md",
        config,
        recommendations_json_path,
    )

    return AutoEDAResult(
        descriptive=descriptive_result,
        missing_values=missing_result,
        outliers=outliers_result,
        correlation=correlation_result,
        target_analysis=target_result,
        recommendations=recommendations_result,
        charts=charts,
        report_path=report_path,
        recommendations_json_path=recommendations_json_path,
        config=config,
    )
