"""
AutoEDA: biblioteca de Análise Exploratória de Dados automatizada,
com foco em problemas de classificação binária.

Uso básico:

    from autoeda import autoeda

    result = autoeda(df, target="inadimplente", lang="pt-br")
    print(result.report_path)
"""

from autoeda.core import AutoEDAResult, autoeda
from autoeda.exceptions import (
    AnalysisError,
    AutoEDAError,
    InvalidDataFrameError,
    InvalidTargetError,
    UnsupportedLanguageError,
)

__all__ = [
    "autoeda",
    "AutoEDAResult",
    "AutoEDAError",
    "InvalidDataFrameError",
    "InvalidTargetError",
    "UnsupportedLanguageError",
    "AnalysisError",
]
