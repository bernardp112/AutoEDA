"""
Exceções customizadas do AutoEDA.

Centralizar os erros aqui evita que os módulos de análise levantem
exceções genéricas (ValueError, KeyError, ...) que seriam difíceis de
diferenciar de bugs internos. Todo erro esperado e "de uso" da
biblioteca deve herdar de AutoEDAError.
"""

from __future__ import annotations


class AutoEDAError(Exception):
    """Classe base para todas as exceções levantadas pelo AutoEDA."""


class InvalidDataFrameError(AutoEDAError):
    """Levantada quando o objeto passado não é um DataFrame válido/utilizável.

    Exemplos: não é um pandas.DataFrame, está vazio, ou não possui
    nenhuma coluna.
    """


class InvalidTargetError(AutoEDAError):
    """Levantada quando a variável alvo (target) é inválida.

    Exemplos: coluna não existe no DataFrame, target totalmente nulo,
    target com um único valor único (sem variância).
    """


class InvalidTemporalColumnError(AutoEDAError):
    """Levantada quando a coluna temporal indicada não pode ser usada
    como tal (não existe, não é conversível para datetime, etc.).
    """


class UnsupportedLanguageError(AutoEDAError):
    """Levantada quando o idioma de saída solicitado não é suportado.

    O AutoEDA atualmente suporta apenas 'pt-br' e 'en-us' (ver
    autoeda.config.SUPPORTED_LANGUAGES).
    """


class AnalysisError(AutoEDAError):
    """Levantada quando uma etapa específica de análise falha
    de forma não recuperável (ex.: cálculo de correlação em dataset
    sem nenhuma coluna numérica).

    Erros aqui devem ser tratados como "essa análise não pôde ser
    feita para este dataset", não como bug — o core.py decide se
    aborta o pipeline ou apenas pula essa etapa e registra no relatório.
    """
