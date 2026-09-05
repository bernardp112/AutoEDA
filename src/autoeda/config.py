"""
Configurações e valores padrão do AutoEDA.

Reunir esses valores aqui (em vez de espalhar "números mágicos" pelos
módulos de análise) permite que o usuário avançado ajuste o
comportamento da biblioteca sem precisar editar o código-fonte, e
facilita testes com diferentes thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# --- Idiomas suportados -----------------------------------------------

SUPPORTED_LANGUAGES = ("pt-br", "en-us")
DEFAULT_LANGUAGE = "pt-br"


# --- Thresholds de análise ---------------------------------------------

# Percentual de valores ausentes acima do qual uma coluna é sinalizada
# como candidata a remoção (em vez de apenas imputação).
MISSING_VALUES_DROP_THRESHOLD = 0.60  # 60%

# Percentual de valores ausentes acima do qual já vale recomendar
# alguma forma de tratamento (imputação, flag de ausência, etc.).
MISSING_VALUES_WARNING_THRESHOLD = 0.05  # 5%

# Método padrão de detecção de outliers: "iqr" (intervalo interquartil)
# ou "zscore".
OUTLIER_METHOD = "iqr"
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3.0

# Correlação (Pearson/Spearman) acima da qual duas variáveis são
# consideradas fortemente correlacionadas (candidatas a redundância).
CORRELATION_HIGH_THRESHOLD = 0.80

# Cardinalidade máxima para uma coluna numérica discreta ainda ser
# tratada como categórica na análise (ex.: notas de 1 a 5).
CATEGORICAL_MAX_CARDINALITY = 20


@dataclass
class AutoEDAConfig:
    """Agrupa as configurações de uma execução do AutoEDA.

    Instanciada internamente pelo core.py a partir dos parâmetros da
    função principal + estes defaults, e repassada aos módulos de
    analysis/ e report/ para que todos usem os mesmos thresholds.
    """

    language: str = DEFAULT_LANGUAGE
    outlier_method: str = OUTLIER_METHOD
    outlier_iqr_multiplier: float = OUTLIER_IQR_MULTIPLIER
    outlier_zscore_threshold: float = OUTLIER_ZSCORE_THRESHOLD
    missing_drop_threshold: float = MISSING_VALUES_DROP_THRESHOLD
    missing_warning_threshold: float = MISSING_VALUES_WARNING_THRESHOLD
    correlation_high_threshold: float = CORRELATION_HIGH_THRESHOLD
    categorical_max_cardinality: int = CATEGORICAL_MAX_CARDINALITY

    # Colunas a ignorar em todas as análises (ex.: IDs identificados
    # automaticamente ou informados pelo usuário).
    ignore_columns: list[str] = field(default_factory=list)
