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

# Razão (valores únicos não nulos / número de linhas) a partir da qual
# uma coluna é sinalizada como possível identificador. Não exigimos
# 100% exato (== 1.0) porque um pequeno número de duplicatas legítimas
# (reenvio de formulário, erro de digitação pontual) não deveria
# impedir a detecção de uma coluna que é, na prática, um ID.
ID_CARDINALITY_RATIO_THRESHOLD = 0.95

# Força mínima (métrica bounded em [0,1] ou [-1,1]: Point-Biserial,
# Spearman, V de Cramér) a partir da qual a associação de um preditor
# com o target é tratada como candidata a "vazamento direto" — a
# variável provavelmente é uma proxy do próprio target, não um
# preditor legítimo. Mutual Information não entra nesse alerta por
# não ser uma métrica limitada na mesma escala.
LEAKAGE_ASSOCIATION_THRESHOLD = 0.95

# Nível de significância (alfa) usado para marcar um p-valor como
# estatisticamente significativo nos testes de associação com o
# target (Point-Biserial, Qui-quadrado, Spearman).
SIGNIFICANCE_ALPHA = 0.05

# Número de preditores testados contra o target a partir do qual o
# relatório inclui um aviso sobre múltiplas comparações (com muitos
# testes simultâneos, é esperado que algumas associações pareçam
# "significativas" por acaso).
MULTIPLE_COMPARISONS_WARNING_THRESHOLD = 10

# Força mínima (mesma escala de LEAKAGE_ASSOCIATION_THRESHOLD) para
# uma associação com o target ser reportada como "preditor forte"
# (fora do contexto de vazamento).
STRONG_ASSOCIATION_THRESHOLD = 0.30

# Diferença mínima (em pontos percentuais, escala 0-1) entre a taxa de
# ausência de uma coluna nas duas classes do target para reportar essa
# diferença como indício de MAR ligado ao problema de classificação
# (ex.: "renda" falta muito mais entre quem não pagou o empréstimo).
MISSING_TARGET_RATE_DIFF_THRESHOLD = 0.10

# Razão (frequência do valor mais comum / frequência do 2º mais comum)
# a partir da qual uma coluna não-constante é sinalizada como "quase
# constante" (near-zero variance). Segue a heurística clássica do
# pacote caret (R): um valor domina tão fortemente os demais que a
# coluna carrega pouca informação, mesmo sem ser tecnicamente
# constante. 19 equivale a uma divisão 95/5 entre o valor dominante e
# o segundo mais frequente.
NEAR_ZERO_VARIANCE_FREQ_RATIO_THRESHOLD = 19.0

# Percentual máximo de valores únicos (sobre o total de linhas) para
# uma coluna ser candidata a "quase constante". Combinado com o
# freq_ratio acima: uma coluna só é sinalizada se AMBOS os critérios
# indicarem baixa variabilidade — isso evita marcar colunas de alta
# cardinalidade que só por acaso têm uma categoria dominante.
NEAR_ZERO_VARIANCE_UNIQUE_PCT_THRESHOLD = 0.10  # 10%

# VIF (Variance Inflation Factor) a partir do qual uma variável é
# sinalizada como candidata a multicolinearidade relevante. 5.0 é o
# corte mais conservador entre as convenções usuais da literatura
# (algumas fontes usam 10.0); preferimos o mais rigoroso como padrão.
VIF_THRESHOLD = 5.0

# Razão (desvio padrão da variável de maior escala / desvio padrão da
# de menor escala) a partir da qual o dataset é sinalizado como tendo
# variáveis numéricas em escalas muito diferentes, candidatas a
# padronização antes de modelos sensíveis a escala (distância,
# regularização).
SCALE_DISPARITY_RATIO_THRESHOLD = 10.0


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
    id_cardinality_ratio_threshold: float = ID_CARDINALITY_RATIO_THRESHOLD
    leakage_association_threshold: float = LEAKAGE_ASSOCIATION_THRESHOLD
    significance_alpha: float = SIGNIFICANCE_ALPHA
    multiple_comparisons_warning_threshold: int = MULTIPLE_COMPARISONS_WARNING_THRESHOLD
    strong_association_threshold: float = STRONG_ASSOCIATION_THRESHOLD
    missing_target_rate_diff_threshold: float = MISSING_TARGET_RATE_DIFF_THRESHOLD
    near_zero_variance_freq_ratio_threshold: float = NEAR_ZERO_VARIANCE_FREQ_RATIO_THRESHOLD
    near_zero_variance_unique_pct_threshold: float = NEAR_ZERO_VARIANCE_UNIQUE_PCT_THRESHOLD
    vif_threshold: float = VIF_THRESHOLD
    scale_disparity_ratio_threshold: float = SCALE_DISPARITY_RATIO_THRESHOLD

    # Colunas a ignorar em todas as análises (ex.: IDs identificados
    # automaticamente ou informados pelo usuário).
    ignore_columns: list[str] = field(default_factory=list)
