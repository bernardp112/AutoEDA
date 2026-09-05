"""
Análise da relação entre as variáveis preditoras e a variável alvo (target).

Escopo restrito a classificação binária (o target sempre tem
exatamente 2 classes, garantido por utils.validate_target antes do
pipeline chegar aqui). A técnica de associação usada depende do tipo
do preditor:

- numérico              -> Point-Biserial (força/direção + p-valor) e
                           Mutual Information (dependência geral,
                           inclusive não linear/não monotônica).
- categórico nominal    -> Qui-quadrado (significância) e V de Cramér
                           (força da associação).
- categórico ordinal    -> Spearman (força/direção + p-valor). Uma
                           coluna "categorical" é tratada como ordinal
                           quando seu dtype subjacente é numérico
                           (ex.: nota de 1 a 5) — nesse caso a ordem
                           dos valores é significativa por construção.
                           Categóricas de texto (dtype objeto) são
                           tratadas como nominais.

Colunas "id", "text" e "datetime" são excluídas da análise de
associação com o target (ver _EXCLUDED_PREDICTOR_TYPES) — não fazem
sentido nessas técnicas sem processamento prévio (encoding, feature
engineering de data), que fica fora do escopo do AutoEDA.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

from autoeda.config import AutoEDAConfig
from autoeda.exceptions import AnalysisError
from autoeda.utils import infer_column_types

# Tipos lógicos (ver utils.infer_column_types) que não entram na
# análise de associação com o target.
_EXCLUDED_PREDICTOR_TYPES = {"id", "text", "datetime"}

_EXCLUSION_REASONS = {
    "id": "Coluna identificada como possível identificador (id); não carrega sinal preditivo.",
    "text": "Cardinalidade muito alta para um teste de associação categórica confiável.",
    "datetime": "Coluna de data/hora precisa de engenharia de features antes de testar associação.",
}

# Random state fixo para reprodutibilidade do cálculo de Mutual
# Information, que usa uma estimativa baseada em k-vizinhos-mais-próximos
# (não determinística por padrão).
_MUTUAL_INFO_RANDOM_STATE = 42


def get_positive_class(series: pd.Series) -> tuple[Any, Any]:
    """Define qual das 2 classes do target é tratada como "positiva"
    (codificada como 1) e qual é "negativa" (codificada como 0).

    Critério: as classes são ordenadas (sorted) e a segunda (índice 1)
    é a positiva. Para targets já codificados como 0/1, isso coincide
    com a convenção usual (0=negativo, 1=positivo). Para targets como
    "sim"/"nao", a ordem alfabética coloca "sim" como positiva.

    Este critério é uma convenção arbitrária, mas precisa ser
    determinística — a codificação exata não muda a força das
    métricas de associação (apenas o sinal do Point-Biserial/Spearman),
    mas precisa ser consistente para o relatório fazer sentido.
    """
    classes = sorted(series.dropna().unique().tolist(), key=str)
    return classes[0], classes[1]


def encode_binary_target(series: pd.Series, positive_class: Any) -> pd.Series:
    """Codifica o target binário como uma Series numérica 0/1,
    preservando o índice original (necessário para alinhar com as
    colunas preditoras antes de descartar linhas nulas em par).
    """
    return (series == positive_class).astype(float).where(series.notna())


def analyze_target_distribution(series: pd.Series, positive_class: Any, negative_class: Any) -> dict[str, Any]:
    """Resume a distribuição das 2 classes do target.

    Expõe explicitamente classe majoritária/minoritária (contagem e
    percentual) e o imbalance ratio (majoritária / minoritária) — a
    razão pedida pelo orientador para orientar a recomendação de
    balanceamento.
    """
    counts = series.dropna().value_counts()
    total = int(counts.sum())

    majority_class = counts.index[0]
    minority_class = counts.index[-1]
    majority_count = int(counts.iloc[0])
    minority_count = int(counts.iloc[-1])

    return {
        "n_classes": 2,
        "positive_class": positive_class,
        "negative_class": negative_class,
        "class_distribution": [
            {"value": str(value), "count": int(count), "pct": float(count / total)}
            for value, count in counts.items()
        ],
        "majority_class": str(majority_class),
        "majority_count": majority_count,
        "majority_pct": float(majority_count / total),
        "minority_class": str(minority_class),
        "minority_count": minority_count,
        "minority_pct": float(minority_count / total),
        "imbalance_ratio": float(majority_count / minority_count) if minority_count > 0 else None,
    }


def compute_point_biserial(numeric_values: pd.Series, binary_target: pd.Series) -> dict[str, Any]:
    """Calcula a correlação Point-Biserial entre uma variável numérica
    e o target binário (0/1), com p-valor.

    Point-Biserial é matematicamente equivalente à correlação de
    Pearson entre a variável contínua e a variável 0/1 — mede
    associação linear e direção (valores mais altos da variável
    tendem à classe positiva ou negativa).

    Retorna None nos campos de valor se houver menos de 3 pares
    válidos ou se a variável numérica não tiver variância.
    """
    paired = pd.DataFrame({"x": numeric_values, "y": binary_target}).dropna()

    if paired.shape[0] < 3 or paired["x"].nunique() < 2:
        return {"correlation": None, "p_value": None, "n": int(paired.shape[0])}

    correlation, p_value = stats.pointbiserialr(paired["y"], paired["x"])
    return {"correlation": float(correlation), "p_value": float(p_value), "n": int(paired.shape[0])}


def compute_mutual_information(numeric_values: pd.Series, binary_target: pd.Series) -> dict[str, Any]:
    """Calcula a Mutual Information entre uma variável numérica e o
    target binário.

    Diferente do Point-Biserial (só captura relação linear), Mutual
    Information captura qualquer forma de dependência estatística —
    útil para apontar relações não lineares/não monotônicas que a
    correlação deixaria passar. Não tem p-valor associado de forma
    simples (exigiria teste de permutação, fora do escopo aqui); o
    valor é sempre >= 0, sem limite superior fixo, então não é
    diretamente comparável em escala com Point-Biserial/Spearman/
    Cramér's V.
    """
    paired = pd.DataFrame({"x": numeric_values, "y": binary_target}).dropna()

    if paired.shape[0] < 3 or paired["x"].nunique() < 2:
        return {"value": None, "n": int(paired.shape[0])}

    mi = mutual_info_classif(
        paired[["x"]],
        paired["y"],
        discrete_features=False,
        random_state=_MUTUAL_INFO_RANDOM_STATE,
    )
    return {"value": float(mi[0]), "n": int(paired.shape[0])}


def compute_chi_square_and_cramers_v(categorical_values: pd.Series, target: pd.Series) -> dict[str, Any]:
    """Calcula o teste Qui-quadrado de independência e o V de Cramér
    (com correção de viés de Bergsma) entre uma variável categórica
    nominal e o target binário, a partir da mesma tabela de
    contingência.

    Qui-quadrado testa significância (a associação observada é
    improvável sob a hipótese de independência?); V de Cramér mede a
    força da associação (0 a 1). Os dois se complementam: uma
    associação pode ser estatisticamente significativa mas fraca em
    magnitude, especialmente em datasets grandes.

    Retorna None nos campos de valor se a tabela de contingência for
    degenerada (menos de 2 categorias válidas na variável preditora).
    """
    paired = pd.DataFrame({"x": categorical_values, "y": target}).dropna()

    if paired.empty:
        return {"chi2": None, "p_value": None, "cramers_v": None, "n": 0}

    contingency = pd.crosstab(paired["x"], paired["y"])
    n_rows, n_cols = contingency.shape

    if n_rows < 2 or n_cols < 2:
        return {"chi2": None, "p_value": None, "cramers_v": None, "n": int(paired.shape[0])}

    chi2, p_value, _dof, _expected = stats.chi2_contingency(contingency)

    n = contingency.values.sum()
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - ((n_rows - 1) * (n_cols - 1)) / (n - 1))
    r_corrected = n_rows - ((n_rows - 1) ** 2) / (n - 1)
    k_corrected = n_cols - ((n_cols - 1) ** 2) / (n - 1)
    denominator = min(k_corrected - 1, r_corrected - 1)

    cramers_v = float(np.sqrt(phi2_corrected / denominator)) if denominator > 0 else None

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "cramers_v": cramers_v,
        "n": int(paired.shape[0]),
    }


def compute_spearman(ordinal_values: pd.Series, binary_target: pd.Series) -> dict[str, Any]:
    """Calcula a correlação de Spearman entre uma variável ordinal
    (categórica com ordem natural, aqui aproximada por dtype numérico
    de baixa cardinalidade — ex.: nota de 1 a 5) e o target binário.

    Spearman usa os ranks dos valores, então captura associação
    monotônica (não necessariamente linear) — mais apropriado que
    Pearson/Point-Biserial quando a escala ordinal não é
    necessariamente uniforme (a diferença entre nota 1 e 2 pode não
    "valer o mesmo" que entre nota 4 e 5).
    """
    paired = pd.DataFrame({"x": ordinal_values, "y": binary_target}).dropna()

    if paired.shape[0] < 3 or paired["x"].nunique() < 2:
        return {"correlation": None, "p_value": None, "n": int(paired.shape[0])}

    correlation, p_value = stats.spearmanr(paired["x"], paired["y"])
    return {"correlation": float(correlation), "p_value": float(p_value), "n": int(paired.shape[0])}


def _is_ordinal_like(series: pd.Series) -> bool:
    """Decide se uma coluna do tipo lógico "categorical" deve ser
    tratada como ordinal (Spearman) ou nominal (Qui-quadrado/Cramér's
    V), com base no dtype subjacente: dtype numérico indica uma escala
    com ordem natural (ex.: nota, faixa etária codificada); dtype
    objeto/string é tratado como categoria sem ordem.
    """
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _get_association_value(predictor_result: dict[str, Any]) -> float | None:
    """Extrai o valor "principal" de força de associação de um
    resultado de preditor, para fins de ranking/threshold — o campo
    varia conforme a técnica usada (correlation, cramers_v).
    """
    metrics = predictor_result.get("metrics", {})
    for key in ("correlation", "cramers_v"):
        if key in metrics and metrics[key] is not None:
            return metrics[key]
    return None


def analyze_predictor_vs_target(
    df: pd.DataFrame,
    predictor: str,
    predictor_type: str,
    binary_target: pd.Series,
    config: AutoEDAConfig,
) -> dict[str, Any]:
    """Analisa a relação entre uma única variável preditora e o
    target binário, escolhendo a técnica conforme o tipo do preditor
    (ver docstring do módulo).
    """
    if predictor_type == "numeric":
        point_biserial = compute_point_biserial(df[predictor], binary_target)
        mutual_info = compute_mutual_information(df[predictor], binary_target)
        return {
            "predictor": predictor,
            "predictor_type": "numeric",
            "relationship": "point_biserial_mutual_info",
            "metrics": {
                "correlation": point_biserial["correlation"],
                "p_value": point_biserial["p_value"],
                "mutual_information": mutual_info["value"],
                "n": point_biserial["n"],
            },
        }

    if predictor_type == "categorical" and _is_ordinal_like(df[predictor]):
        spearman = compute_spearman(df[predictor], binary_target)
        return {
            "predictor": predictor,
            "predictor_type": "categorical_ordinal",
            "relationship": "spearman",
            "metrics": spearman,
        }

    # categórico nominal (ou booleano, tratado como categórico de 2 níveis)
    chi_square = compute_chi_square_and_cramers_v(df[predictor], binary_target)
    return {
        "predictor": predictor,
        "predictor_type": "categorical_nominal" if predictor_type != "boolean" else "boolean",
        "relationship": "chi_square_cramers_v",
        "metrics": chi_square,
    }


def analyze_target(df: pd.DataFrame, target: str, config: AutoEDAConfig) -> dict[str, Any]:
    """Executa a análise completa da variável alvo binária em relação
    a todas as demais colunas do dataset.

    Pressupõe que `target` já foi validado por utils.validate_target
    (garantindo exatamente 2 classes); levanta AnalysisError como
    salvaguarda caso essa invariante seja violada.

    Retorna um dict no formato:
    {
        "target": ..., "distribution": {... analyze_target_distribution ...},
        "predictors": [{... analyze_predictor_vs_target ...}, ...],
        "excluded_predictors": [{"predictor": ..., "type": ..., "reason": ...}, ...],
        "strong_predictors": [...],
        "possible_leakage": [
            {"predictor": ..., "association": float, "metric": str}, ...
        ],
        "significant_predictors": [<preditores com p_value < config.significance_alpha>],
        "multiple_comparisons_warning": str | None,
    }
    """
    if target not in df.columns:
        raise AnalysisError(f"A coluna alvo '{target}' não existe no DataFrame.")

    target_series = df[target]
    n_classes = target_series.nunique(dropna=True)
    if n_classes != 2:
        raise AnalysisError(
            f"analyze_target espera um target binário (2 classes); "
            f"'{target}' possui {n_classes}. Valide com utils.validate_target antes de chamar esta função."
        )

    negative_class, positive_class = get_positive_class(target_series)
    binary_target = encode_binary_target(target_series, positive_class)
    distribution = analyze_target_distribution(target_series, positive_class, negative_class)

    column_types = infer_column_types(
        df, config.categorical_max_cardinality, config.id_cardinality_ratio_threshold
    )

    predictor_results: list[dict[str, Any]] = []
    excluded_predictors: list[dict[str, Any]] = []

    for column, col_type in column_types.items():
        if column == target:
            continue

        if col_type in _EXCLUDED_PREDICTOR_TYPES:
            excluded_predictors.append(
                {"predictor": column, "type": col_type, "reason": _EXCLUSION_REASONS[col_type]}
            )
            continue

        result = analyze_predictor_vs_target(df, column, col_type, binary_target, config)
        predictor_results.append(result)

    predictor_results.sort(
        key=lambda r: abs(_get_association_value(r)) if _get_association_value(r) is not None else -1,
        reverse=True,
    )

    strong_predictors = [
        r["predictor"]
        for r in predictor_results
        if (value := _get_association_value(r)) is not None
        and abs(value) >= config.strong_association_threshold
    ]

    possible_leakage = [
        {
            "predictor": r["predictor"],
            "association": _get_association_value(r),
            "metric": "correlation" if "correlation" in r["metrics"] else "cramers_v",
        }
        for r in predictor_results
        if (value := _get_association_value(r)) is not None
        and abs(value) >= config.leakage_association_threshold
    ]

    significant_predictors = [
        r["predictor"]
        for r in predictor_results
        if r["metrics"].get("p_value") is not None
        and r["metrics"]["p_value"] < config.significance_alpha
    ]

    n_tested = len(predictor_results)
    multiple_comparisons_warning = None
    if n_tested > config.multiple_comparisons_warning_threshold:
        corrected_alpha = config.significance_alpha / n_tested
        multiple_comparisons_warning = (
            f"{n_tested} preditores foram testados contra o target simultaneamente; "
            "com tantos testes, é esperado que algumas associações pareçam "
            "significativas por acaso. Considere um nível de significância "
            f"corrigido (ex.: Bonferroni, alfa ≈ {corrected_alpha:.4f}) ao "
            "interpretar os p-valores individualmente."
        )

    return {
        "target": target,
        "distribution": distribution,
        "predictors": predictor_results,
        "excluded_predictors": excluded_predictors,
        "strong_predictors": strong_predictors,
        "possible_leakage": possible_leakage,
        "significant_predictors": significant_predictors,
        "multiple_comparisons_warning": multiple_comparisons_warning,
    }
