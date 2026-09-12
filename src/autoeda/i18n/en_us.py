"""
English (en-us) text catalog for AutoEDA.

Mirrors pt_br.py key for key — i18n/__init__.py checks both catalogs
stay in sync (assert_catalogs_in_sync). See pt_br.py's module
docstring for the templating convention (Python str.format).
"""

TRANSLATIONS: dict[str, str] = {
    # --- analysis.missing_values: missingness mechanism hints -------
    "missing.evidence.correlated": "missingness correlated with that of '{other}' (r={corr:.2f})",
    "missing.evidence.target_diff": "missing rate differs across the target classes ({rates})",
    "missing.hint.mar": "MAR",
    "missing.hint.indeterminate": "undetermined",
    "missing.note.mar": (
        "Evidence of MAR (missingness related to observed variables). This is "
        "not a confirmation — the actual mechanism cannot be determined with "
        "certainty from the data alone."
    ),
    "missing.note.indeterminate": (
        "No evidence of MAR found (no correlation with the missingness of other "
        "columns, nor a relevant difference across the target classes). This "
        "does not confirm MCAR: the missingness may still depend on the "
        "unobserved value itself (MNAR), which cannot be verified from the "
        "dataset."
    ),

    # --- analysis.target_analysis -----------------------------------
    "target.multiple_comparisons_warning": (
        "{n} predictors were tested against the target simultaneously; with so "
        "many tests, some associations are expected to look significant by "
        "chance. Consider a corrected significance level (e.g., Bonferroni, "
        "alpha ≈ {alpha:.4f}) when interpreting individual p-values."
    ),

    # --- recommendations: missing values ------------------------------
    "rec.missing_high.recommendation": "Consider removing the '{column}' column from the dataset.",
    "rec.missing_high.explanation": (
        "Column with {pct} of missing values. Such a high percentage makes "
        "imputation unreliable; the column tends to add more noise than "
        "useful information."
    ),
    "rec.missing_numeric.recommendation": "Consider imputing missing values in '{column}' with the median.",
    "rec.missing_numeric.reason": "The median is robust to outliers, a safer default than the mean.",
    "rec.missing_categorical.recommendation": (
        "Consider imputing missing values in '{column}' with the mode (most frequent category)."
    ),
    "rec.missing_categorical.reason": (
        "For categorical columns, the mode preserves the original class distribution."
    ),
    "rec.missing_generic.explanation": "Column with {pct} of missing values. {reason}",
    "rec.missing_correlated.recommendation": (
        "Investigate whether the joint missingness of '{column_a}' and "
        "'{column_b}' reflects a common process (e.g., the same optional "
        "collection step) before imputing each column separately."
    ),
    "rec.missing_correlated.explanation": (
        "'{column_a}' and '{column_b}' tend to be missing together (missingness "
        "correlation {corr:.2f}). Correlated missingness suggests a systematic "
        "pattern (evidence of MAR), not random — imputing each column "
        "independently can distort the relationship between them."
    ),
    "rec.missing_target_assoc.recommendation": (
        "Consider creating a binary missing indicator feature for '{column}' "
        "in addition to imputing the value."
    ),
    "rec.missing_target_assoc.explanation": (
        "The missing rate of '{column}' differs across the target classes "
        "({rates}). Missingness that differs by class is, in itself, "
        "predictive information (evidence of MAR tied to the problem) — "
        "discarding it during imputation throws away signal the model could "
        "use."
    ),

    # --- recommendations: outliers ------------------------------------
    "rec.outliers.recommendation_low": (
        "Consider applying winsorization (capping) to the extreme values of '{column}'."
    ),
    "rec.outliers.recommendation_medium": (
        "Manually investigate the extreme values of '{column}' before treating them."
    ),
    "rec.outliers.reason_low": "low, consistent with isolated noise",
    "rec.outliers.reason_medium": (
        "high for isolated outliers; may indicate a collection error or a distinct subpopulation"
    ),
    "rec.outliers.explanation": (
        "{count} outlier(s) detected ({pct} of observations, {method} method) "
        "— {reason} percentage. Outliers should not be removed automatically, "
        "as they may represent legitimate domain information."
    ),

    # --- recommendations: descriptive ---------------------------------
    "rec.duplicate_rows.recommendation": "Remove duplicate rows before any analysis/modeling.",
    "rec.duplicate_rows.explanation": (
        "{count} duplicate row(s) ({pct} of the dataset). Duplicate rows "
        "distort descriptive statistics and can cause data leakage between "
        "train and test if not removed before the split."
    ),
    "rec.id_column.recommendation": "Consider removing '{column}' from correlation analyses and modeling.",
    "rec.id_column.explanation": (
        "Column identified as a possible identifier (unique-value ratio close "
        "to 100%). Identifiers carry no causal/predictive relationship with "
        "the target; including them can produce spurious correlations."
    ),
    "rec.skewed.recommendation_log": "Consider applying a log transformation to '{column}'.",
    "rec.skewed.recommendation_yeo": (
        "Consider applying a Yeo-Johnson transformation to '{column}' (there are values <= 0)."
    ),
    "rec.skewed.explanation": (
        "Skewness of {skewness:.2f}. Strongly skewed distributions violate the "
        "normality assumption of several models and metrics; the "
        "transformation brings the distribution closer to a more symmetric shape."
    ),
    "rec.high_cardinality.recommendation": (
        "Consider Frequency Encoding, Target Encoding, or grouping rare "
        "categories in '{column}' (or removal, if it is a possible identifier)."
    ),
    "rec.high_cardinality.explanation": (
        "High cardinality ({unique_count} categories). One-hot encoding on "
        "high-cardinality columns produces an excessive number of new sparse "
        "columns; frequency/target-based encoding or grouping rare categories "
        "into 'other' tends to generalize better."
    ),
    "rec.constant.recommendation": "Remove the '{column}' column from the dataset.",
    "rec.constant.explanation": (
        "Constant column (a single value across the whole sample). A constant "
        "column has zero variance and, by definition, cannot contribute to "
        "separating the target classes."
    ),
    "rec.near_zero_variance.recommendation": (
        "Consider removing '{column}' or treating it as low-information."
    ),
    "rec.near_zero_variance.explanation": (
        "Near-constant column: a single value accounts for {pct} of "
        "observations. Carries little information to separate the classes, "
        "even without technically zero variance, and may destabilize models "
        "sensitive to low-variance features."
    ),
    "rec.mixed_type.recommendation": (
        "Standardize the format of '{column}' (e.g., convert spelled-out "
        "values to numbers, or treat them as an 'invalid' category) before "
        "any analysis."
    ),
    "rec.mixed_type.explanation": (
        "Column mixes numeric ({numeric_pct}) and non-numeric ({non_numeric_pct}) "
        "values. Usually indicates a typing or export error; without a fix, "
        "the column is misclassified and statistical calculations become "
        "distorted."
    ),

    # --- recommendations: correlation -----------------------------------
    "rec.correlation_pair.recommendation": (
        "Consider removing one of the columns ('{column_a}' or '{column_b}') "
        "or combining them into a single feature."
    ),
    "rec.correlation_pair.explanation": (
        "'{column_a}' and '{column_b}' have a {method} correlation of {corr:.2f}. "
        "Strongly correlated columns carry redundant information; keeping both "
        "increases multicollinearity without a proportional gain in signal."
    ),
    "rec.vif.recommendation": (
        "Consider removing '{column}' or reducing the dimensionality of the "
        "redundant variable group (e.g., PCA) before a linear model."
    ),
    "rec.vif.explanation": (
        "VIF of {vif}. A high VIF indicates the variable is nearly a linear "
        "combination of other variables in the dataset — unlike pairwise "
        "correlation, VIF captures multivariate redundancy even when no single "
        "pair looks strongly correlated."
    ),
    "rec.scale_disparity.recommendation": (
        "Standardize (StandardScaler) or normalize the numeric variables "
        "before scale-sensitive models (e.g., KNN, SVM, L1/L2-regularized regression)."
    ),
    "rec.scale_disparity.explanation": (
        "'{largest_column}' (std {largest_std:.2f}) is on a much larger scale "
        "than '{smallest_column}' (std {smallest_std:.2f}), a ratio of "
        "{ratio:.0f}x. Variables on very different scales dominate the "
        "distance calculation or the regularization term purely because of "
        "magnitude, not because they carry more signal."
    ),

    # --- recommendations: target -----------------------------------------
    "rec.target_imbalance.recommendation": (
        "Consider resampling (over/undersampling or SMOTE, applied only to the "
        "training set) or class weights (class_weight) at the modeling stage."
    ),
    "rec.target_imbalance.explanation": (
        "Imbalanced classes: '{majority_class}' ({majority_pct}) vs "
        "'{minority_class}' ({minority_pct}), a ratio of {ratio:.1f}:1. Accuracy "
        "alone can be misleading in this scenario — prefer metrics such as F1, "
        "minority-class recall, or AUC-ROC."
    ),
    "rec.target_leakage.recommendation": (
        "Investigate whether '{predictor}' is a proxy for the target itself "
        "(e.g., filled in after the event the target represents) before using "
        "it as a predictor."
    ),
    "rec.target_leakage.explanation": (
        "Very strong association with the target ({metric} = {association:.2f}). "
        "A near-perfect association is more consistent with data leakage than "
        "with a legitimate predictor — including it artificially inflates the "
        "model's train/validation performance without generalizing to production."
    ),
    "rec.target_strong_predictors.recommendation": (
        "Prioritize these variables in the model's feature selection."
    ),
    "rec.target_strong_predictors.explanation": (
        "{n} variable(s) with a strong association with the target: {predictors}. "
        "Variables with a strong association (high Point-Biserial, Cramér's V, "
        "or Spearman) tend to carry more predictive signal."
    ),
    "rec.target_multiple_comparisons.recommendation": (
        "Interpret the individual p-values of the association tests with "
        "caution; prefer predictors with higher association strength (not just "
        "significance) when selecting features."
    ),

    # --- recommendations: data leakage (workflow) -------------------------
    "rec.data_leakage.recommendation": (
        "Split train and test before computing any data-preparation statistic; "
        "fit imputation, normalization, feature selection, and SMOTE only on "
        "the training set, and apply the same (already-fitted) transformations "
        "to the test set."
    ),
    "rec.data_leakage.explanation": (
        "The statistics and recommendations in this report were computed over "
        "the full dataset, for exploratory diagnostic purposes. Using means, "
        "medians, categories, or resampling parameters computed over the test "
        "set (or the whole dataset) to prepare the data before the train/test "
        "split is a common form of data leakage: the model gains indirect "
        "access to test information during training, unrealistically "
        "inflating validation metrics."
    ),

    # --- report/builder: report structure ----------------------------
    "report.title": "AutoEDA Report",
    "report.summary": (
        "Dataset: {rows} rows × {cols} columns. Target variable: '{target}' "
        "(binary classification)."
    ),
    "report.table.metric": "Metric",
    "report.table.value": "Value",
    "report.table.na": "N/A",

    "report.section1.title": "1. Dataset overview",
    "report.section1.rows": "Rows",
    "report.section1.columns": "Columns",
    "report.section1.memory": "Memory usage",
    "report.section1.duplicate_rows": "Duplicate rows",
    "report.section1.missing_cells": "Missing cells (total)",

    "report.section2.title": "2. Target variable: '{target}'",
    "report.section2.majority_class": "Majority class",
    "report.section2.minority_class": "Minority class",
    "report.section2.imbalance_ratio": "Imbalance ratio",
    "report.section2.imbalance_warning": (
        "⚠️ Imbalanced classes — see the corresponding recommendation in section 8."
    ),

    "report.section3.title": "3. Missing values",
    "report.section3.no_missing": "No missing values found in the dataset.",
    "report.section3.table.column": "Column",
    "report.section3.table.pct_missing": "% missing",
    "report.section3.table.severity": "Severity",
    "report.section3.table.mechanism_hint": "Mechanism hint",
    "report.section3.mechanism_note": (
        "Note: MCAR, MAR, and MNAR cannot be determined with certainty from the "
        "data alone. \"MAR\" above indicates evidence (missingness correlated "
        "with another column, or with the target classes); \"undetermined\" does "
        "not confirm MCAR — MNAR can never be ruled out from observed data alone."
    ),

    "report.section4.title": "4. Descriptive statistics",
    "report.section4.numeric_subtitle": "Numeric variables",
    "report.section4.categorical_subtitle": "Categorical variables",
    "report.section4.constant_subtitle": "Constant / near-constant variables",
    "report.section4.mixed_type_subtitle": "Mixed-type columns",
    "report.section4.table.column": "Column",
    "report.section4.table.mean": "Mean",
    "report.section4.table.median": "Median",
    "report.section4.table.std": "Std",
    "report.section4.table.min": "Min",
    "report.section4.table.max": "Max",
    "report.section4.table.iqr": "IQR",
    "report.section4.table.skewness": "Skewness",
    "report.section4.table.pct_outliers": "% Outliers",
    "report.section4.table.categories": "Categories",
    "report.section4.table.dominant_category": "Dominant category",
    "report.section4.table.pct_dominant": "% dominant",
    "report.section4.table.pct_missing": "% missing",
    "report.section4.table.type": "Type",
    "report.section4.table.pct_dominant_value": "% dominant value",
    "report.section4.table.pct_numeric": "% numeric",
    "report.section4.table.pct_non_numeric": "% non-numeric",
    "report.section4.constant_label": "constant",
    "report.section4.near_constant_label": "near-constant",

    "report.section5.title": "5. Outliers",
    "report.section5.no_outliers": "No relevant outliers detected in the numeric variables.",
    "report.section5.method_note": (
        "Method: {method}. Outliers are not removed automatically — they may "
        "represent legitimate domain information."
    ),
    "report.section5.table.column": "Column",
    "report.section5.table.count": "Outlier count",
    "report.section5.table.pct": "% outliers",

    "report.section6.title": "6. Correlation and multicollinearity",
    "report.section6.strong_corr_subtitle": "Strongly correlated pairs",
    "report.section6.vif_subtitle": "High VIF (Variance Inflation Factor)",
    "report.section6.scale_subtitle": "Scale disparity",
    "report.section6.scale_text": (
        "'{largest_column}' is on a scale {ratio:.0f}x larger than "
        "'{smallest_column}'. Consider standardization."
    ),
    "report.section6.table.column_a": "Column A",
    "report.section6.table.column_b": "Column B",
    "report.section6.table.method": "Method",
    "report.section6.table.correlation": "Correlation",
    "report.section6.table.column": "Column",
    "report.section6.table.vif": "VIF",

    "report.section7.title": "7. Relationship between variables and the target '{target}'",
    "report.section7.table.predictor": "Predictor",
    "report.section7.table.type": "Type",
    "report.section7.table.technique": "Technique",
    "report.section7.table.strength": "Strength",
    "report.section7.table.p_value": "p-value",
    "report.section7.excluded_subtitle": "Predictors excluded from the analysis",
    "report.section7.table.reason": "Reason",
    "report.section7.leakage_warning": "⚠️ **Possible data leakage detected** — see section 8.",

    "report.section8.title": "8. Recommendations",
    "report.section8.json_link": "Full version in JSON:",
    "report.section8.severity_high": "High severity",
    "report.section8.severity_medium": "Medium severity",
    "report.section8.severity_low": "Low severity",
    "report.section8.table.feature": "Feature",
    "report.section8.table.problem": "Problem",
    "report.section8.table.recommendation": "Recommendation",
    "report.section8.dataset_placeholder": "_dataset_",

    "report.no_data": "_No data for this section._",
}
