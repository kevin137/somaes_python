"""
somaes_python
Functions created for the final project in Software matemático y estadístico
"""

from .functions import ( discretizeEW, discretizeEF, 
           discretize_EW_by_column, discretize_EF_by_column,
           discretize_EW_EF_by_column,
           calculate_variance, calculate_roc_auc, calculate_entropy,
           dataset_metrics_summary, select_variables_by_metrics,
           normalize_by_column, standardize_by_column,
           correlation, mutual_information,
           column_relationships, plot_relationships,
           plot_roc_auc
)

from .reference_datasets import (
    titanic_df,
    p03_disc_values,
    p03_disc_bins,
    sample_variance_23p5_df,
    pop_variance_2p917_df,
    auc_1p0_df,
    auc_0p75_df,
    p03_entropy_0p971,
    std_mixed_df,
    corr_m0p685_df,
)

__all__ = [
    'discretizeEW', 'discretizeEF', 
    'discretize_EW_by_column', 'discretize_EF_by_column',
    'discretize_EW_EF_by_column',
    'calculate_variance', 'calculate_roc_auc', 'calculate_roc_auc',
    'dataset_metrics_summary', 'select_variables_by_metrics',
    'normalize_by_column', 'standardize_by_column',
    'correlation', 'mutual_information',
    'column_relationships', 'plot_relationships',
    'plot_roc_auc'
    'titanic_df',
    'p03_disc_values',
    'p03_disc_bins',
    'sample_variance_23p5_df',
    'pop_variance_2p917_df',
    'auc_1p0_df',
    'auc_0p75_df',
    'p03_entropy_0p971',
    'std_mixed_df',
    'corr_m0p685_df',
]

