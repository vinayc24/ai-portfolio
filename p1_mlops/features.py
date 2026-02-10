"""
Docstring for p1_mlops.features
    Handles feature-target separation
    This ensures-
        -Models never accidentally see the target columns
        -Feature selection is consistent across experiments
"""

from config import TARGET_COL
from schema import FEATURE_COLUMNS

def build_features(df):
    """
    Docstring for build_features
        Separate input features and target label

    
    :param df: Input dataframe

    Returns:
        x (pd.dataframe): Feature matrix
        y(pd.series): Target labels
    """
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]
    return x,y

