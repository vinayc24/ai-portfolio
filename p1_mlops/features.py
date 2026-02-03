from config import TARGET_COL
from schema import FEATURE_COLUMNS

def build_features(df):
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COL]
    return x,y

