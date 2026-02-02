from config import TARGET_COL

def build_features(df):
    x = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return x,y

