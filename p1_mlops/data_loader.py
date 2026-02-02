import pandas as pd
from config import TIME_COL, TRAIN_SPLIT

def load_data(path):
    df = pd.read_csv(path)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df

def time_split(df):
    split_idx = int(len(df)*TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    return train_df, val_df