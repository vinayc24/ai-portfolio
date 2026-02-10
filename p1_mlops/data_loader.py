"""
Docstring for p1_mlops.data_loader
Responsible for:
-Loading raw CSV data
-Sorting by time to prevent data leakage
-Performing time-based train/validation split

"""

import pandas as pd
from config import TIME_COL, TRAIN_SPLIT

def load_data(path):
    """
    Docstring for load_data
    Loads dataset from CSV and sort by time
    sorting by time is critival to:
        -Simulate real world deployment
        - Avoid training on future information

    :param path: path to csv

    Returns:
        pd.Dataframe: sorted dataset
    """
    df = pd.read_csv(path)
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    return df

def time_split(df):
    """
    Docstring for time_split
        Splits dataset into train and validation sets using time-based split.


    :param df: Full dataset

    Returns:
        train_df (pd.dataframe): Earlier data
        val_df (pd.dataframe): Later data
    """

    split_idx = int(len(df)*TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    return train_df, val_df