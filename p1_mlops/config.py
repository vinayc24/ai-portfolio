"""
Docstring for p1_mlops.config
Central configuration file for project 1(Fraud detection)

All constants are defined here so that:

- Hyperparameters are easy to change
- Paths are centralized 
-Training logic remains clean and reusable
"""





DATA_PATH = "..\common\data\creditcard.csv"
TARGET_COL = "Class"
TIME_COL = "Time"
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42