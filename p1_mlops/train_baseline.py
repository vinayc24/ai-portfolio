"""
Docstring for p1_mlops.train_baseline
    Trains a baseline Logistic Regression model without feature scaling

    Purpose:
        -Establish a simple benchmark
        -validate the pipeline end-to-end
"""


import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from data_loader import load_data, time_split
from features import build_features
from config import DATA_PATH

# Load and split data
df = load_data(DATA_PATH)
train_df, val_df = time_split(df)

# Build features
x_train, y_train = build_features(train_df)
x_val, y_val = build_features(val_df)

# Track experiment with MLflow
with mlflow.start_run(run_name="logreg_baseline"):

    #Initialise and train model
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train)
    
    #Predict probabilities for validation set
    val_probs = model.predict_proba(x_val)[:,1]


    #Compute Evaluation metrics
    roc = roc_auc_score(y_val, val_probs)
    precision, recall, _= precision_recall_curve(y_val, val_probs)
    pr_auc = auc(recall, precision)


    #Log metrics
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("pr_auc", pr_auc)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")