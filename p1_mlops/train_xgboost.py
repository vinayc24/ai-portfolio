import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from data_loader import load_data, time_split
from features import build_features
from config import DATA_PATH
import joblib

df = load_data(DATA_PATH)
train_df, val_df = time_split(df)

x_train, y_train = build_features(train_df)
x_val, y_val = build_features(val_df)

model = xgb.XGBClassifier(
    n_estimators =200,
    max_depth =5,
    learning_rate =0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    eval_metric="logloss",
    random_state=42
)

with mlflow.start_run(run_name="xgboost"):
    model.fit(x_train, y_train)

    #SAVE MODEL LOCALLY
    joblib.dump(model, "model.joblib")

    probs = model.predict_proba(x_val)[:,1]
    roc =  roc_auc_score(y_val, probs)
    precision,recall, _ = precision_recall_curve(y_val, probs)
    pr_auc = auc(recall, precision)

    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("pr_auc", pr_auc)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")   