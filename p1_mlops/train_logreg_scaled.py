import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from data_loader import load_data, time_split
from features import build_features
from config import DATA_PATH

df = load_data(DATA_PATH)
train_df, val_df = time_split(df)

x_train, y_train = build_features(train_df)
x_val, y_val = build_features(val_df)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000))
])

with mlflow.start_run(run_name="logreg_scaled"):
    pipeline.fit(x_train,y_train)

    probs = pipeline.predict_proba(x_val)[:,1]
    roc = roc_auc_score(y_val, probs)
    precision,recall, _ = precision_recall_curve(y_val, probs)
    pr_auc = auc(recall, precision)

    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("pr_auc", pr_auc)

    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")


