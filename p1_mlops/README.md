
---

# 2️⃣ **Project 1 — `p1_mlops/README.md`**

---

```md
# Project 1 — Fraud Detection (ML + MLOps)

This project demonstrates a **production-style machine learning pipeline** for detecting fraudulent transactions using classical ML models and MLOps best practices.

---

## 🎯 Problem Statement

Given transaction-level features, predict whether a transaction is **fraudulent or legitimate**.

The goal is not just high accuracy, but:
- Robust evaluation
- Reproducibility
- Deployability

---

## 🧠 What This Project Covers

- Data loading & feature preparation
- Baseline model (Logistic Regression)
- Improved models (XGBoost)
- Proper train/validation splitting
- ROC-AUC & PR-AUC evaluation
- Experiment tracking with MLflow
- FastAPI inference endpoint
- Dockerized deployment

---

## 📁 Project Structure

```text
p1_mlops/
├── train_baseline.py        # Baseline ML model
├── train_logreg_scaled.py   # Scaled logistic regression
├── train_xgboost.py         # XGBoost training
├── inference_api.py         # FastAPI inference service
├── features.py              # Feature processing logic
├── data_loader.py           # Data loading utilities
├── schema.py                # Input validation schema
├── config.py                # Centralized config
├── Dockerfile
├── requirements.txt
└── README.md


🚀 How to Run
Train a model
python train_xgboost.py

Start inference API
uvicorn inference_api:app --reload

Dockerized run
docker build -t fraud-ml-service .
docker run -p 8000:8000 fraud-ml-service

📊 Key Metrics Used

ROC-AUC

PR-AUC (important for imbalanced data)

💡 Key Takeaways

Emphasis on model evaluation over accuracy

Clear separation of training and inference

Real-world MLOps considerations