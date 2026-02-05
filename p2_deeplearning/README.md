# Transformer-Based Text Classification (CPU-Only)

## Overview
This project implements a **transformer-based text classification system** using a pretrained **DistilBERT** model.  
The goal was to build, train, and evaluate a deep learning model **locally on CPU**, while demonstrating correct use of **transfer learning, evaluation metrics, and error analysis**.

The model is trained and evaluated on the **AG News** dataset, which consists of short news headlines across four categories.

---

## Problem Statement
Given a news headline, predict its category:
- World
- Sports
- Business
- Science & Technology

This is a **multi-class text classification** problem with inherent semantic overlap between certain classes.

---

## Model Choice
- **DistilBERT (distilbert-base-uncased)** was selected due to:
  - Strong pretrained language representations
  - Reduced size and faster inference compared to BERT
  - Practical suitability for CPU-only training

---

## Training Strategy
1. **Frozen Backbone Training**
   - All transformer layers frozen
   - Only the classification head trained
   - Efficient and stable training on CPU

2. **Partial Fine-Tuning Experiment**
   - Unfroze only the **last transformer layer**
   - Compared performance against the frozen baseline
   - Used to study trade-offs between capacity and stability

---

## Evaluation Metrics
- Accuracy
- Precision, Recall, F1-score (per class)
- Confusion Matrix

### Final Performance (Frozen Model)
- **Accuracy:** ~86%
- Strong performance on Sports news
- Most errors occurred between **World** and **Business** categories

---

## Error Analysis
The primary source of misclassification was between **World** and **Business** news.  
This is expected due to:
- Semantic overlap in economic and geopolitical headlines
- Limited context from short headline-only inputs

The model’s errors were **semantically reasonable**, indicating meaningful learned representations.

---

## Key Findings
- Frozen transformers can perform extremely well on small datasets
- Partial fine-tuning did not improve performance under CPU and data constraints
- Stability and efficiency were prioritized over marginal gains

---

## Project Structure
p2_deeplearning/
├── config.py # Training configuration
├── dataset.py # Dataset loading and tokenization
├── model.py # Model definition and freezing logic
├── train.py # Training loop
├── evaluate.py # Evaluation and confusion matrix
└── README.md



---

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Scikit-learn
- Matplotlib / Seaborn

---

## Key Learnings
- Practical transfer learning with transformers
- CPU-aware deep learning training
- Model comparison and selection
- Confusion-matrix–driven error analysis
- Clean and reproducible ML project structuring
