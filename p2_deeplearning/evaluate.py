"""
evaluate.py

Evaluates the trained text classification model.
Outputs:
- Accuracy
- Classification report
- Confusion matrix
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import get_datasets
from model import get_model
from config import DEVICE


def evaluate():
    """
    Runs evaluation on the test dataset and prints metrics.
    Also plots a confusion matrix for deeper error analysis.
    """

    # Load test dataset
    _, test_dataset = get_datasets()

    # Load trained model
    model = get_model()
    model.load_state_dict(torch.load("model.pt", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    all_preds = []
    all_labels = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in test_dataset:
            inputs = {
                "input_ids": batch["input_ids"].unsqueeze(0).to(DEVICE),
                "attention_mask": batch["attention_mask"].unsqueeze(0).to(DEVICE),
            }

            # Forward pass
            outputs = model(**inputs)
            logits = outputs.logits

            # Predicted class
            pred = torch.argmax(logits, dim=1).item()

            all_preds.append(pred)
            all_labels.append(batch["label"])

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Print standard metrics
    print("Accuracy:", accuracy_score(all_labels, all_preds))
    print(classification_report(all_labels, all_preds))

    # -------------------------------
    # Confusion Matrix
    # -------------------------------
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["World", "Sports", "Business", "Sci/Tech"],
        yticklabels=["World", "Sports", "Business", "Sci/Tech"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - AG News")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    evaluate()
