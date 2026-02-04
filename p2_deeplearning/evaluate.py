import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report

from dataset import get_datasets
from model import get_model
from config import DEVICE, BATCH_SIZE

_, val_ds = get_datasets()
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = get_model().to(DEVICE)
model.load_state_dict(torch.load("model.pt"))
model.eval()


preds, labels = [],[]

with torch.no_grad():
    for batch in val_loader:
        outputs = model(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"]

        )
        predictions = outputs.logits.argmax(dim =1)
        preds.extend(predictions.tolist())
        labels.extend(batch["label"].tolist())


print("Accuracy:", accuracy_score(labels, preds))
print(classification_report(labels, preds))