import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler

from dataset import get_datasets
from model import get_model
from config import DEVICE, BATCH_SIZE, EPOCHS, LR

train_ds, val_ds = get_datasets()
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = get_model().to(DEVICE)

#Freeze Backbone
for param in model.base_model.parameters():
    param.requires_grad = False

optimizer = AdamW(model.parameters(), lr = LR)
num_training_steps = EPOCHS* len(train_loader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCHS):
    total_loss=0
    for batch in train_loader:
        optimizer.zero_grad()

        outputs = model(
            input_ids = batch["input_ids"],
            attention_mask = batch["attention_mask"],
            labels=batch["label"]
        )
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "model.pt")