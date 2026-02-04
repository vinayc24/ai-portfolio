from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME, NUM_CLASSES

def get_model():
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_CLASSES
    )
    return model