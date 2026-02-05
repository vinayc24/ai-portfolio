"""
model.py

Responsible for:
-Loading a pretrained DistilBERT model
- Freezing most layers for efficient CPU training
- Optionally unfreezing the last transformer layer for fine-tuning

"""



from transformers import AutoModelForSequenceClassification
from config import MODEL_NAME, NUM_CLASSES

def get_model(unfreeze_last_layer: bool = True):
    '''
    Loads a DistilBERT model for sequence classification:

    Args:
        nums_labels: Number of output classes.
        unfreeze_last_layer(bool): Whether to unfreeze the last ltransformer layer or not

    Returns:
        model: Higging Face transformer model ready for training
    '''
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = NUM_CLASSES
    )
    # ---------------------------------------------------------
    # STEP 1: Freeze ALL model parameters
    # ---------------------------------------------------------
    # This prevents the base language model from updating,
    # making training faster and more stable on CPU.
    for param in model.parameters():
        param.requires_grad = False

    # ---------------------------------------------------------
    # STEP 2: Always train the classification head
    # ---------------------------------------------------------
    # The classifier layer is newly initialized and must be trained.
    
    for param in model.pre_classifier.parameters():
        param.requires_grad = True

    for param in model.pre_classifier.parameters():
        param.requires_grad = True

    # ---------------------------------------------------------
    # STEP 3: Optionally unfreeze ONLY the last transformer layer
    # ---------------------------------------------------------

    if unfreeze_last_layer:
        # DistilBERT has 6 transformer layers (0 to 5)
        # We unfreeze ONLY the last one (layer 5)        
        for param in model.distilbert.transformer.layer[-1].parameters():
            param.requires_grad = True

    return model