import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Configuration
MODEL_NAME = "roberta-base"
NUM_LABELS = 2

# Label mappings for model output
label2id = {
    "safe": 0,
    "unsafe": 1,
}

# Reverse mapping
id2label = {v: k for k, v in label2id.items()}

class SingleTaskDataset(Dataset):
    """
    Dataset for single-task classification with RoBERTa.
    Each item returns input_ids, attention_mask, and a single label.
    """
    def __init__(self, data, tokenizer, labels=None, label_col="Q_overall_binary", max_length=256):
        # Handle DataFrame input
        if isinstance(data, pd.DataFrame):
            if label_col not in data.columns:
                raise ValueError(
                    f"Label column '{label_col}' not found in DataFrame. "
                    f"Available columns: {data.columns.tolist()}"
                )
            self.texts = data["text"].values
            self.labels = data[label_col].astype(int).values
            
        # Handle array/list input
        else:
            if labels is None:
                raise ValueError("labels must be provided when data is not a DataFrame")
            
            # Convert to numpy arrays for consistency
            self.texts = np.asarray(data)
            self.labels = np.asarray(labels).astype(int)
            
            if len(self.texts) != len(self.labels):
                raise ValueError(f"Mismatch: {len(self.texts)} texts but {len(self.labels)} labels")
        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return number of examples in dataset
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        Get a single training example
        """
        # Ensure types
        text = str(self.texts[idx])
        label_id = int(self.labels[idx])

        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label_id, dtype=torch.long),
        }

def load_model(model_name=MODEL_NAME, device="cuda"):
    """
    Load the RoBERTa model and tokenizer.
    """
    # Define device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    model.to(device)
    model.eval() 

    return model, tokenizer

def load_finetuned(model_path, device="cuda"):
    """
    Load the finetuned Roberta model and tokenizer.
    """
    # Define Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Define Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    # Load model from checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handles both checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()
    return model, tokenizer
        