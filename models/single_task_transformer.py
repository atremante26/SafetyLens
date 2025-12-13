import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "roberta-base"
NUM_LABELS = 2

label2id = {
    "safe": 0,
    "unsafe": 1,
}
id2label = {v: k for k, v in label2id.items()}

class HateSpeechDataset(Dataset):
    """
    Dataset for single-task classification with RoBERTa.
    Each item returns input_ids, attention_mask, and a single label.
    """
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.texts = dataframe["text"].values
        self.labels = dataframe["label"].map(label2id).values  # map strings -> ids
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label_id = self.labels[idx]

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
    Load a RoBERTa sequence classification model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_LABELS,
        id2label=id2label,
        label2id=label2id,
    )

    model.to(device)
    model.eval() 

    return model, tokenizer
        
print("hello world")