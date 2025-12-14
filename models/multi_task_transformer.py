import torch
import torch.nn as nn
from transformers import RobertaModel, AutoTokenizer
from torch.utils.data import Dataset


DEFAULT_TASKS = ["Q_overall", "Q2_harmful", "Q3_bias", "Q6_policy"]

TASK_TO_COL = {
    "Q_overall": "Q_overall_binary",
    "Q2_harmful": "Q2_harmful_binary",
    "Q3_bias": "Q3_bias_binary",
    "Q6_policy": "Q6_policy_binary",
}


class MultiTaskRoBERTa(nn.Module):
    """
    Shared RoBERTa encoder + separate heads per task.

    For binary tasks:
        - Each head outputs a single logit per example: shape [B, 1]
        - Use BCEWithLogitsLoss in training code
    """
    def __init__(
        self,
        model_name: str = "roberta-base",
        tasks=None,
        hidden_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.tasks = tasks if tasks is not None else DEFAULT_TASKS

        # Shared encoder (RoBERTa base model)
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        # Define Dropout (for regularization)
        self.dropout = nn.Dropout(hidden_dropout_prob)

        # Task-specific classification heads (one linear layer per task)
        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, 1) for task in self.tasks
        })

    def forward(self, input_ids, attention_mask):
        # Encode input
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]  
        pooled = self.dropout(pooled)

        # Pass through each task-specific head
        logits = {task: self.heads[task](pooled) for task in self.tasks} 
        return logits

class MultiTaskDataset(Dataset):
    """
    Dataset for 2-head or 4-head multitask training
    """
    def __init__(self, dataframe, tokenizer, max_length=256, tasks=None):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tasks = tasks if tasks is not None else DEFAULT_TASKS

        # Validate tasks
        for t in self.tasks:
            if t not in TASK_TO_COL:
                raise ValueError(f"Unknown task '{t}'. Must be one of {list(TASK_TO_COL.keys())}")

    def __len__(self):
        """
        Return number of examples in dataset
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single training example
        """
        text = str(self.data.loc[idx, "text"])

        # Tokenize text
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Start with tokenized inputs
        item = {
            "input_ids": enc["input_ids"].squeeze(0),         # [L]
            "attention_mask": enc["attention_mask"].squeeze(0) # [L]
        }

        # Add labels for each task as float tensors
        for task in self.tasks:
            col = TASK_TO_COL[task]
            val = self.data.loc[idx, col]

            # HDefensive check for NaN
            if val != val:  
                raise ValueError(f"NaN label found at idx={idx} task={task} col={col}")

            # Store as [1] shape for proper batching
            item[f"labels_{task}"] = torch.tensor([float(val)], dtype=torch.float)  

        return item


def load_model(ckpt_path: str, device: torch.device):
    """
    Load a trained MultiTaskRoBERTa model
    """
    # Load from checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    model_name = ckpt.get("model_name", "roberta-base")
    tasks = ckpt["tasks"]

    # Define Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define Model
    model = MultiTaskRoBERTa(
        model_name=model_name,
        tasks=tasks,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, tokenizer