import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizerFast
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

        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.dropout = nn.Dropout(hidden_dropout_prob)

        self.heads = nn.ModuleDict({
            task: nn.Linear(hidden_size, 1) for task in self.tasks
        })

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # CLS token representation
        pooled = outputs.last_hidden_state[:, 0, :]  # [B, H]
        pooled = self.dropout(pooled)

        logits = {task: self.heads[task](pooled) for task in self.tasks}  # each [B, 1]
        return logits

    def freeze_encoder(self):
        for p in self.roberta.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.roberta.parameters():
            p.requires_grad = True


class MultiTaskDataset(Dataset):
    """
    Dataset for 2-head or 4-head multitask training.

    IMPORTANT:
    - Labels are returned as float tensors (0.0/1.0) for BCEWithLogitsLoss.
    - Each label is shape [1] so batching yields [B, 1].
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
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.loc[idx, "text"])

        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),         # [L]
            "attention_mask": enc["attention_mask"].squeeze(0) # [L]
        }

        # Labels as float (for BCEWithLogitsLoss)
        for task in self.tasks:
            col = TASK_TO_COL[task]
            val = self.data.loc[idx, col]

            # Handle NaNs defensively (should already be removed in preprocessing)
            if val != val:  # NaN check
                raise ValueError(f"NaN label found at idx={idx} task={task} col={col}")

            item[f"labels_{task}"] = torch.tensor([float(val)], dtype=torch.float)  # [1]

        return item
