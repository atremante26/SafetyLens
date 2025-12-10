import torch
import torch.nn as nn 
import numpy as np
from transformers import RobertaModel, RobertaTokenizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset

# Model Config
NUM_LABELS = 3
MODEL_NAME = 'roberta-3'

class MultiTaskRoBERTa(nn.Module):
    '''
    Input Text -> ROBERTa Encoder (shared) -> [CLS] token -> 4 separate heads -> 4 predictions
    Tasks:
        - Q_overall
        - Q2_harmful_content_overall
        - Q3_bias_overall
        - Q6_policy_guidelines_overall
    '''
    def __init__(self, model_name=MODEL_NAME, num_labels=NUM_LABELS, dropout=0.1):
        super().__init__()
        # Shared encoder
        self.roberta = RobertaModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size 

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 4 classification heads (1 per task)
        self.overall_head = nn.Linear(hidden_size, num_labels)
        self.harmful_content_head = nn.Linear(hidden_size, num_labels)
        self.bias_head = nn.Linear(hidden_size, num_labels)
        self.policy_guidelines_head = nn.Linear(hidden_size, num_labels)

        # Task names
        self.task_names = ['Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy']

    def forward(self, input_ids, attention_mask):
        # Shared encoding
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)

        # Get [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Return 4 predictions
        return {
            'Q_overall': self.overall_head(cls_output),              
            'Q2_harmful': self.harmful_content_head(cls_output),     
            'Q3_bias': self.bias_head(cls_output),               
            'Q6_policy': self.policy_guidelines_head(cls_output)    
        }

class MultiTaskDataset(Dataset):
    '''
    Dataset for multi-task learning with 4 target variables
    '''
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.texts = dataframe['text'].values
        self.labels_overall = dataframe['Q_overall_3class'].values
        self.labels_harmful = dataframe['Q2_harmful_content_overall_3class'].values
        self.labels_bias = dataframe['Q3_bias_overall_3class'].values
        self.labels_policy = dataframe['Q6_policy_guidelines_overall_3class'].values
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_overall': torch.tensor(self.labels_overall[idx], dtype=torch.long),
            'labels_harmful': torch.tensor(self.labels_harmful[idx], dtype=torch.long),
            'labels_bias': torch.tensor(self.labels_bias[idx], dtype=torch.long),
            'labels_policy': torch.tensor(self.labels_policy[idx], dtype=torch.long)
        }


def load_model(model_path=None, device='cuda'):
    '''
    Load trained multi-task model
    '''
    model = MultiTaskRoBERTa(num_labels=NUM_LABELS)
    
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    
    model.to(device)
    model.eval()
    
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    return model, tokenizer


def weighted_loss_functions(train_df, device='cuda'):
    '''
    Create weighted cross-entropy loss functions for each task (addresses class imbalance)
    '''
    task_columns = {
        'Q_overall': 'Q_overall_3class',
        'Q2_harmful': 'Q2_harmful_content_overall_3class',
        'Q3_bias': 'Q3_bias_overall_3class',
        'Q6_policy': 'Q6_policy_guidelines_overall_3class'
    }
    
    weighted_losses = {}
    
    for task_key, col_name in task_columns.items():
        # Get label distribution
        labels = train_df[col_name].values
        
        # Compute balanced class weights
        weights = compute_class_weight(
            'balanced',
            classes=np.array([0, 1, 2]),
            y=labels
        )
        
        # Convert to tensor
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        
        # Create weighted loss
        weighted_losses[task_key] = nn.CrossEntropyLoss(weight=weights_tensor)

    return weighted_losses