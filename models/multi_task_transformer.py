import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset

class MultiTaskRoBERTa(nn.Module):
    '''
    Input Text -> ROBERTa Encoder (shared) -> [CLS] token -> 2 or 4 separate heads -> 2 or 4 predictions
    
    Supports both:
    - Binary classification with BCEWithLogitsLoss (num_labels=1)
    - Multi-class with CrossEntropyLoss (num_labels=2+)
    
    Tasks:
        - Q_overall
        - Q2_harmful
        - Q3_bias
        - Q6_policy
    '''
    def __init__(self, num_labels=1, tasks=None):
        super(MultiTaskRoBERTa, self).__init__()
        
        # Default to all 4 tasks if not specified
        if tasks is None:
            tasks = ['Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy']
        
        self.tasks = tasks
        self.num_labels = num_labels
        
        # Shared encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # Create only the heads needed
        self.heads = nn.ModuleDict()
        for task in tasks:
            self.heads[task] = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each active task
        logits = {}
        for task in self.tasks:
            output = self.heads[task](pooled_output)
            # Only squeeze if binary classification (num_labels=1)
            if self.num_labels == 1:
                logits[task] = output.squeeze(-1)  # [batch_size, 1] -> [batch_size]
            else:
                logits[task] = output  # Keep [batch_size, num_labels]
        
        return logits


class MultiTaskDataset(Dataset):
    '''
    Flexible dataset that supports 2 or 4 tasks
    '''
    def __init__(self, dataframe, tokenizer, max_length=512, tasks=None):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Default to all 4 tasks if not specified
        if tasks is None:
            tasks = ['Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy']
        
        self.tasks = tasks
        
        # Map task names to column names
        self.task_to_col = {
            'Q_overall': 'Q_overall_binary',
            'Q2_harmful': 'Q2_harmful_binary',
            'Q3_bias': 'Q3_bias_binary',
            'Q6_policy': 'Q6_policy_binary'
        }
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['text'])
        
        # Get labels for active tasks only
        labels = {}
        for task in self.tasks:
            col_name = self.task_to_col[task]
            labels[f'labels_{task}'] = torch.tensor(
                int(self.data.iloc[idx][col_name]), 
                dtype=torch.long
            )
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        result.update(labels)
        
        return result