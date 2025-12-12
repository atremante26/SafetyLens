import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset

class MultiTaskRoBERTa(nn.Module):
    '''
    Input Text -> ROBERTa Encoder (shared) -> [CLS] token -> 4 separate heads -> 4 predictions
    Tasks:
        - Q_overall
        - Q2_harmful_content_overall
        - Q3_bias_overall
        - Q6_policy_guidelines_overall
    '''
    def __init__(self, num_labels=2):
        super(MultiTaskRoBERTa, self).__init__()
        
        # Shared encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
        # 4 classification heads
        self.overall_head = nn.Linear(768, num_labels)
        self.harmful_content_head = nn.Linear(768, num_labels)
        self.bias_head = nn.Linear(768, num_labels)
        self.policy_guidelines_head = nn.Linear(768, num_labels)
    
    def forward(self, input_ids, attention_mask):
        # Get embeddings
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Get predictions for each task
        logits_overall = self.overall_head(pooled_output)
        logits_harmful = self.harmful_content_head(pooled_output)
        logits_bias = self.bias_head(pooled_output)
        logits_policy = self.policy_guidelines_head(pooled_output)
        
        return {
            'Q_overall': logits_overall,
            'Q2_harmful': logits_harmful,
            'Q3_bias': logits_bias,
            'Q6_policy': logits_policy
        }


class MultiTaskDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        '''
        Dataset for multi-task learning with 4 target variables
        '''
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        # Tells DataLoader how many examples exist
        return len(self.data)
    
    def __getitem__(self, idx):
        # Gets text at position idx
        text = str(self.data.iloc[idx]['text'])
        
        # Binary labels
        label_overall = int(self.data.iloc[idx]['Q_overall_binary'])
        label_harmful = int(self.data.iloc[idx]['Q2_harmful_binary'])
        label_bias = int(self.data.iloc[idx]['Q3_bias_binary'])
        label_policy = int(self.data.iloc[idx]['Q6_policy_binary'])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels_overall': torch.tensor(label_overall, dtype=torch.long),
            'labels_harmful': torch.tensor(label_harmful, dtype=torch.long),
            'labels_bias': torch.tensor(label_bias, dtype=torch.long),
            'labels_policy': torch.tensor(label_policy, dtype=torch.long)
        }